"""One interface in front of every model provider. Swapping providers is config-only.

    provider = make_provider("gemini")
    r = provider.generate(system_text, user_text)
    r.text, r.raw, r.finish_reason, r.in_tokens, r.out_tokens

The provider is handed two flat strings and returns a Reply. It never builds prompts, never
knows what a persona is, and never parses a rating - those stay in src/prompts.py, which is
unchanged by the provider switch.

Also here: a rate limiter (RPM/TPM/RPD) and exponential backoff with jitter on 429 and 5xx.
"""
import os, time, random, sys
from dataclasses import dataclass, field
sys.path.insert(0, "src")
import phase2_config as CFG


@dataclass
class Reply:
    text: str | None
    raw: dict
    finish_reason: str | None = None
    in_tokens: int | None = None
    out_tokens: int | None = None
    error: str | None = None
    attempts: int = 1
    headers: dict = field(default_factory=dict)


class RateLimitHit(Exception):
    """Retryable: 429 or 5xx."""
    def __init__(self, msg, retry_after=None):
        super().__init__(msg)
        self.retry_after = retry_after


class DailyQuotaExhausted(Exception):
    """Not retryable within this run. The runner stops cleanly so a resume can pick up."""


# --------------------------------------------------------------------------- rate limiting
class RateLimiter:
    """Spaces requests to respect RPM, and tracks daily request/token budgets.

    Deliberately simple and conservative: it paces by wall-clock gap rather than a sliding
    window, so it can never burst over RPM. Daily counters are per-process; on resume the
    runner reports how many calls the previous sessions already spent today.
    """

    def __init__(self, rpm=None, tpm=None, rpd=None, tpd=None, spent_today=0):
        self.rpm, self.tpm, self.rpd, self.tpd = rpm, tpm, rpd, tpd
        self.min_gap = 60.0 / rpm if rpm else 0.0
        self._last = 0.0
        self.n_today = spent_today
        self.tok_today = 0
        self._minute_start = time.monotonic()
        self._tok_this_minute = 0

    def before(self, est_tokens=0):
        if self.rpd and self.n_today >= self.rpd:
            raise DailyQuotaExhausted(
                f"daily request limit reached: {self.n_today}/{self.rpd}")
        if self.tpd and self.tok_today + est_tokens > self.tpd:
            raise DailyQuotaExhausted(
                f"daily token limit reached: {self.tok_today}/{self.tpd}")
        # token-per-minute pacing
        if self.tpm:
            now = time.monotonic()
            if now - self._minute_start >= 60.0:
                self._minute_start, self._tok_this_minute = now, 0
            if self._tok_this_minute + est_tokens > self.tpm:
                sleep = 60.0 - (now - self._minute_start)
                if sleep > 0:
                    time.sleep(sleep)
                self._minute_start, self._tok_this_minute = time.monotonic(), 0
        gap = time.monotonic() - self._last
        if gap < self.min_gap:
            time.sleep(self.min_gap - gap)

    def after(self, in_tokens=0, out_tokens=0):
        self._last = time.monotonic()
        self.n_today += 1
        tot = (in_tokens or 0) + (out_tokens or 0)
        self.tok_today += tot
        self._tok_this_minute += tot


def _backoff_sleep(attempt, retry_after=None):
    if retry_after:
        d = float(retry_after)
    else:
        d = min(CFG.RETRY_BASE_SECONDS * (2 ** (attempt - 1)), CFG.RETRY_CAP_SECONDS)
    d *= 1.0 + random.uniform(-CFG.RETRY_JITTER, CFG.RETRY_JITTER)
    time.sleep(max(0.5, d))
    return d


# ------------------------------------------------------------------------------- providers
class BaseProvider:
    name = "base"

    def __init__(self, cfg):
        self.cfg = cfg
        self.model = cfg["model"]
        self.limiter = RateLimiter(cfg.get("rpm"), cfg.get("tpm"),
                                   cfg.get("rpd"), cfg.get("tpd"))
        self._resolved_version = None
        self.last_headers = {}

    def _key(self):
        for env in self.cfg["api_key_env"]:
            v = os.environ.get(env)
            if v:
                return v
        raise SystemExit(
            f"NO API KEY FOUND for provider {self.cfg['provider']!r}.\n"
            f"  Looked for: {', '.join(self.cfg['api_key_env'])}\n"
            f"  This provider has a free tier; get a key and export it, e.g.\n"
            f"      export {self.cfg['api_key_env'][0]}=...\n"
            f"  No script here will guess at or invent credentials.")

    def version_string(self):
        """Exact version string for the paper's data section. Prefers what the API
        actually reports over what we asked for."""
        return self._resolved_version or self.model

    def _call(self, system_text, user_text, temperature, max_tokens):
        raise NotImplementedError

    def generate(self, system_text, user_text, temperature=None, max_tokens=None,
                 est_tokens=0):
        temperature = CFG.TEMPERATURE if temperature is None else temperature
        max_tokens = CFG.MAX_OUTPUT_TOKENS if max_tokens is None else max_tokens
        last = None
        for attempt in range(1, CFG.RETRY_MAX_ATTEMPTS + 1):
            try:
                self.limiter.before(est_tokens)
            except DailyQuotaExhausted:
                raise
            try:
                r = self._call(system_text, user_text, temperature, max_tokens)
                r.attempts = attempt
                self.limiter.after(r.in_tokens, r.out_tokens)
                return r
            except DailyQuotaExhausted:
                raise
            except RateLimitHit as e:
                last = f"{type(e).__name__}: {e}"
                self.limiter.after(0, 0)
                if attempt == CFG.RETRY_MAX_ATTEMPTS:
                    break
                _backoff_sleep(attempt, e.retry_after)
            except Exception as e:                      # non-retryable, recorded not raised
                self.limiter.after(0, 0)
                return Reply(None, {}, error=f"{type(e).__name__}: {e}", attempts=attempt)
        return Reply(None, {}, error=f"retries exhausted; last: {last}",
                     attempts=CFG.RETRY_MAX_ATTEMPTS)


class GeminiProvider(BaseProvider):
    name = "gemini"

    def __init__(self, cfg):
        super().__init__(cfg)
        from google import genai
        self._genai = genai
        self.client = genai.Client(api_key=self._key())

    def _call(self, system_text, user_text, temperature, max_tokens):
        from google.genai import types
        from google.genai import errors as gerr
        try:
            resp = self.client.models.generate_content(
                model=self.model,
                contents=user_text,
                config=types.GenerateContentConfig(
                    system_instruction=system_text,
                    temperature=temperature,
                    max_output_tokens=max_tokens,
                    # No thinking: this is a one-word annotation task, and a thinking budget
                    # would consume max_output_tokens before any answer is emitted.
                    thinking_config=types.ThinkingConfig(thinking_budget=0),
                ),
            )
        except gerr.APIError as e:
            code = getattr(e, "code", None)
            if code == 429 or (code is not None and 500 <= int(code) < 600):
                raise RateLimitHit(str(e))
            raise
        raw = resp.model_dump(mode="json") if hasattr(resp, "model_dump") else {}
        um = raw.get("usage_metadata") or {}
        cands = raw.get("candidates") or []
        finish = cands[0].get("finish_reason") if cands else None
        self._resolved_version = raw.get("model_version") or self.model
        text = None
        try:
            text = resp.text
        except Exception:
            text = None
        return Reply(text, raw, finish_reason=finish,
                     in_tokens=um.get("prompt_token_count"),
                     out_tokens=um.get("candidates_token_count"))


class GroqProvider(BaseProvider):
    name = "groq"

    def __init__(self, cfg):
        super().__init__(cfg)
        import groq
        self._groq = groq
        self.client = groq.Groq(api_key=self._key())

    # Groq does not document when the daily buckets reset, but it returns the reset time on
    # every response. Capturing these lets the run learn the reset empirically instead of
    # assuming midnight UTC. See RATE_HEADERS in the quota report.
    RATE_HEADERS = ("x-ratelimit-limit-requests", "x-ratelimit-remaining-requests",
                    "x-ratelimit-reset-requests", "x-ratelimit-limit-tokens",
                    "x-ratelimit-remaining-tokens", "x-ratelimit-reset-tokens",
                    "retry-after")

    def _headers(self, hdrs):
        if not hdrs:
            return {}
        return {k: hdrs.get(k) for k in self.RATE_HEADERS if hdrs.get(k) is not None}

    def _call(self, system_text, user_text, temperature, max_tokens):
        try:
            raw_resp = self.client.chat.completions.with_raw_response.create(
                model=self.model,
                messages=[{"role": "system", "content": system_text},
                          {"role": "user", "content": user_text}],
                temperature=temperature,
                max_completion_tokens=max_tokens,
            )
            headers = self._headers(getattr(raw_resp, "headers", None))
            resp = raw_resp.parse()
        except self._groq.RateLimitError as e:
            hdrs = getattr(getattr(e, "response", None), "headers", None)
            h = self._headers(hdrs)
            self.last_headers = h
            raise RateLimitHit(str(e), retry_after=h.get("retry-after"))
        except self._groq.InternalServerError as e:
            raise RateLimitHit(str(e))
        raw = resp.model_dump(mode="json") if hasattr(resp, "model_dump") else {}
        ch = (raw.get("choices") or [{}])[0]
        us = raw.get("usage") or {}
        self._resolved_version = raw.get("model") or self.model
        self.last_headers = headers
        return Reply((ch.get("message") or {}).get("content"), raw,
                     finish_reason=ch.get("finish_reason"),
                     in_tokens=us.get("prompt_tokens"),
                     out_tokens=us.get("completion_tokens"),
                     headers=headers)


REGISTRY = {"gemini": GeminiProvider, "groq": GroqProvider, "groq-gptoss": GroqProvider}


def make_provider(name=None):
    cfg = CFG.profile(name)
    cls = REGISTRY[cfg["provider"]]
    return cls(cfg)
