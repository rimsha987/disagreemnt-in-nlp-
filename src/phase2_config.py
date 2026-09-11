"""Phase 2 provider configuration. Swapping providers is a change to this file only.

Nothing about prompt construction, persona injection, or parsing lives here or changes when
the provider changes. Those stay in src/prompts.py and src/personas.py.

RATE LIMITS - READ THIS BEFORE TRUSTING THE NUMBERS BELOW

Google no longer publishes free-tier RPM/TPM/RPD figures in its documentation. As of the
2026-08-18 revision, https://ai.google.dev/gemini-api/docs/rate-limits says only that limits
"depend on a variety of factors (such as your usage tier) and can be viewed in Google AI
Studio", and points to https://aistudio.google.com/rate-limit for the account's actual
numbers. It also warns that "specified rate limits are not guaranteed".

So the Gemini limits below are NOT verified. They are a conservative placeholder based on the
last figures that were public (10 RPM / 250 RPD for 2.5 Flash, documented mid-2025) and they
are almost certainly wrong for any given account. Check the dashboard and set them here before
sizing a real run. The runner does not depend on them being right - it backs off adaptively on
429 - but the wall-clock projection does.

Groq's limits ARE published, per model, and are transcribed here from
https://console.groq.com/docs/rate-limits (fetched 2026-08-26).
"""

# ----------------------------------------------------------------- which provider is live
PROVIDER = "groq"            # "gemini" | "groq" | "groq-gptoss"

# --------------------------------------------------------------------- generation settings
# Unchanged from the Anthropic design: sampling is the point, so temperature stays at 1.0.
TEMPERATURE = 1.0
MAX_OUTPUT_TOKENS = 8

# ----------------------------------------------------------------------- provider profiles
PROVIDERS = {
    "gemini": dict(
        model="gemini-2.5-flash",
        api_key_env=["GEMINI_API_KEY", "GOOGLE_API_KEY"],
        # UNVERIFIED - see the module docstring. Check aistudio.google.com/rate-limit.
        rpm=10, tpm=250_000, rpd=250,
        limits_verified=False,
        limits_source="ai.google.dev/gemini-api/docs/rate-limits (no free-tier table as of "
                      "2026-08-18); placeholder from last public figures, mid-2025",
    ),
    "groq": dict(
        model="qwen/qwen3.8-27b",
        api_key_env=["GROQ_API_KEY"],
        # Published per-model free-plan limits, transcribed 2026-08-26.
        rpm=30, tpm=8_000, rpd=1_000, tpd=2_000_000,
        limits_verified=True,
        limits_source="console.groq.com/docs/rate-limits, fetched 2026-08-26",
    ),
    # Fallback candidate if qwen fails format compliance. Same RPM/RPD, but TPD is 200K
    # rather than 2M, which is the binding constraint - see results/06_run_sizing.txt.
    "groq-gptoss": dict(
        model="openai/gpt-oss-120b",
        api_key_env=["GROQ_API_KEY"],
        rpm=30, tpm=8_000, rpd=1_000, tpd=200_000,
        limits_verified=True,
        limits_source="console.groq.com/docs/rate-limits, fetched 2026-08-26",
    ),
}

# ------------------------------------------------------------------------ retry behaviour
RETRY_MAX_ATTEMPTS = 8
RETRY_BASE_SECONDS = 2.0      # exponential: 2, 4, 8, 16, 32, 64, 120, 120
RETRY_CAP_SECONDS = 120.0
RETRY_JITTER = 0.25           # +/- 25%, so parallel restarts do not resynchronise


def profile(name=None):
    name = name or PROVIDER
    if name not in PROVIDERS:
        raise KeyError(f"unknown provider {name!r}; have {sorted(PROVIDERS)}")
    return dict(PROVIDERS[name], provider=name)
