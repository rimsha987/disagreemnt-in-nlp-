"""Phase 2 runner. Sequential, rate-limited, and RESUMABLE. Used for both the pilot and the
full run so they exercise the same code path.

RESUMABILITY - the design point

Every response is appended to a JSONL shard in raw_responses/phase2/ the moment it arrives,
flushed immediately, keyed by (item_id, condition). On start the runner reads every shard it
finds, builds the set of completed keys, and skips them. A crash, a Ctrl-C, a daily quota
wall, or a machine reboot costs at most the one call in flight.

  * shards are append-only and never rewritten; each session opens a NEW shard file
  * a key counts as complete only if that record has error=null, so failures are retried
  * duplicate keys across shards are tolerated on read (first wins) and reported

Usage
  .venv/Scripts/python.exe src/07_run.py --mode pilot --dry-run
  .venv/Scripts/python.exe src/07_run.py --mode pilot
  .venv/Scripts/python.exe src/07_run.py --mode full
  .venv/Scripts/python.exe src/07_run.py --mode full --provider groq --limit 500
  .venv/Scripts/python.exe src/07_run.py --mode full --status
"""
import sys, os, json, glob, time, argparse, datetime, signal, hashlib
import pandas as pd
sys.path.insert(0, "src")
import dices_io as D, personas as P, prompts as PR
import phase2_config as CFG
from llm_client import make_provider, DailyQuotaExhausted

RAW_DIR = "raw_responses/phase2"
CHARS_PER_TOKEN = 3.6
_stop = {"flag": False}


def _sigint(sig, frame):
    if _stop["flag"]:
        sys.exit("second interrupt; exiting hard")
    _stop["flag"] = True
    print("\n[interrupt] finishing the call in flight, then stopping cleanly. "
          "Re-run the same command to resume.")


def key_of(item_id, condition):
    return f"{int(item_id)}|{condition}"


ITEM_BLOCKS = "results/09_item_blocks.csv"


def frozen_items(blocks):
    """The frozen item sample. Never redrawn here; this only reads it."""
    if not os.path.exists(ITEM_BLOCKS):
        sys.exit(f"{ITEM_BLOCKS} not found. Run src/09_select_items.py first - it draws "
                 f"the 175-item sample ONCE and freezes it. The run must not draw its own.")
    b = pd.read_csv(ITEM_BLOCKS)
    assert len(b) == 350 and b.item_id.is_unique
    sel = b[b.block.isin(blocks)]
    return sorted(sel.item_id.astype(int)), b


def build_plan(mode, blocks=(1,)):
    """The full list of (item_id, condition, kind, cell, persona_k) for this mode."""
    df = D.load_350()
    items = df.drop_duplicates("item_id").set_index("item_id")[["context", "response"]]
    if mode == "pilot":
        plan = pd.read_csv("results/06_pilot_plan.csv")
        assert len(plan) == 60 and plan.request_id.is_unique
        rows = [dict(item_id=int(r.item_id), condition=r.condition, kind=r.kind,
                     cell=(r.cell if isinstance(r.cell, str) else ""),
                     persona_k=int(r.persona_k), ent_bin=r.ent_bin)
                for r in plan.itertuples()]
    elif mode == "full":
        ent = pd.read_csv("results/03_item_jsd.csv").set_index("item_id").ent_bin
        chosen, _ = frozen_items(blocks)
        rows = []
        for iid in chosen:
            for cid, kind, cell, s in P.all_conditions():
                k = int(cid.split("#")[1]) - 1 if kind == "persona" else 0
                rows.append(dict(item_id=int(iid), condition=cid, kind=kind,
                                 cell=cell or "", persona_k=k, ent_bin=ent.loc[iid]))
    else:
        raise ValueError(mode)
    for r in rows:
        r["key"] = key_of(r["item_id"], r["condition"])
        ctx, rsp = items.loc[r["item_id"], "context"], items.loc[r["item_id"], "response"]
        r["system_text"], r["user_text"] = PR.build_parts(
            ctx, rsp, P.condition_prompt_block(r["kind"], r["cell"] or None, r["persona_k"]))
    assert len({r["key"] for r in rows}) == len(rows), "duplicate keys in plan"
    return rows


def load_done(mode):
    """Completed keys from every existing shard. error=null only."""
    done, seen, dupes, n_err = {}, set(), 0, 0
    for path in sorted(glob.glob(f"{RAW_DIR}/{mode}_*.jsonl")):
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue            # a torn final line from a hard kill; ignore it
                k = r.get("key") or key_of(r["item_id"], r["condition"])
                if r.get("error"):
                    n_err += 1
                    continue
                if k in seen:
                    dupes += 1
                    continue
                seen.add(k)
                done[k] = path
    return done, dupes, n_err


def est_tokens(system_text, user_text):
    return int((len(system_text) + len(user_text)) / CHARS_PER_TOKEN)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["pilot", "full"], default="pilot")
    ap.add_argument("--blocks", default="1",
                    help="which frozen item blocks to run, e.g. '1' or '1,2'. Block 2 is the "
                         "queued continuation; adding it EXTENDS the same item-keyed store "
                         "and re-runs nothing.")
    ap.add_argument("--provider", default=None,
                    help="override phase2_config.PROVIDER")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--status", action="store_true",
                    help="report progress and exit without calling anything")
    ap.add_argument("--limit", type=int, default=None,
                    help="stop after this many NEW calls this session")
    args = ap.parse_args()

    blocks = tuple(int(x) for x in str(args.blocks).split(",") if x.strip())
    prof = CFG.profile(args.provider)
    plan = build_plan(args.mode, blocks)
    done, dupes, n_err = load_done(args.mode)
    todo = [r for r in plan if r["key"] not in done]

    print("=" * 92)
    print(f"PHASE 2 RUNNER - mode={args.mode}  provider={prof['provider']}  "
          f"model={prof['model']}")
    print("=" * 92)
    if args.mode == "full":
        chosen, tbl = frozen_items(blocks)
        print(f"   item blocks        {list(blocks)}  ({len(chosen)} items, "
              f"frozen in {ITEM_BLOCKS})")
        for b in sorted(set(tbl.block)):
            mark = "RUNNING" if b in blocks else "queued"
            print(f"      block {b}: {int((tbl.block==b).sum())} items  [{mark}]")
    print(f"   planned calls      {len(plan):,}")
    print(f"   already complete   {len(done):,}")
    print(f"   remaining          {len(todo):,}")
    if dupes:
        print(f"   duplicate keys across shards (first kept): {dupes}")
    if n_err:
        print(f"   prior error records (will be retried):     {n_err}")
    shards = sorted(glob.glob(f"{RAW_DIR}/{args.mode}_*.jsonl"))
    print(f"   existing shards    {len(shards)}")
    for s in shards[-5:]:
        print(f"      {s}")

    rl = f"{prof.get('rpm')} RPM / {prof.get('rpd')} RPD / {prof.get('tpm')} TPM"
    if prof.get("tpd"):
        rl += f" / {prof['tpd']} TPD"
    print(f"\n   configured limits  {rl}")
    print(f"   limits verified    {prof['limits_verified']}")
    print(f"   limits source      {prof['limits_source']}")
    if not prof["limits_verified"]:
        print(f"   *** these limits are a PLACEHOLDER, not a published figure. The runner")
        print(f"       backs off adaptively regardless, but the projection below is a guess.")

    if todo:
        mean_tok = sum(est_tokens(r["system_text"], r["user_text"]) for r in todo) / len(todo)
        per_min = prof["rpm"]
        if prof.get("tpm"):
            per_min = min(per_min, prof["tpm"] / mean_tok)
        cap_day = prof.get("rpd") or float("inf")
        if prof.get("tpd"):
            cap_day = min(cap_day, prof["tpd"] / mean_tok)
        print(f"\n   PROJECTION for the {len(todo):,} remaining calls")
        print(f"      mean estimated input tokens/call   {mean_tok:.0f}")
        print(f"      throughput ceiling                 {per_min:.1f} calls/min")
        print(f"      daily ceiling                      {cap_day:,.0f} calls/day"
              f"   ({'RPD' if cap_day == prof.get('rpd') else 'TPD'}-bound)")
        if cap_day < float("inf"):
            print(f"      calendar days needed               {len(todo)/cap_day:,.1f}")
        print(f"      in-day wall-clock at the ceiling   "
              f"{len(todo)/per_min/60:,.1f} hours of actual calling")

    if args.status:
        return
    if args.dry_run:
        print("\nDRY RUN. Validating requests; no API call, no file written.")
        for r in plan:
            assert r["system_text"].startswith(PR.RUBRIC)
            assert "FINAL CHATBOT RESPONSE" in r["user_text"]
            if r["kind"] == "persona":
                assert "About you:" in r["system_text"], r["key"]
            else:
                assert "About you:" not in r["system_text"], r["key"]
        n_txt = len({r["system_text"].replace(PR.RUBRIC, "") for r in plan})
        print(f"   all {len(plan):,} requests valid")
        print(f"   distinct persona/baseline texts: {n_txt}")
        print("\nDRY RUN PASSED.")
        return

    if not todo:
        print("\nnothing to do; every planned call is already complete.")
        return

    provider = make_provider(args.provider)
    os.makedirs(RAW_DIR, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
    shard = f"{RAW_DIR}/{args.mode}_{prof['provider']}_{stamp}.jsonl"
    if os.path.exists(shard):
        sys.exit(f"refusing to overwrite {shard}")
    signal.signal(signal.SIGINT, _sigint)

    limit = args.limit or len(todo)
    print(f"\n   writing to {shard}")
    print(f"   this session will attempt up to {min(limit, len(todo)):,} calls")
    print(f"   Ctrl-C once to stop cleanly and resume later\n")

    t0, n_ok, n_bad = time.time(), 0, 0
    quota_hit = None
    with open(shard, "w", encoding="utf-8") as fh:
        for i, r in enumerate(todo[:limit], 1):
            if _stop["flag"]:
                break
            # Provenance of what was actually sent. Without this, "the model ignored the
            # persona" cannot be distinguished from "the persona was never in the prompt".
            persona_line = r["system_text"][len(PR.RUBRIC):].strip()
            rec = dict(key=r["key"], item_id=r["item_id"], condition=r["condition"],
                       kind=r["kind"], cell=r["cell"], persona_k=r["persona_k"],
                       ent_bin=r["ent_bin"], provider=prof["provider"],
                       model_requested=prof["model"], temperature=CFG.TEMPERATURE,
                       max_output_tokens=CFG.MAX_OUTPUT_TOKENS,
                       system_sha256=hashlib.sha256(
                           r["system_text"].encode("utf-8")).hexdigest()[:16],
                       user_sha256=hashlib.sha256(
                           r["user_text"].encode("utf-8")).hexdigest()[:16],
                       persona_sent=persona_line,
                       sent_at=datetime.datetime.now().isoformat())
            try:
                rep = provider.generate(r["system_text"], r["user_text"],
                                        est_tokens=est_tokens(r["system_text"], r["user_text"]))
            except DailyQuotaExhausted as e:
                quota_hit = str(e)
                break
            rec.update(model_version=provider.version_string(), text=rep.text,
                       finish_reason=rep.finish_reason, in_tokens=rep.in_tokens,
                       out_tokens=rep.out_tokens, attempts=rep.attempts,
                       error=rep.error, rate_headers=rep.headers, raw=rep.raw)
            if i == 1 and rep.headers:
                print(f"   provider quota headers on the first call:")
                for k, v in rep.headers.items():
                    print(f"      {k:<32} {v}")
                print(f"   (x-ratelimit-reset-requests is the observed daily reset; Groq does"
                      f" not document it)")
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fh.flush()
            os.fsync(fh.fileno())          # survive a hard power loss, not just a crash
            n_ok += rep.error is None
            n_bad += rep.error is not None
            if i % 25 == 0 or i == min(limit, len(todo)):
                el = time.time() - t0
                rate = i / el * 60 if el else 0
                print(f"   {i:,}/{min(limit,len(todo)):,}  ok={n_ok:,} err={n_bad:,}  "
                      f"{rate:.1f} calls/min  elapsed {el/60:.1f}m")

    el = time.time() - t0
    print(f"\n   session done: {n_ok:,} ok, {n_bad:,} errors, {el/60:.1f} minutes")
    if quota_hit:
        print(f"   STOPPED ON QUOTA: {quota_hit}")
    if _stop["flag"]:
        print(f"   STOPPED ON INTERRUPT")
    done2, _, _ = load_done(args.mode)
    print(f"   overall progress: {len(done2):,}/{len(plan):,} "
          f"({100*len(done2)/len(plan):.1f}%)")
    if len(done2) < len(plan):
        cmd = f"src/07_run.py --mode {args.mode}"
        if args.provider:
            cmd += f" --provider {args.provider}"
        print(f"\n   RESUME with the same command:\n      .venv/Scripts/python.exe {cmd}")
    else:
        print(f"\n   complete. Now run:\n      .venv/Scripts/python.exe "
              f"src/08_pilot_report.py" if args.mode == "pilot" else "")


if __name__ == "__main__":
    main()
