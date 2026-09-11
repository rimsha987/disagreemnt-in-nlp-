"""Checkpoint-3 diagnostic round. ONE round, 120 calls, no third iteration.

DIAGNOSTIC A - noise floor (75 calls)
  Asian-Woman#1, the SAME persona, 5 independent repetitions on the same 15 pilot items.
  Two personas from the same cell (the pilot's 4/15) still differ in wording; repeating ONE
  persona isolates pure sampling noise at temperature 1.0 with nothing else varying. That is
  the floor the 2/15 between-cell difference has to clear.

DIAGNOSTIC B - prompt variant v2 (45 calls)
  Asian-Woman#1, White-Man#1, baseline#1 on the same 15 items, using src/prompts_v2.py.
  v1 records are untouched; v2 records carry variant="v2_turnscope_persona_adjacent" and live
  in their own shard.

Resumability and the append-only store work exactly as in src/07_run.py: keyed records,
flushed and fsync'd before the next call, completed keys skipped on restart.

Usage:
  .venv/Scripts/python.exe src/10_run_diagnostics.py --dry-run
  .venv/Scripts/python.exe src/10_run_diagnostics.py
"""
import sys, os, json, glob, time, argparse, datetime, signal, hashlib
import pandas as pd
sys.path.insert(0, "src")
import dices_io as D, personas as P
import prompts as PR1, prompts_v2 as PR2
import phase2_config as CFG
from llm_client import make_provider, DailyQuotaExhausted

RAW_DIR = "raw_responses/phase2"
CHARS_PER_TOKEN = 3.6
N_REPS_A = 5
B_CONDITIONS = ["Asian-Woman#1", "White-Man#1", "baseline#1"]
_stop = {"flag": False}


def _sigint(sig, frame):
    if _stop["flag"]:
        sys.exit("second interrupt; exiting hard")
    _stop["flag"] = True
    print("\n[interrupt] finishing the call in flight, then stopping cleanly.")


def cond_spec(cid):
    """(kind, cell, persona_k) for a condition id."""
    if cid.startswith("baseline"):
        return "baseline", None, 0
    cell, k = cid.split("#")
    return "persona", cell, int(k) - 1


def build_plan():
    plan = pd.read_csv("results/06_pilot_plan.csv")
    items = sorted(set(int(x) for x in plan.item_id))
    assert len(items) == 15, len(items)
    ent = plan.drop_duplicates("item_id").set_index("item_id").ent_bin
    df = D.load_350()
    conv = df.drop_duplicates("item_id").set_index("item_id")[["context", "response"]]

    rows = []
    # ---- Diagnostic A: one persona, 5 reps, v1 prompt
    kind, cell, k = cond_spec("Asian-Woman#1")
    blk = P.condition_prompt_block(kind, cell, k)
    for iid in items:
        for rep in range(1, N_REPS_A + 1):
            s, u = PR1.build_parts(conv.loc[iid, "context"], conv.loc[iid, "response"], blk)
            rows.append(dict(diagnostic="A", key=f"{iid}|Asian-Woman#1|v1|rep{rep}",
                             item_id=iid, condition="Asian-Woman#1", kind=kind,
                             cell=cell or "", persona_k=k, rep=rep, variant="v1_original",
                             ent_bin=ent.loc[iid], system_text=s, user_text=u))
    # ---- Diagnostic B: three conditions, 1 rep, v2 prompt
    for iid in items:
        for cid in B_CONDITIONS:
            kind, cell, k = cond_spec(cid)
            blk = P.condition_prompt_block(kind, cell, k)
            s, u = PR2.build_parts(conv.loc[iid, "context"], conv.loc[iid, "response"], blk)
            rows.append(dict(diagnostic="B", key=f"{iid}|{cid}|{PR2.VARIANT_ID}|rep1",
                             item_id=iid, condition=cid, kind=kind, cell=cell or "",
                             persona_k=k, rep=1, variant=PR2.VARIANT_ID,
                             ent_bin=ent.loc[iid], system_text=s, user_text=u))
    assert len({r["key"] for r in rows}) == len(rows)
    assert sum(r["diagnostic"] == "A" for r in rows) == 75
    assert sum(r["diagnostic"] == "B" for r in rows) == 45
    return rows


def load_done():
    done, dupes, n_err = set(), 0, 0
    for path in sorted(glob.glob(f"{RAW_DIR}/diag_*.jsonl")):
        for line in open(path, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("error"):
                n_err += 1
                continue
            if r["key"] in done:
                dupes += 1
                continue
            done.add(r["key"])
    return done, dupes, n_err


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--provider", default=None)
    args = ap.parse_args()

    prof = CFG.profile(args.provider)
    plan = build_plan()
    done, dupes, n_err = load_done()
    todo = [r for r in plan if r["key"] not in done]

    print("=" * 92)
    print(f"DIAGNOSTIC ROUND - provider={prof['provider']} model={prof['model']}")
    print("=" * 92)
    print(f"   Diagnostic A (noise floor, v1 prompt, 5 reps of Asian-Woman#1): "
          f"{sum(r['diagnostic']=='A' for r in plan)} calls")
    print(f"   Diagnostic B (v2 prompt, 3 conditions):                        "
          f"{sum(r['diagnostic']=='B' for r in plan)} calls")
    print(f"   total planned      {len(plan)}")
    print(f"   already complete   {len(done)}")
    print(f"   remaining          {len(todo)}")
    if dupes or n_err:
        print(f"   duplicates {dupes}, prior errors to retry {n_err}")

    if args.dry_run:
        print("\nDRY RUN. Validating; no API call, no file written.")
        a = [r for r in plan if r["diagnostic"] == "A"]
        b = [r for r in plan if r["diagnostic"] == "B"]
        assert len({r["system_text"] for r in a}) == 1, "A must hold the prompt constant"
        assert all(r["system_text"] == PR1.RUBRIC + "\n\n" +
                   P.persona_text("Asian-Woman", 0) for r in a)
        print(f"   A: 1 distinct system text across all 75 calls (prompt held constant) OK")
        print(f"   A: {len({r['item_id'] for r in a})} items x {N_REPS_A} reps OK")
        assert all(r["system_text"] == PR2.RUBRIC_V2 for r in b)
        assert all("About you:" in r["user_text"] for r in b if r["kind"] == "persona")
        assert all("About you:" not in r["user_text"] for r in b if r["kind"] == "baseline")
        assert all("THIS IS WHAT YOU ARE RATING" in r["user_text"] for r in b)
        print(f"   B: v2 system used for all 45, persona is in the USER message OK")
        print(f"   B: {len({r['condition'] for r in b})} conditions x "
              f"{len({r['item_id'] for r in b})} items OK")
        v1s = PR1.build_parts("C", "R", P.persona_text("Asian-Woman", 0))
        v2s = PR2.build_parts("C", "R", P.persona_text("Asian-Woman", 0))
        print(f"\n   v1 system {len(v1s[0])} chars / user {len(v1s[1])} chars")
        print(f"   v2 system {len(v2s[0])} chars / user {len(v2s[1])} chars")
        print("\nDRY RUN PASSED.")
        return

    if not todo:
        print("\nnothing to do.")
        return

    provider = make_provider(args.provider)
    os.makedirs(RAW_DIR, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
    shard = f"{RAW_DIR}/diag_{prof['provider']}_{stamp}.jsonl"
    if os.path.exists(shard):
        sys.exit(f"refusing to overwrite {shard}")
    signal.signal(signal.SIGINT, _sigint)
    print(f"\n   writing to {shard}\n")

    t0, n_ok, n_bad, quota = time.time(), 0, 0, None
    with open(shard, "w", encoding="utf-8") as fh:
        for i, r in enumerate(todo, 1):
            if _stop["flag"]:
                break
            persona_line = (r["system_text"][len(PR1.RUBRIC):].strip()
                            if r["variant"] == "v1_original" else
                            [l for l in r["user_text"].split("\n\n")
                             if "About you:" in l or "one specific individual" in l][:1])
            rec = dict(key=r["key"], diagnostic=r["diagnostic"], item_id=r["item_id"],
                       condition=r["condition"], kind=r["kind"], cell=r["cell"],
                       persona_k=r["persona_k"], rep=r["rep"], variant=r["variant"],
                       ent_bin=r["ent_bin"], provider=prof["provider"],
                       model_requested=prof["model"], temperature=CFG.TEMPERATURE,
                       max_output_tokens=CFG.MAX_OUTPUT_TOKENS,
                       system_sha256=hashlib.sha256(
                           r["system_text"].encode("utf-8")).hexdigest()[:16],
                       user_sha256=hashlib.sha256(
                           r["user_text"].encode("utf-8")).hexdigest()[:16],
                       persona_sent=(persona_line if isinstance(persona_line, str)
                                     else (persona_line[0] if persona_line else "")),
                       sent_at=datetime.datetime.now().isoformat())
            try:
                rep = provider.generate(r["system_text"], r["user_text"],
                                        est_tokens=int((len(r["system_text"]) +
                                                        len(r["user_text"])) / CHARS_PER_TOKEN))
            except DailyQuotaExhausted as e:
                quota = str(e)
                break
            rec.update(model_version=provider.version_string(), text=rep.text,
                       finish_reason=rep.finish_reason, in_tokens=rep.in_tokens,
                       out_tokens=rep.out_tokens, attempts=rep.attempts,
                       error=rep.error, rate_headers=rep.headers, raw=rep.raw)
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
            n_ok += rep.error is None
            n_bad += rep.error is not None
            if i % 20 == 0 or i == len(todo):
                el = time.time() - t0
                print(f"   {i}/{len(todo)}  ok={n_ok} err={n_bad}  "
                      f"{i/el*60:.1f}/min  {el/60:.1f}m elapsed")
    print(f"\n   done: {n_ok} ok, {n_bad} errors, {(time.time()-t0)/60:.1f} min")
    if quota:
        print(f"   STOPPED ON QUOTA: {quota}")
    d2, _, _ = load_done()
    print(f"   progress {len(d2)}/{len(plan)}")
    print(f"\n   now run: .venv/Scripts/python.exe src/11_diagnostic_report.py")


if __name__ == "__main__":
    main()
