"""Checkpoint 3 report from the pilot's raw responses. Reads raw_responses/*.jsonl,
never modifies it.

Answers, in order:
  1. did anything fail or refuse?
  2. did anything fail to parse? (every failure printed verbatim, never coerced)
  3. do two DISTINCT personas in the SAME cell differ?    <- validates the item-4 design
  4. do two DIFFERENT cells differ?                       <- validates persona conditioning
  5. do personas differ from the no-persona baseline?
  6. measured tokens, throughput, and the model's exact version string for the paper

Prompt-caching and cost analysis were removed with the move to free-tier providers.

Usage: .venv/Scripts/python.exe src/08_pilot_report.py [raw_responses/phase2/pilot_*.jsonl]
       with no argument it reads every pilot shard it finds.
"""
import sys, json, glob
import numpy as np, pandas as pd
sys.path.insert(0, "src")
import personas as P, prompts as PR, persona_check as PC

N_FULL_CALLS = 350 * 21


def load(paths):
    """Read every shard. Records are keyed by (item_id, condition); first occurrence wins."""
    rows, seen = [], set()
    for path in paths:
        for line in open(path, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            k = r.get("key") or f"{r['item_id']}|{r['condition']}"
            if k in seen:
                continue
            seen.add(k)
            rows.append(dict(
                key=k, item_id=r["item_id"], condition=r["condition"], kind=r["kind"],
                cell=r.get("cell", ""), persona_k=r.get("persona_k"),
                ent_bin=r.get("ent_bin"), error=r.get("error"),
                finish_reason=r.get("finish_reason"), raw_text=r.get("text"),
                rating=PR.parse_rating(r.get("text")),
                provider=r.get("provider"), model_requested=r.get("model_requested"),
                model_version=r.get("model_version"), attempts=r.get("attempts"),
                in_tok=r.get("in_tokens"), out_tok=r.get("out_tokens"),
                sent_at=r.get("sent_at"),
                # provenance: without these, "the model ignored the persona" cannot be
                # told apart from "the persona was never sent"
                persona_sent=r.get("persona_sent"),
                system_sha256=r.get("system_sha256"),
                user_sha256=r.get("user_sha256"),
                rate_headers=r.get("rate_headers")))
    df = pd.DataFrame(rows)
    assert df.empty or df.key.is_unique
    return df


def main():
    paths = sys.argv[1:] or sorted(glob.glob("raw_responses/phase2/pilot_*.jsonl"))
    if not paths:
        sys.exit("no raw_responses/phase2/pilot_*.jsonl found; run "
                 "src/07_run.py --mode pilot first")
    print("=" * 96)
    print("PILOT REPORT")
    print("=" * 96)
    for p_ in paths:
        print(f"   shard {p_}")
    d = load(paths)
    print(f"responses in file: {len(d)}   expected: 60   "
          f"{'MATCH' if len(d) == 60 else '*** MISMATCH ***'}")

    # ---------------------------------------------------------------- 1. failures
    print("\n" + "=" * 96)
    print("1. TRANSPORT FAILURES AND REFUSALS")
    print("=" * 96)
    n_err = int(d.error.notna().sum())
    print(f"   API errors:        {n_err} / {len(d)}")
    if n_err:
        for r in d[d.error.notna()].itertuples():
            print(f"      {r.key}: {r.error}")
    print(f"   finish_reason values: {dict(d.finish_reason.value_counts(dropna=False))}")
    print(f"   retry attempts: max {d.attempts.max()}, "
          f"calls needing >1 attempt: {int((d.attempts.fillna(1) > 1).sum())}")
    n_ref = int(d.finish_reason.astype(str).str.upper().isin(
        ["SAFETY", "BLOCKED", "PROHIBITED_CONTENT", "RECITATION", "CONTENT_FILTER"]).sum())
    print(f"   provider-side blocks/refusals (finish_reason): {n_ref} / {len(d)} "
          f"({100*n_ref/len(d):.1f}%)")

    prov_ok = PC.provenance(d)

    # ---------------------------------------------------------------- 2. parsing
    print("\n" + "=" * 96)
    print("2. PARSE SUCCESS")
    print("=" * 96)
    ok = d.rating.notna()
    print(f"   parsed:     {int(ok.sum())} / {len(d)} ({100*ok.mean():.1f}%)")
    print(f"   unparsed:   {int((~ok).sum())}")
    if (~ok).any():
        print("\n   EVERY unparsed reply, verbatim (not coerced into a rating):")
        for r in d[~ok].itertuples():
            print(f"      {r.key:<26} finish={r.finish_reason} text={r.raw_text!r}")
    print(f"\n   rating distribution over parsed replies:")
    for k, v in d.rating.value_counts(dropna=False).items():
        print(f"      {str(k):<10} {v:>4}  ({100*v/len(d):5.1f}%)")

    p = d[ok].copy()
    if p.empty:
        sys.exit("\nnothing parsed; cannot continue")

    piv = p.pivot_table(index="item_id", columns="condition", values="rating",
                        aggfunc="first")
    print(f"\n   per-item ratings by condition ({len(piv)} items):")
    print(piv.to_string())

    def agree(a, b):
        both = piv[[a, b]].dropna()
        return len(both), float((both[a] == both[b]).mean()) if len(both) else float("nan")

    # -------------------------------------------- 3-5. does anything actually differ?
    CONTRASTS = [
        ("3. WITHIN-CELL  (validates the distinct-persona design)",
         "Asian-Woman#1", "Asian-Woman#2",
         "identical => distinct personas inside a cell add nothing; the cell collapses to\n"
         "      one persona and 3-per-cell is not worth buying"),
        ("4. BETWEEN-CELL (validates persona conditioning at all)",
         "Asian-Woman#1", "White-Man#1",
         "identical => persona conditioning does not move ratings on the most distant\n"
         "      human pair available; it will not move them anywhere"),
        ("5. PERSONA vs BASELINE",
         "Asian-Woman#1", "baseline#1",
         "identical => the demographic text changed nothing relative to no persona"),
    ]
    for title, a, b, reading in CONTRASTS:
        print("\n" + "=" * 96)
        print(title)
        print("=" * 96)
        if a not in piv.columns or b not in piv.columns:
            print(f"   {a} or {b} missing from parsed results; cannot compare")
            continue
        n, ag = agree(a, b)
        print(f"   {a} vs {b}: agree on {ag*100:.1f}% of {n} items "
              f"({int(ag*n)} identical, {n-int(ag*n)} different)")
        print(f"   marginals: {a} {dict(p[p.condition==a].rating.value_counts())}")
        print(f"              {b} {dict(p[p.condition==b].rating.value_counts())}")
        ct = pd.crosstab(piv[a], piv[b])
        print(f"\n   {a} (rows) x {b} (cols):")
        print(ct.to_string())
        if ag == 1.0:
            print(f"\n   *** IDENTICAL ON EVERY ITEM ***")
            print(f"      {reading}")
        n_dis = int(len(piv[[a, b]].dropna()) - (piv[a] == piv[b]).sum())
        if 0 < n_dis:
            print(f"\n   items where they differ:")
            m = piv[piv[a] != piv[b]][[a, b]]
            print(m.to_string())

    PC.verdict(d, piv, provenance_ok=prov_ok)
    PC.baseline_degeneracy(d, piv)

    print("\n" + "=" * 96)
    print("HOW MUCH VARIATION IS THERE AT ALL?")
    print("=" * 96)
    per_item_uniq = piv.nunique(axis=1)
    print(f"   items where all 4 conditions gave the SAME rating: "
          f"{int((per_item_uniq==1).sum())} / {len(piv)}")
    print(f"   items with 2 distinct ratings: {int((per_item_uniq==2).sum())}")
    print(f"   items with 3 distinct ratings: {int((per_item_uniq==3).sum())}")
    print(f"\n   by disagreement tertile of the HUMAN pool:")
    for b in ["low", "medium", "high"]:
        s = p[p.ent_bin == b]
        if s.empty:
            continue
        pv = s.pivot_table(index="item_id", columns="condition", values="rating",
                           aggfunc="first")
        print(f"      {b:<8} {len(pv)} items, "
              f"{int((pv.nunique(axis=1)==1).sum())} fully unanimous across conditions, "
              f"%Yes={100*(s.rating=='Yes').mean():.1f}")

    # -------------------------------------------- 6. model identity, tokens, throughput
    print("\n" + "=" * 96)
    print("6. MODEL IDENTITY, MEASURED TOKENS, THROUGHPUT")
    print("=" * 96)
    print("   FOR THE PAPER'S DATA SECTION - copy these verbatim:")
    print(f"      provider              {sorted(set(d.provider.dropna()))}")
    print(f"      model requested       {sorted(set(d.model_requested.dropna()))}")
    print(f"      model version (as reported by the API)")
    for v, n in d.model_version.value_counts(dropna=False).items():
        print(f"                            {str(v):<40} {n} calls")
    if d.model_version.nunique(dropna=True) > 1:
        print(f"      *** more than one version string appeared. The provider silently")
        print(f"          rerouted mid-run; report every version and the split, and check")
        print(f"          whether ratings differ across them before pooling.")

    u = d.dropna(subset=["in_tok"])
    if u.empty:
        print("\n   no usage data (all calls failed)")
        return
    print(f"\n   input_tokens   mean {u.in_tok.mean():8.1f}  min {u.in_tok.min():.0f}  "
          f"max {u.in_tok.max():.0f}  total {u.in_tok.sum():,.0f}")
    print(f"   output_tokens  mean {u.out_tok.mean():8.2f}  max {u.out_tok.max():.0f}")
    print(f"   (measured by the provider's own tokeniser; supersedes the chars/token"
          f" estimate in results/06_token_profile.txt)")

    ts = pd.to_datetime(d.sent_at.dropna(), errors="coerce").dropna().sort_values()
    if len(ts) > 1:
        span = (ts.iloc[-1] - ts.iloc[0]).total_seconds()
        print(f"\n   wall-clock for {len(ts)} calls: {span/60:.1f} min "
              f"({len(ts)/(span/60):.1f} calls/min achieved)")
        print(f"   PROJECTED for the full {N_FULL_CALLS:,} calls at that observed rate:")
        print(f"      {N_FULL_CALLS/(len(ts)/(span/60))/60:,.1f} hours of continuous calling")
        print(f"      (the daily-quota ceiling, not this rate, is what actually binds -")
        print(f"       see src/07_run.py --mode full --status)")
    print(f"\n   COST: none. Free tier.")

    if "rate_headers" in d.columns:
        hdr = [h for h in d.rate_headers.dropna() if isinstance(h, dict) and h]
        if hdr:
            print("\n   PROVIDER QUOTA HEADERS (last observed):")
            for k, v in hdr[-1].items():
                print(f"      {k:<34} {v}")
            print("   x-ratelimit-reset-requests is the OBSERVED daily reset. Groq does not")
            print("   document reset timing, so this header is the only authority on it.")

    out = "results/08_pilot_parsed.csv"
    d.to_csv(out, index=False)
    print(f"\n   wrote {out} (raw shards untouched)")


if __name__ == "__main__":
    main()
