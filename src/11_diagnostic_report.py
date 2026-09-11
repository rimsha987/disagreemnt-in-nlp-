"""Checkpoint-3 diagnostic report. Applies the decision rule fixed in advance.

DECISION RULE (fixed before the calls were made, not after):
  * if the between-cell difference clears the noise floor under EITHER prompt
        -> proceed to the full 3,675-call run using that prompt
  * if it clears under neither
        -> STOP the LLM arm; write up as a documented negative with the noise floor as
           evidence, and pivot to Phase 1 + Phase 3 human conditions only

THE COMPARISON HAS TO BE LIKE FOR LIKE. The pilot's between-cell figure (2/15) came from ONE
draw of Asian-Woman#1 against ONE draw of White-Man#1. So the floor must also be built from
ONE draw against ONE draw of the SAME persona. Diagnostic A gives 5 independent repetitions of
Asian-Woman#1, which yields C(5,2)=10 same-persona pairs per comparison; each pair produces a
count of disagreeing items out of 15, and that distribution is the null the observed 2/15 is
tested against.

Outputs: results/11_diagnostic_report.txt (via tee), results/11_diag_parsed.csv
"""
import sys, json, glob, itertools
import numpy as np, pandas as pd
sys.path.insert(0, "src")
import dices_io as D, prompts as PR

RAW = "raw_responses/phase2"


TAIL_ALPHA = 0.05        # an observed count clears the floor only if it is unlikely under it


def clears(obs, null, gap=None):
    """Does an observed disagreement count clear the same-persona noise floor?

    CORRECTED. The first version of this script used `obs > null.mean()`, which is far too
    weak: the mean is the CENTRE of the noise distribution, so a coin flip beats it half the
    time. Clearing a noise floor has to mean lying OUTSIDE the noise, i.e. being unlikely
    under it. Two conditions, both required:
      1. P(noise >= obs) <= TAIL_ALPHA
      2. the directional P(Yes) gap runs the same way as the human gap, when supplied -
         label churn that nets to zero is not a demographic effect
    """
    tail = float((null >= obs).mean())
    ok = tail <= TAIL_ALPHA
    if gap is not None:
        ok = ok and gap > 0
    return ok


def load():
    rows = []
    for path in sorted(glob.glob(f"{RAW}/diag_*.jsonl")):
        for line in open(path, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            r["rating"] = PR.parse_rating(r.get("text"))
            rows.append(r)
    d = pd.DataFrame(rows)
    if not d.empty:
        d = d.drop_duplicates("key", keep="first")
    return d


def pilot():
    rows = []
    for path in sorted(glob.glob(f"{RAW}/pilot_*.jsonl")):
        for line in open(path, encoding="utf-8"):
            if line.strip():
                r = json.loads(line)
                r["rating"] = PR.parse_rating(r.get("text"))
                rows.append(r)
    return pd.DataFrame(rows).drop_duplicates("key", keep="first")


def human_ref(items):
    df = D.load_350()
    sub = df[df.item_id.isin(items)]
    R3 = {"White": "White", "Black/African American": "Black",
          "Asian/Asian subcontinent": "Asian"}
    r = sub[sub.rater_race.isin(R3)].copy()
    r["cell"] = r.rater_race.map(R3) + "-" + r.rater_gender
    out = {"ALL 123": float((sub.Q_overall == "Yes").mean())}
    for c in ["Asian-Woman", "White-Man"]:
        out[c] = float((r[r.cell == c].Q_overall == "Yes").mean())
    return out


def main():
    d, pil = load(), pilot()
    if d.empty:
        sys.exit("no diagnostic records; run src/10_run_diagnostics.py first")
    print("=" * 96)
    print("CHECKPOINT-3 DIAGNOSTIC REPORT")
    print("=" * 96)
    print(f"   diagnostic records {len(d)}   (A {int((d.diagnostic=='A').sum())}, "
          f"B {int((d.diagnostic=='B').sum())})")
    print(f"   API errors         {int(d.error.notna().sum())}")
    print(f"   parse failures     {int(d.rating.isna().sum())}")
    if d.rating.isna().any():
        for r in d[d.rating.isna()].itertuples():
            print(f"      {r.key}: text={r.text!r} finish={r.finish_reason}")
    print(f"   model_version      {sorted(set(d.model_version.dropna()))}")

    items = sorted(set(d.item_id))
    href = human_ref(items)

    # ================================================================ DIAGNOSTIC A
    A = d[d.diagnostic == "A"]
    print("\n" + "=" * 96)
    print("DIAGNOSTIC A - THE NOISE FLOOR")
    print("Asian-Woman#1, v1 prompt, 5 independent repetitions, same 15 items")
    print("=" * 96)
    pv = A.pivot_table(index="item_id", columns="rep", values="rating", aggfunc="first")
    pv = pv[sorted(pv.columns)]
    print("\n   per-item ratings across the 5 repetitions:")
    print(pv.to_string())

    nuniq = pv.nunique(axis=1)
    print(f"\n   items identical on all 5 reps : {int((nuniq==1).sum())} / {len(pv)}")
    print(f"   items with 2 distinct ratings  : {int((nuniq==2).sum())}")
    print(f"   items with 3 distinct ratings  : {int((nuniq==3).sum())}")
    modal = pv.apply(lambda row: row.value_counts().iloc[0] / row.notna().sum(), axis=1)
    print(f"   mean per-item modal share      : {modal.mean():.3f}")
    print(f"   mean per-item flip rate        : {1-modal.mean():.3f}  "
          f"(chance a single draw differs from the item's own mode)")

    # like-for-like null: one draw vs one draw of the SAME persona
    reps = list(pv.columns)
    counts = []
    for a, b in itertools.combinations(reps, 2):
        counts.append(int((pv[a] != pv[b]).sum()))
    counts = np.array(counts)
    print(f"\n   SAME-PERSONA PAIRWISE DISAGREEMENT (the like-for-like floor)")
    print(f"      {len(counts)} pairs of repetitions, each 1 draw vs 1 draw on 15 items")
    print(f"      counts: {sorted(counts.tolist())}")
    print(f"      mean {counts.mean():.2f}/15   median {np.median(counts):.1f}/15   "
          f"min {counts.min()}   max {counts.max()}")
    print(f"      95% range [{np.percentile(counts,2.5):.1f}, {np.percentile(counts,97.5):.1f}]")

    pp_pilot = pil.pivot_table(index="item_id", columns="condition",
                               values="rating", aggfunc="first")
    obs_between = int((pil.pivot_table(index="item_id", columns="condition",
                                       values="rating", aggfunc="first")
                       .pipe(lambda p: p["Asian-Woman#1"] != p["White-Man#1"])).sum())
    p_ge = float((counts >= obs_between).mean())
    v1_gap = (float((pp_pilot["Asian-Woman#1"] == "Yes").mean())
              - float((pp_pilot["White-Man#1"] == "Yes").mean()))
    v1_clears = clears(obs_between, counts, v1_gap)
    print(f"\n   OBSERVED between-cell (pilot, v1): {obs_between}/15")
    print(f"   P(noise >= {obs_between}) = {p_ge:.2f}   percentile "
          f"{100*float((counts < obs_between).mean()):.0f}th")
    print(f"   directional P(Yes) gap AW-WM under v1: {v1_gap:+.3f}   "
          f"(human {0.4333-0.1091:+.3f})")
    print(f"   -> {'CLEARS' if v1_clears else 'DOES NOT CLEAR'} the noise floor")

    # ================================================================ DIAGNOSTIC B
    B = d[d.diagnostic == "B"]
    print("\n" + "=" * 96)
    print("DIAGNOSTIC B - PROMPT VARIANT v2")
    print("=" * 96)
    pb = B.pivot_table(index="item_id", columns="condition", values="rating", aggfunc="first")
    pp = pil.pivot_table(index="item_id", columns="condition", values="rating", aggfunc="first")
    print("\n   per-item ratings under v2:")
    print(pb.to_string())

    print(f"\n   P(Yes) by condition, matched 15 items")
    print(f"      {'condition':<18} {'v1':>8} {'v2':>8}   human")
    for c in ["Asian-Woman#1", "White-Man#1", "baseline#1"]:
        v1 = float((pp[c] == "Yes").mean()) if c in pp else np.nan
        v2 = float((pb[c] == "Yes").mean()) if c in pb else np.nan
        hh = href.get(c.split("#")[0], href["ALL 123"])
        print(f"      {c:<18} {v1:>8.3f} {v2:>8.3f}   {hh:.3f}")
    allv1 = float((pil.rating == "Yes").mean())
    allv2 = float((B.rating == "Yes").mean())
    print(f"      {'ALL calls':<18} {allv1:>8.3f} {allv2:>8.3f}   {href['ALL 123']:.3f} (all 123 humans)")
    print(f"\n   rating mix under v2: {dict(B.rating.value_counts())}")

    obs_b = int((pb["Asian-Woman#1"] != pb["White-Man#1"]).sum())
    p_ge_b = float((counts >= obs_b).mean())
    v2_gap = (float((pb['Asian-Woman#1']=='Yes').mean())
              - float((pb['White-Man#1']=='Yes').mean()))
    v2_clears = clears(obs_b, counts, v2_gap)
    print(f"\n   between-cell under v2 (Asian-Woman#1 vs White-Man#1): {obs_b}/15")
    print(f"   P(noise >= {obs_b}) = {p_ge_b:.2f}   percentile "
          f"{100*float((counts<obs_b).mean()):.0f}th")
    print(f"   directional P(Yes) gap AW-WM under v2: {v2_gap:+.3f}  "
          f"(human {0.4333-0.1091:+.3f})")
    print(f"   between-cell {obs_b} vs noise floor mean {counts.mean():.2f}  ->  "
          f"{'CLEARS' if v2_clears else 'DOES NOT CLEAR'}")
    print(f"\n   CAVEAT: the floor was measured under v1. v2 changes the prompt, so its own")
    print(f"   noise floor was not measured and could differ. This is the one round agreed,")
    print(f"   so the v1 floor is used as the best available reference and the limitation is")
    print(f"   stated rather than hidden.")

    # ================================================================ DECISION
    print("\n" + "=" * 96)
    print("DECISION RULE, APPLIED")
    print("=" * 96)
    print(f"   noise floor (same persona, 1 draw vs 1 draw): {sorted(counts.tolist())}")
    print(f"      mean {counts.mean():.2f}/15, range {counts.min()}-{counts.max()}")
    print(f"   criterion: P(noise >= observed) <= {TAIL_ALPHA}, AND the directional P(Yes)")
    print(f"      gap runs the human way. Beating the floor's MEAN is not sufficient - the")
    print(f"      mean is the centre of the noise, not its edge.")
    print(f"   POWER LIMIT: with {len(counts)} pairs the smallest achievable tail probability")
    print(f"      is {1/len(counts):.2f}, so this round cannot return a p<=0.05 positive by")
    print(f"      construction. It can return a clear negative or an inconclusive.")
    print(f"   between-cell under v1: {obs_between}/15  -> "
          f"{'CLEARS' if v1_clears else 'does not clear'}")
    print(f"   between-cell under v2: {obs_b}/15  -> "
          f"{'CLEARS' if v2_clears else 'does not clear'}")
    print()
    if v1_clears or v2_clears:
        winner = "v2" if (v2_clears and (not v1_clears or obs_b >= obs_between)) else "v1"
        print(f"   >>> PROCEED to the full 3,675-call run using prompt {winner}.")
        print(f"       A between-cell difference exceeds the same-persona noise floor, so the")
        print(f"       demographic manipulation has a measurable effect to estimate.")
    else:
        print(f"   >>> STOP THE LLM ARM. Do not run the 3,675 calls.")
        print(f"       Under neither prompt does the between-cell difference exceed the")
        print(f"       same-persona sampling floor. Persona conditioning cannot be")
        print(f"       distinguished from noise in this model, so a 3,675-call dataset would")
        print(f"       not make the central RQ2 comparison interpretable - it would just")
        print(f"       measure the same noise more precisely.")
        print(f"       Write up as a DOCUMENTED NEGATIVE with the noise floor as the evidence,")
        print(f"       and pivot to Phase 1 + Phase 3 human conditions only.")

    d.to_csv("results/11_diag_parsed.csv", index=False)
    print(f"\n   wrote results/11_diag_parsed.csv (raw shards untouched)")


if __name__ == "__main__":
    main()
