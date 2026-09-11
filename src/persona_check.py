"""Does the model follow the persona instruction AT ALL?

This is a capability check on the model, not a finding about persona-prompting, and the two
must not be confused. If qwen3.8-27b returns identical ratings for two personas whose real
human counterparts differ by ~30 points of P(Yes), the correct reading is "this model does not
condition on the persona", which limits what the paper can claim about persona-prompting in
general. Reporting that as "personas fail to add diversity" would be a claim the design cannot
support.

Three things are separated here, in order:
  1. PROVENANCE  was the demographic text actually in the prompt?  (from recorded hashes)
  2. SENSITIVITY does the output change at all between conditions?
  3. DIRECTION   where it changes, does it move the way the humans do?

Only if 1 passes and 2 fails do we have a model-capability problem.
"""
import sys
import numpy as np, pandas as pd
sys.path.insert(0, "src")
import dices_io as D, personas as P

RACE3 = {"White": "White", "Black/African American": "Black",
         "Asian/Asian subcontinent": "Asian"}


def provenance(d):
    """Confirm the persona text reached the API, from the per-record provenance fields."""
    print("\n" + "=" * 96)
    print("0. PROVENANCE - was the demographic text actually sent?")
    print("=" * 96)
    if "persona_sent" not in d.columns or d.persona_sent.isna().all():
        print("   NO PROVENANCE RECORDED. These records predate provenance capture, so")
        print("   'the model ignored the persona' CANNOT be distinguished from 'the persona")
        print("   was never sent'. Re-run the pilot before drawing any capability conclusion.")
        return False
    ok = True
    for cond, sub in d.groupby("condition"):
        texts = sorted(set(sub.persona_sent.dropna()))
        shas = sorted(set(sub.system_sha256.dropna())) if "system_sha256" in sub else []
        expect_demo = sub.kind.iloc[0] == "persona"
        has_demo = all("About you:" in t for t in texts)
        good = (len(texts) == 1) and (has_demo == expect_demo)
        ok &= good
        print(f"\n   {cond}  ({len(sub)} calls)")
        print(f"      distinct system texts: {len(texts)}   distinct system sha256: {len(shas)}")
        print(f"      demographic sentence present: {has_demo}   expected: {expect_demo}"
              f"   {'OK' if good else '*** PROBLEM ***'}")
        print(f"      sent verbatim: {texts[0][:150]!r}")
    print(f"\n   PROVENANCE: {'PASS - every persona reached the API as written' if ok else 'FAIL'}")
    return ok


def human_reference(item_ids, cells=("Asian-Woman", "White-Man")):
    """What the real humans in those cells did, on exactly these items.

    The 30-point gap quoted at Checkpoint 3 is a pool-wide figure over all 350 items. The
    pilot runs 15. Comparing the model against the pool-wide gap would be comparing across
    different item sets, so the reference is recomputed on the pilot's own items.
    """
    df = D.load_350()
    r = df[df.rater_race.isin(RACE3)].copy()
    r["cell"] = r.rater_race.map(RACE3) + "-" + r.rater_gender
    r = r[r.cell.isin(cells) & r.item_id.isin(item_ids)]
    out = {}
    for cell, sub in r.groupby("cell"):
        n_raters = sub.rater_id.nunique()
        out[cell] = dict(
            n_raters=n_raters, n_ratings=len(sub),
            p_yes=float((sub[D.LABEL_COL] == "Yes").mean()),
            p_no=float((sub[D.LABEL_COL] == "No").mean()),
            p_unsure=float((sub[D.LABEL_COL] == "Unsure").mean()),
            per_item=sub.groupby("item_id")[D.LABEL_COL]
                        .apply(lambda s: float((s == "Yes").mean())))
    return out


def verdict(d, piv, a="Asian-Woman#1", b="White-Man#1", provenance_ok=True,
            within=("Asian-Woman#1", "Asian-Woman#2")):
    """The capability call. Prints loudly either way.

    CORRECTED AFTER THE FIRST PILOT. The original version branched only on the number of
    between-cell disagreements, which is wrong: with one sample per persona at temperature
    1.0, a between-cell difference confounds persona conditioning with sampling noise. The
    within-cell pair (two distinct personas from the SAME cell) is the noise floor, and a
    between-cell difference only counts as evidence of conditioning if it EXCEEDS it.

    On the first pilot the original logic printed "THE MODEL DOES RESPOND" on 2/15
    between-cell differences while the within-cell pair differed on 4/15. That reading was
    too generous and is superseded here.
    """
    print("\n" + "=" * 96)
    print("PERSONA SENSITIVITY VERDICT")
    print("=" * 96)
    items = sorted(piv.index)
    href = human_reference(items)
    print(f"   HUMAN REFERENCE on these same {len(items)} items:")
    for cell, v in href.items():
        print(f"      {cell:<14} {v['n_raters']:>3} raters, {v['n_ratings']:>5} ratings   "
              f"P(Yes)={v['p_yes']:.3f}  P(No)={v['p_no']:.3f}  P(Unsure)={v['p_unsure']:.3f}")
    if len(href) == 2:
        (ca, va), (cb, vb) = sorted(href.items())
        gap_h = abs(va["p_yes"] - vb["p_yes"])
        print(f"      human gap in P(Yes) between the two cells: {100*gap_h:.1f} points")
    else:
        gap_h = float("nan")

    if a not in piv.columns or b not in piv.columns:
        print(f"\n   {a} or {b} missing; cannot judge sensitivity.")
        return
    both = piv[[a, b]].dropna()
    n_diff = int((both[a] != both[b]).sum())
    pa = float((both[a] == "Yes").mean())
    pb = float((both[b] == "Yes").mean())
    print(f"\n   MODEL on the same items ({len(both)} items, 1 sample per condition):")
    print(f"      {a:<16} P(Yes)={pa:.3f}   {dict(both[a].value_counts())}")
    print(f"      {b:<16} P(Yes)={pb:.3f}   {dict(both[b].value_counts())}")
    print(f"      model gap in P(Yes): {100*abs(pa-pb):.1f} points")
    print(f"      items where the two personas disagree: {n_diff} / {len(both)}")

    # ---- the noise floor: two DISTINCT personas from the SAME cell -------------------
    n_within = None
    if within[0] in piv.columns and within[1] in piv.columns:
        wb = piv[list(within)].dropna()
        n_within = int((wb[within[0]] != wb[within[1]]).sum())
        print(f"\n   NOISE FLOOR - {within[0]} vs {within[1]} (same cell):")
        print(f"      items where they disagree: {n_within} / {len(wb)}")
        print(f"      these two share race and gender, so any disagreement here is NOT")
        print(f"      demographic conditioning. It is persona-wording plus sampling noise.")
        print(f"\n   BETWEEN-cell {n_diff}/{len(both)}  vs  WITHIN-cell {n_within}/{len(wb)}")

    print("\n   " + "-" * 90)
    if not provenance_ok:
        print("   VERDICT: INDETERMINATE.")
        print("   The persona text could not be confirmed as sent, so no capability claim")
        print("   can be made. Fix provenance and re-run.")
    elif n_diff == 0:
        print("   *** VERDICT: MODEL CAPABILITY PROBLEM - FLAGGED ***")
        print("")
        print("   The persona text was verifiably sent and the model produced IDENTICAL")
        print("   ratings on every item for two personas whose real human counterparts")
        print(f"   differ by {100*gap_h:.1f} points of P(Yes) on these same items.")
        print("")
        print("   This is a statement about qwen3.8-27b, NOT a finding about persona")
        print("   prompting. The paper CANNOT claim 'personas fail to reproduce human")
        print("   diversity' from this - the instrument did not respond to the manipulation")
        print("   at all, so the experiment has no measurable treatment.")
        print("")
        print("   What the paper CAN say: this model does not condition its safety ratings")
        print("   on demographic persona text under this prompt. That is a real and")
        print("   reportable negative, and it is a limitation of the model tested.")
        print("")
        print("   Before concluding, try in this order:")
        print("      1. a different model (the Groq fallback list, or gpt-oss-120b on a")
        print("         reduced item count given its 200K TPD cap)")
        print("      2. temperature is already 1.0; confirm it is being honoured")
        print("      3. a stronger persona framing as a SENSITIVITY CHECK ONLY, reported")
        print("         separately - do not silently swap it into the main design")
        print("   Do NOT proceed to the 3,675-call run until one of these produces variation.")
    elif n_diff <= 1:
        print(f"   *** VERDICT: NEAR-DEGENERATE - {n_diff} differing item of {len(both)} ***")
        print("   Barely above zero. On 15 items this is indistinguishable from sampling")
        print("   noise at temperature 1.0. Treat as a probable capability problem and run")
        print("   the checks above before committing to the full run.")
    elif n_within is not None and n_diff <= n_within:
        print("   *** VERDICT: PERSONA CONDITIONING NOT DEMONSTRATED ***")
        print("")
        print(f"   The output does vary ({n_diff}/{len(both)} between-cell differences), but it")
        print(f"   varies AT LEAST AS MUCH between two personas from the SAME cell")
        print(f"   ({n_within}/{len(both)}). Two personas sharing race and gender cannot differ")
        print("   BECAUSE of race or gender, so that variation is noise. A demographic")
        print("   effect that does not exceed its own noise floor has not been shown.")
        print("")
        print("   This is NOT the same as the model being insensitive - the output is not")
        print("   frozen. It is that the variation present is not attributable to the")
        print("   manipulation. On 15 items with one sample each, the two cannot be")
        print("   separated, so this is UNRESOLVED rather than negative.")
        print("")
        print("   DO NOT read this as 'personas fail to reproduce human diversity'. The")
        print("   design cannot support that claim yet.")
        print("")
        print("   Decisive next step, cheap: re-run ONE persona on the SAME items several")
        print("   times to measure the pure sampling noise floor directly, then test whether")
        print("   the between-cell difference exceeds it. Until that is done, neither a")
        print("   positive nor a negative persona finding is supportable.")
    else:
        print(f"   VERDICT: THE MODEL DOES RESPOND to persona conditioning.")
        print(f"   {n_diff} of {len(both)} items differ between the two personas, exceeding the")
        print(f"   within-cell noise floor of {n_within}. There is a measurable treatment")
        print(f"   effect and the full run is worth executing.")
        print(f"   Whether the model moves in the HUMAN direction is a separate question,")
        print(f"   answered by the full run, not by 15 items.")
        if np.isfinite(gap_h) and gap_h > 0:
            same_dir = np.sign(pa - pb) == np.sign(
                href.get("Asian-Woman", {}).get("p_yes", 0)
                - href.get("White-Man", {}).get("p_yes", 0))
            print(f"\n   directional preview (15 items, not inferential): model gap "
                  f"{100*(pa-pb):+.1f} pts vs human gap "
                  f"{100*(href['Asian-Woman']['p_yes']-href['White-Man']['p_yes']):+.1f} pts"
                  f"  -> {'SAME' if same_dir else 'OPPOSITE'} direction")
    print("   " + "-" * 90)


def baseline_degeneracy(d, piv, baseline="baseline#1"):
    print("\n" + "=" * 96)
    print("BASELINE DEGENERACY")
    print("=" * 96)
    if baseline not in piv.columns:
        print(f"   {baseline} missing")
        return
    v = piv[baseline].dropna()
    vc = v.value_counts()
    print(f"   {baseline} over {len(v)} items: {dict(vc)}")
    top = vc.iloc[0] / len(v)
    print(f"   modal share: {top:.3f}")
    if top == 1.0:
        print(f"   *** the baseline is CONSTANT - it returns '{vc.index[0]}' on every item.")
        print(f"       A constant baseline has zero spread by construction, which makes it a")
        print(f"       degenerate floor for Phase 3 rather than an informative one. Report it")
        print(f"       as such; it is still a valid floor, but it carries no information about")
        print(f"       item difficulty.")
    elif top >= 0.9:
        print(f"   the baseline is near-constant ({top:.0%} one label). Note it in Phase 3.")
    else:
        print(f"   the baseline varies across items; it is an informative floor.")
