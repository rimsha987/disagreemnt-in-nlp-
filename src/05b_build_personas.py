"""Derive the within-cell (age, education) profiles from the real rater pool and verify
they match the values frozen in src/personas.py.

src/personas.py hardcodes CELL_PROFILES so the prompt text is version-controlled and cannot
drift silently between runs. This script is the check that those hardcoded values are in fact
what the data says. If it fails, personas.py is wrong - fix personas.py, do not fix this.

Selection rule, deterministic: within each cell take the three most frequent
(rater_age, rater_education) profiles; break frequency ties by personas.AGE_ORDER then
personas.EDU_ORDER.

Outputs: results/05b_persona_profiles.csv
"""
import sys
import pandas as pd
sys.path.insert(0, "src")
import dices_io as D, personas as P

RACE3 = {"White": "White", "Black/African American": "Black",
         "Asian/Asian subcontinent": "Asian"}


def main():
    df = D.load_350()
    r = D.rater_table(df)
    r = r[r.rater_race.isin(RACE3)].copy()
    r["cell"] = r.rater_race.map(RACE3) + "-" + r.rater_gender
    assert sorted(r.cell.unique()) == sorted(P.CELLS), sorted(r.cell.unique())

    print("=" * 92)
    print("WITHIN-CELL (age, education) PROFILES, derived from the real rater pool")
    print("=" * 92)
    rows, ok = [], True
    for cell in P.CELLS:
        sub = r[r.cell == cell]
        counts = sub.groupby(["rater_age", "rater_education"]).size().reset_index(name="n")
        counts["age_rank"] = counts.rater_age.map(P.AGE_ORDER.index)
        counts["edu_rank"] = counts.rater_education.map(P.EDU_ORDER.index)
        counts = counts.sort_values(["n", "age_rank", "edu_rank"],
                                    ascending=[False, True, True])
        top3 = [(a, e) for a, e in zip(counts.rater_age, counts.rater_education)][:3]
        frozen = P.CELL_PROFILES[cell]
        match = top3 == frozen
        ok &= match
        print(f"\n-- {cell}  (n={len(sub)} raters, {len(counts)} distinct profiles) --")
        for i, (a, e) in enumerate(top3):
            n = int(counts[(counts.rater_age == a) & (counts.rater_education == e)].n.iloc[0])
            print(f"   persona {i+1}: {a:<10} | {e:<26} | {n} real raters have this profile")
            rows.append(dict(cell=cell, persona=P.persona_id(cell, i), k=i + 1,
                             rater_age=a, rater_education=e, n_real_raters=n,
                             n_cell_raters=len(sub), n_distinct_profiles=len(counts)))
        print(f"   frozen in personas.py: {'MATCH' if match else '*** MISMATCH ***'}")
        if not match:
            print(f"      data says   {top3}")
            print(f"      frozen says {frozen}")
        if len(counts) < 3:
            print(f"   *** WARNING: only {len(counts)} distinct profiles; personas would repeat")

    pd.DataFrame(rows).to_csv("results/05b_persona_profiles.csv", index=False)
    print("\n" + "=" * 92)
    print(f"VERIFICATION: {'PASS - personas.py matches the data' if ok else 'FAIL'}")
    print("=" * 92)
    if not ok:
        sys.exit(1)

    texts = {}
    for cell in P.CELLS:
        for k in range(3):
            texts[P.persona_id(cell, k)] = P.persona_text(cell, k)
    texts[P.BASELINE_ID] = P.baseline_text()
    n_uniq = len(set(texts.values()))
    print(f"\n   condition texts generated: {len(texts)}   distinct: {n_uniq}")
    assert n_uniq == 19, f"expected 19 distinct condition texts, got {n_uniq}"
    print(f"   all 18 persona texts are distinct from each other and from the baseline: yes")
    print(f"\n   wrote results/05b_persona_profiles.csv")


if __name__ == "__main__":
    main()
