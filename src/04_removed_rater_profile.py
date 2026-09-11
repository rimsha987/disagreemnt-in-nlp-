"""Item 8 (Checkpoint 2): do the 19 removed raters actually look like low-quality raters?

DESCRIPTIVE ONLY. No threshold is fitted, no proxy filter is constructed, and nothing here
feeds any other analysis. The point is to check whether the published removal list is
consistent with the criteria the DICES authors state, since we cannot see the criteria
themselves.

Per-rater signals, all computed over that rater's 350 ratings:
  * median answer_time_ms          - speed
  * modal-label share and entropy  - constant / straight-line labelling on Q_overall
  * granular straight-lining       - share of rows where all 20 Q2-Q6 answers are identical
  * majority agreement             - share of items where the rater matches the pool majority
  * Q1 non-comprehension rate      - share of rows flagged nonsensical / unfamiliar / non-English

Outputs: results/04_rater_profile.csv
"""
import sys
import numpy as np, pandas as pd
from scipy import stats
sys.path.insert(0, "src")
import dices_io as D

GRANULAR = [
    "Q2_harmful_content_medical_legal_financial_or_relationship_advice",
    "Q2_harmful_content_dangerous_content", "Q2_harmful_content_derogation_of_boat",
    "Q2_harmful_content_insensitive", "Q2_harmful_content_obscene_and_profane",
    "Q2_harmful_content_personal_information", "Q2_harmful_content_regulated_goods",
    "Q2_harmful_content_sexually_suggestive_content", "Q2_harmful_content_other",
    "Q3_bias_targeting_beliefs", "Q3_bias_targeting_gender_sexual_orientation",
    "Q3_bias_targeting_inherited_attributes", "Q3_bias_targeting_status", "Q3_bias_other",
    "Q4_misinformation", "Q5_political_affiliation",
    "Q6_policy_guidelines_polarizing_topics", "Q6_policy_guidelines_making_endorsement",
    "Q6_policy_guidelines_other",
]
Q1_OK = "None of the above - it is in English, it makes sense, and I am familiar with the topic"


def entropy2(counts):
    p = np.asarray(counts, float)
    p = p[p > 0] / p.sum()
    return float(-(p * np.log2(p)).sum()) + 0.0   # + 0.0 normalises -0.0


def main():
    df = D.load_350()
    for c in GRANULAR:
        assert c in df.columns, c
    print("=" * 96)
    print("DESCRIPTIVE PROFILE OF THE 19 REMOVED RATERS")
    print("no thresholds fitted, nothing here feeds another analysis")
    print("=" * 96)

    # pool majority label per item, from ALL 123 raters
    maj = (df.groupby("item_id")[D.LABEL_COL].agg(lambda s: s.value_counts().idxmax())
             .rename("pool_majority"))
    df = df.merge(maj, on="item_id", validate="many_to_one")
    assert len(df) == 43050

    df["_agree"] = df[D.LABEL_COL] == df.pool_majority
    df["_q1_bad"] = df.Q1_whole_conversation_evaluation != Q1_OK
    df["_flat"] = df[GRANULAR].nunique(axis=1) == 1

    g = df.groupby("rater_id")
    prof = pd.DataFrame({
        "removed": g.removed.first(),
        "gender": g.rater_gender.first(),
        "race": g.rater_race.first(),
        "age": g.rater_age.first(),
        "n_ratings": g.size(),
        "median_time_ms": g.answer_time_ms.median(),
        "mean_time_ms": g.answer_time_ms.mean(),
        "modal_share": g[D.LABEL_COL].agg(lambda s: s.value_counts().iloc[0] / len(s)),
        "label_entropy": g[D.LABEL_COL].agg(lambda s: entropy2(s.value_counts().to_numpy())),
        "pct_yes": g[D.LABEL_COL].agg(lambda s: (s == "Yes").mean()),
        "granular_flat_share": g._flat.mean(),
        "majority_agreement": g._agree.mean(),
        "q1_noncomprehension": g._q1_bad.mean(),
    }).reset_index()
    assert len(prof) == 123 and prof.n_ratings.eq(350).all()
    prof.to_csv("results/04_rater_profile.csv", index=False)

    METRICS = [
        ("median_time_ms", "median answer time (ms)", "lower = faster", 0),
        ("modal_share", "modal-label share", "higher = more constant", 4),
        ("label_entropy", "label entropy (bits, max 1.585)", "lower = more constant", 4),
        ("granular_flat_share", "share of rows with all 20 sub-answers identical", "higher = flatter", 4),
        ("majority_agreement", "agreement with pool majority", "lower = more deviant", 4),
        ("q1_noncomprehension", "Q1 non-comprehension rate", "higher = more flagged", 4),
        ("pct_yes", "share of ratings 'Yes'", "context only", 4),
    ]
    rm, kp = prof[prof.removed], prof[~prof.removed]
    print(f"\n   removed n={len(rm)}   kept n={len(kp)}   (per-rater values over 350 ratings each)")
    print(f"\n   {'metric':<48} {'removed med':>12} {'kept med':>11} {'MWU p':>9} {'CLES':>7}")
    out = []
    for col, label, _, dec in METRICS:
        a, b = rm[col].to_numpy(float), kp[col].to_numpy(float)
        u = stats.mannwhitneyu(a, b, alternative="two-sided")
        cles = u.statistic / (len(a) * len(b))     # P(removed > kept), ties at 0.5
        fmt = f"{{:.{dec}f}}"
        print(f"   {label:<48} {fmt.format(np.median(a)):>12} {fmt.format(np.median(b)):>11} "
              f"{u.pvalue:>9.4f} {cles:>7.3f}")
        out.append(dict(metric=col, label=label, removed_median=np.median(a),
                        kept_median=np.median(b), removed_mean=a.mean(), kept_mean=b.mean(),
                        mwu_p=u.pvalue, cles=cles, n_removed=len(a), n_kept=len(b)))
    print(f"\n   CLES = P(a removed rater scores higher than a kept rater); 0.5 = no separation.")
    print(f"   Mann-Whitney p-values are DESCRIPTIVE: 7 metrics on the same 123 raters, no")
    print(f"   correction applied, and the removal list was not defined by any of them.")
    pd.DataFrame(out).to_csv("results/04_profile_tests.csv", index=False)

    print("\n" + "=" * 96)
    print("ALL 19 REMOVED RATERS, INDIVIDUALLY")
    print("=" * 96)
    cols = ["rater_id", "gender", "race", "age", "median_time_ms", "modal_share",
            "label_entropy", "granular_flat_share", "majority_agreement", "q1_noncomprehension"]
    show = rm[cols].sort_values("median_time_ms")
    show = show.assign(rater_id=show.rater_id.str[-6:])
    print(show.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    print("\n   kept-pool reference (n=104):")
    for col in ["median_time_ms", "modal_share", "label_entropy", "granular_flat_share",
                "majority_agreement", "q1_noncomprehension"]:
        q = kp[col].quantile([0, .05, .25, .5, .75, .95, 1]).to_numpy()
        print(f"      {col:<24} min {q[0]:>10.3f}  p5 {q[1]:>9.3f}  p25 {q[2]:>9.3f}  "
              f"med {q[3]:>9.3f}  p75 {q[4]:>9.3f}  p95 {q[5]:>9.3f}  max {q[6]:>10.3f}")

    print("\n" + "=" * 96)
    print("HOW MANY REMOVED RATERS ARE EXTREME ON EACH SIGNAL?")
    print("(counting against the KEPT pool's 5th/95th percentile, purely to describe overlap)")
    print("=" * 96)
    for col, label, direction, _ in METRICS[:-1]:
        if "lower" in direction:
            thr = kp[col].quantile(.05); n = int((rm[col] < thr).sum()); s = "below kept p5"
        else:
            thr = kp[col].quantile(.95); n = int((rm[col] > thr).sum()); s = "above kept p95"
        print(f"   {label:<48} {n:>2}/19 {s} ({thr:.4f})")

    n_any = 0
    for _, row in rm.iterrows():
        flags = [row.median_time_ms < kp.median_time_ms.quantile(.05),
                 row.modal_share > kp.modal_share.quantile(.95),
                 row.granular_flat_share > kp.granular_flat_share.quantile(.95),
                 row.majority_agreement < kp.majority_agreement.quantile(.05),
                 row.q1_noncomprehension > kp.q1_noncomprehension.quantile(.95)]
        n_any += any(flags)
    print(f"\n   removed raters extreme on AT LEAST ONE of the five signals: {n_any}/19")
    print(f"   removed raters extreme on NONE of them:                      {19-n_any}/19")

    print("\n" + "=" * 96)
    print("READING")
    print("=" * 96)
    print("   This cannot validate the filter, because the criteria are unpublished and the")
    print("   list is fixed. It can only say whether the removed raters look unusual on")
    print("   signals a quality check would plausibly use. Report the overlap honestly:")
    print("   where removed and kept raters overlap heavily on a signal, that signal is not")
    print("   what drove removal, or removal used information we cannot see.")
    print("\n   wrote results/04_rater_profile.csv, results/04_profile_tests.csv")


if __name__ == "__main__":
    main()
