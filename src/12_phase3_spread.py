"""Phase 3 (RQ3): does filtering compress between-group spread?

Human conditions only. The LLM arm was stopped at Checkpoint 3 (FINDINGS 2.13); its measured
values are carried into the headline figure by src/13_phase3_figure.py, marked as sitting
inside the same-persona noise floor.

DEFINITION. For a set of raters and a demographic grouping, per item build each group's soft
label distribution over (No, Yes, Unsure), then take the mean pairwise distance between groups.
Average over items. TVD primary, JSD secondary (FINDINGS 1.5).

TWO NULLS, BOTH NEEDED. They answer different questions and are easy to confuse.

  NULL A - label permutation, group sizes preserved.
      Shuffle which rater belongs to which group. Answers: is the observed spread real, or is
      it what any partition of this size would show from finite-sample noise?
      This is what makes the AGE NEGATIVE CONTROL interpretable. Note carefully: a negative
      control does NOT predict near-zero RAW spread - raw spread can never be zero, because
      each group's distribution is estimated from 10-56 raters and sampling noise alone forces
      the groups apart. What it predicts is near-zero EXCESS over this null. Reporting raw
      spread for a control would look like a failure of the control when it is arithmetic.

  NULL B - random removal of 19 raters.
      Answers: does the PUBLISHED filter compress spread more than removing 19 arbitrary
      raters? Same construction as Phase 1.

Also reported per the brief: mean distance from the pooled ("majority/average") position.

Outputs: results/12_spread_by_condition.csv, results/12_spread_nullA.csv,
         results/12_spread_nullB.csv, results/12_spread_breakdowns.csv
"""
import sys, itertools, json
import numpy as np, pandas as pd
sys.path.insert(0, "src")
import dices_io as D

SEED = 20260826
N_PERM_A = 5000         # label permutations, per condition x grouping.
                        # Raised from 500 so Table 6 and Table 9 converge and the
                        # 'two independent permutation runs' footnote can be dropped.
N_PERM_B = 5000         # random 19-rater removals. Raised from 1000: at 1000 the
                        # gender p-value moved 0.026 -> 0.035 purely from RNG ordering,
                        # which is too unstable to report to three decimals.
RACE3 = {"White": "White", "Black/African American": "Black",
         "Asian/Asian subcontinent": "Asian"}


# ------------------------------------------------------------------------- distributions
def group_dists(oh, idx_list):
    """groups x items x 3 normalised distributions. idx_list = rater column indices."""
    P = []
    for idx in idx_list:
        c = oh[:, idx, :].sum(axis=1)
        tot = c.sum(axis=1, keepdims=True)
        P.append(np.divide(c, tot, out=np.full_like(c, np.nan), where=tot > 0))
    return np.stack(P)


def _H(x):
    return -(np.where(x > 0, x * np.log2(np.where(x > 0, x, 1)), 0.0)).sum(axis=-1)


def spread(oh, idx_list):
    """(mean pairwise TVD, mean pairwise JSD, mean TVD to the pooled position).

    Pooled position = distribution over ALL raters in the grouping, per item. Items where any
    group has zero raters are dropped from that grouping's average, and the count is returned.
    """
    P = group_dists(oh, idx_list)                       # G x I x 3
    ok = ~np.isnan(P).any(axis=(0, 2))                  # items where every group is populated
    P = P[:, ok, :]
    tv, js = [], []
    for a, b in itertools.combinations(range(P.shape[0]), 2):
        tv.append(0.5 * np.abs(P[a] - P[b]).sum(axis=1))
        m = 0.5 * (P[a] + P[b])
        js.append(_H(m) - 0.5 * (_H(P[a]) + _H(P[b])))
    allidx = np.concatenate(idx_list)
    c = oh[:, allidx, :].sum(axis=1)[ok]
    pooled = c / c.sum(axis=1, keepdims=True)
    to_pool = np.mean([0.5 * np.abs(P[g] - pooled).sum(axis=1) for g in range(P.shape[0])],
                      axis=0)
    return (float(np.mean(tv)), float(np.mean(js)), float(np.mean(to_pool)), int(ok.sum()),
            np.mean(tv, axis=0))          # last: per-item mean pairwise TVD, for breakdowns


def main():
    rng = np.random.default_rng(SEED)
    df = D.load_350()
    code = {l: i for i, l in enumerate(D.LABELS)}
    piv = df.pivot(index="item_id", columns="rater_id", values=D.LABEL_COL)
    Mi = np.vectorize(code.get)(piv.to_numpy())
    oh = np.zeros((*Mi.shape, 3))
    np.put_along_axis(oh, Mi[:, :, None], 1, axis=2)
    rater_ids = list(piv.columns)
    item_ids = list(piv.index)
    r = D.rater_table(df).set_index("rater_id").loc[rater_ids].reset_index()
    assert len(r) == 123 and len(item_ids) == 350
    print(f"loaded {oh.shape[0]} items x {oh.shape[1]} raters (all 350 items, full pool)")

    r["race3"] = r.rater_race.map(RACE3)
    r["cell6"] = np.where(r.race3.notna(), r.race3 + "-" + r.rater_gender, None)

    # All six candidate schemes are computed here so that the candidate-selection
    # table and the main spread table are produced from ONE permutation run.
    r["race3only"] = r.race3
    r["gender_age"] = r.rater_gender + "-" + r.rater_age
    GROUPINGS = {
        "race3 x gender2 (6 chosen cells)": r.cell6.to_numpy(),
        "gender2 x age3":                   r.gender_age.to_numpy(),
        "gender2":                          r.rater_gender.to_numpy(),
        "race3 (White/Black/Asian only)":   r.race3only.to_numpy(),
        "race5":                            r.rater_race.to_numpy(),
        "age3 (NEGATIVE CONTROL)":          r.rater_age.to_numpy(),
    }
    CONDITIONS = {
        "All 123 human raters":      np.ones(len(r), bool),
        "104 kept (filtered)":       (~r.removed).to_numpy(),
    }

    def idx_for(labels, cond_mask):
        # a rater outside the grouping (e.g. Latine/x under race3 x gender2) carries NaN or
        # None, not a group name; both must be excluded before sorting
        names = sorted({g for g, m in zip(labels, cond_mask)
                        if m and isinstance(g, str)})
        return names, [np.where((labels == g) & cond_mask)[0] for g in names]

    # ------------------------------------------------------ observed + NULL A per cell
    rows, nullA_rows, per_item_store = [], [], {}
    print("\n" + "=" * 100)
    print("BETWEEN-GROUP SPREAD BY CONDITION AND GROUPING")
    print("TVD primary. NULL A = shuffle rater->group labels, group sizes preserved.")
    print("=" * 100)
    for gname, labels in GROUPINGS.items():
        print(f"\n### {gname}")
        print(f"   {'condition':<24} {'groups':>7} {'raters':>7} {'TVD':>8} {'nullA':>8} "
              f"{'excess':>8} {'z':>7} {'JSD':>9} {'to-pool':>8}")
        for cname, cmask in CONDITIONS.items():
            names, idx = idx_for(labels, cmask)
            sizes = [len(i) for i in idx]
            tv, js, tp, n_ok, per_item = spread(oh, idx)
            per_item_store[(gname, cname)] = per_item

            assigned = np.concatenate(idx)
            null = np.empty(N_PERM_A)
            for k in range(N_PERM_A):
                perm = rng.permutation(assigned)
                cuts, s = [], 0
                for n in sizes:
                    cuts.append(perm[s:s + n]); s += n
                null[k] = spread(oh, cuts)[0]
            z = (tv - null.mean()) / null.std()
            print(f"   {cname:<24} {len(names):>7} {sum(sizes):>7} {tv:>8.5f} "
                  f"{null.mean():>8.5f} {tv-null.mean():>+8.5f} {z:>+7.2f} {js:>9.6f} {tp:>8.5f}")
            rows.append(dict(grouping=gname, condition=cname, n_groups=len(names),
                             n_raters=sum(sizes), group_sizes=";".join(map(str, sizes)),
                             n_items=n_ok, tvd=tv, jsd=js, tvd_to_pooled=tp,
                             nullA_tvd=null.mean(), nullA_sd=null.std(),
                             excess=tv - null.mean(), z=z))
            nullA_rows.append(dict(grouping=gname, condition=cname,
                                   null_mean=null.mean(), null_sd=null.std(),
                                   null_lo=np.percentile(null, 2.5),
                                   null_hi=np.percentile(null, 97.5), n_perm=N_PERM_A))
    out = pd.DataFrame(rows)
    out.to_csv("results/12_spread_by_condition.csv", index=False)
    pd.DataFrame(nullA_rows).to_csv("results/12_spread_nullA.csv", index=False)

    # --------------------------------------------------------------- negative control
    print("\n" + "=" * 100)
    print("NEGATIVE CONTROL CHECK")
    print("=" * 100)
    ctl = out[out.grouping.str.contains("NEGATIVE")]
    real = out[~out.grouping.str.contains("NEGATIVE")]
    print("   The control predicts near-zero EXCESS over the size-matched null, NOT near-zero")
    print("   raw spread. Raw spread cannot be zero: each group's distribution rests on 10-56")
    print("   raters, and sampling noise alone separates them.\n")
    print(f"   {'grouping':<34} {'condition':<24} {'raw TVD':>9} {'excess':>9} {'z':>7}")
    for t in out.itertuples():
        mark = "  <- control" if "NEGATIVE" in t.grouping else ""
        print(f"   {t.grouping:<34} {t.condition:<24} {t.tvd:>9.5f} {t.excess:>+9.5f} "
              f"{t.z:>+7.2f}{mark}")
    ok = (ctl.z.abs() < 2).all() and (real.z > 2).any()
    print(f"\n   control |z| < 2 in both conditions: {bool((ctl.z.abs()<2).all())}")
    print(f"   at least one real grouping z > 2:    {bool((real.z>2).any())}")
    print(f"   -> the measure {'BEHAVES AS PREDICTED' if ok else 'DOES NOT behave as predicted'}"
          f": it separates real demographic structure from noise.")

    # --------------------------------------------------- NULL B: filtering vs random 19
    print("\n" + "=" * 100)
    print("NULL B - DOES FILTERING COMPRESS SPREAD MORE THAN REMOVING 19 RATERS AT RANDOM?")
    print(f"{N_PERM_B} random removals, seed {SEED}")
    print("=" * 100)
    rngB = np.random.default_rng(SEED)
    nb_rows = []
    # Null B answers the RQ3 question and is reported over the four RQ3 groupings.
    # The two extra schemes above exist only to populate the cell-selection table.
    NULLB = [g for g in GROUPINGS if g not in
             ("gender2 x age3", "race3 (White/Black/Asian only)")]
    print(f"   {'grouping':<34} {'all123':>9} {'kept104':>9} {'delta':>9} {'nullB mean':>11} "
          f"{'z':>7} {'p':>7}")
    for gname in NULLB:
        labels = GROUPINGS[gname]
        names_all, idx_all = idx_for(labels, np.ones(len(r), bool))
        tv_all = spread(oh, idx_all)[0]
        names_k, idx_k = idx_for(labels, (~r.removed).to_numpy())
        tv_kept = spread(oh, idx_k)[0]
        delta = tv_kept - tv_all
        null = np.empty(N_PERM_B)
        for k in range(N_PERM_B):
            drop = rngB.choice(len(r), 19, replace=False)
            m = np.ones(len(r), bool); m[drop] = False
            _, idx = idx_for(labels, m)
            null[k] = spread(oh, idx)[0] - tv_all
        z = (delta - null.mean()) / null.std()
        p = float((np.abs(null - null.mean()) >= abs(delta - null.mean())).mean())
        print(f"   {gname:<34} {tv_all:>9.5f} {tv_kept:>9.5f} {delta:>+9.5f} "
              f"{null.mean():>+11.5f} {z:>+7.2f} {p:>7.3f}")
        nb_rows.append(dict(grouping=gname, tvd_all=tv_all, tvd_kept=tv_kept, delta=delta,
                            nullB_mean=null.mean(), nullB_sd=null.std(),
                            nullB_lo=np.percentile(null, 2.5),
                            nullB_hi=np.percentile(null, 97.5), z=z, p_perm=p,
                            n_perm=N_PERM_B))
    pd.DataFrame(nb_rows).to_csv("results/12_spread_nullB.csv", index=False)
    print(f"\n   delta > 0 means filtering INCREASED measured spread; < 0 means it compressed it.")
    print(f"   Removing raters shrinks every group, and smaller groups are noisier, so the null")
    print(f"   itself is positive. That is why the comparison is against the null, not zero.")

    # ------------------------------------------------------------------- breakdowns
    print("\n" + "=" * 100)
    print("BREAKDOWN BY ENTROPY TERTILE AND DEGREE OF HARM (primary grouping, TVD)")
    print("=" * 100)
    meta = pd.read_csv("results/03_item_jsd.csv")[["item_id", "ent_bin"]]
    harm = df.drop_duplicates("item_id")[["item_id", "degree_of_harm"]]
    meta = meta.merge(harm, on="item_id", validate="one_to_one").set_index("item_id")
    meta = meta.loc[item_ids]
    PRIM = "race3 x gender2 (6 chosen cells)"
    bd = []
    for split, order in [("ent_bin", ["low", "medium", "high"]),
                         ("degree_of_harm", ["Extreme", "Moderate", "Benign", "Debatable"])]:
        print(f"\n   -- by {split} --")
        print(f"      {'level':<12} {'n items':>8} {'all123':>9} {'kept104':>9} {'delta':>9}")
        for lev in order:
            mask = (meta[split] == lev).to_numpy()
            if mask.sum() == 0:
                continue
            a = float(np.mean(per_item_store[(PRIM, "All 123 human raters")][mask]))
            k = float(np.mean(per_item_store[(PRIM, "104 kept (filtered)")][mask]))
            star = "   <- primary" if lev in ("Extreme", "Moderate") and split == "degree_of_harm" else ""
            print(f"      {lev:<12} {int(mask.sum()):>8} {a:>9.5f} {k:>9.5f} {k-a:>+9.5f}{star}")
            bd.append(dict(grouping=PRIM, split=split, level=lev, n_items=int(mask.sum()),
                           tvd_all=a, tvd_kept=k, delta=k - a))
    pd.DataFrame(bd).to_csv("results/12_spread_breakdowns.csv", index=False)
    print("\n   wrote results/12_spread_by_condition.csv, _nullA.csv, _nullB.csv, "
          "_breakdowns.csv")

    np.save("results/12_per_item_tvd.npy",
            np.stack([per_item_store[(PRIM, c)] for c in CONDITIONS]))
    with open("results/12_per_item_meta.json", "w") as f:
        json.dump(dict(grouping=PRIM, conditions=list(CONDITIONS), item_ids=item_ids), f)


if __name__ == "__main__":
    main()
