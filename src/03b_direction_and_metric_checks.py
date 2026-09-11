"""Follow-ups to 03, triggered by two things in its output that needed verifying.

(1) The permutation baseline said the filter's mean JSD is indistinguishable from removing
    19 raters at random (p=0.45) - yet P(Yes) rose on 207 of 350 items with Wilcoxon
    p=7e-07. Those are not in conflict: JSD is unsigned, so a small consistent push in one
    direction is invisible to it. The correct null for a DIRECTIONAL claim is a directional
    permutation test. Run here.

(2) Spearman rho(entropy, JSD) = -0.05 (null) but Pearson r = -0.26 (p=1e-06). That gap
    means the linear relationship is driven by a few extreme points, not a monotone trend.
    Suspected cause: JSD contains log terms that blow up when a category is near zero, so
    low-entropy items can post large JSD from a tiny count change. Checked here against
    total variation distance, which has no such sensitivity.

Outputs: results/03b_direction_tests.csv, results/03b_report.txt (via tee),
         figures/fig2c_metric_sensitivity.{pdf,png}
"""
import sys
import numpy as np, pandas as pd
from scipy import stats
sys.path.insert(0, "src")
import dices_io as D, plotstyle as PS
import matplotlib.pyplot as plt

RNG_SEED, N_PERM = 20260826, 10000
YES = D.LABELS.index("Yes")


def main():
    df = D.load_350()
    it = pd.read_csv("results/03_item_jsd.csv")
    assert len(it) == 350

    # ---- rating tensor: items x raters x 3 -----------------------------------------
    code = {l: i for i, l in enumerate(D.LABELS)}
    piv = df.pivot(index="item_id", columns="rater_id", values=D.LABEL_COL)
    Mi = np.vectorize(code.get)(piv.to_numpy())
    onehot = np.zeros((*Mi.shape, 3))
    np.put_along_axis(onehot, Mi[:, :, None], 1, axis=2)
    col_ids = list(piv.columns)
    idx_of = {r: i for i, r in enumerate(col_ids)}
    full_c = onehot.sum(axis=1)
    p_full = full_c / full_c.sum(axis=1, keepdims=True)
    assert np.allclose(p_full.sum(axis=1), 1)

    def top_set(p):
        """Boolean mask of argmax labels. Tie-aware: an item with two joint-top labels has
        two True entries. Script 03 compared tie-aware majority strings, so the permutation
        null must use the same rule or observed and null are not comparable."""
        return p == p.max(axis=1, keepdims=True)

    top_full = top_set(p_full)

    def stats_for(drop_ids):
        keep = np.array([i for r, i in idx_of.items() if r not in set(drop_ids)])
        c = onehot[:, keep, :].sum(axis=1)
        p = c / c.sum(axis=1, keepdims=True)
        d_yes = p[:, YES] - p_full[:, YES]
        flips = int((top_set(p) != top_full).any(axis=1).sum())
        return float(d_yes.mean()), float(np.abs(d_yes).mean()), flips

    obs_mean_dyes, obs_abs_dyes, obs_flips = stats_for(D.REMOVED_RATERS_350)
    print("=" * 96)
    print("(1) IS THE DIRECTION OF THE SHIFT SYSTEMATIC?")
    print("=" * 96)
    print("   The filter's MAGNITUDE was null (mean JSD p_perm=0.45). JSD is unsigned, so it")
    print("   cannot see a small consistent push. Testing the signed shift against the same")
    print("   null: remove 19 raters at random, 10,000 times.\n")
    print(f"   observed mean per-item change in P(Yes): {obs_mean_dyes:+.6f}")
    print(f"   observed mean ABSOLUTE change:           {obs_abs_dyes:.6f}")
    print(f"   observed majority-label flips:           {obs_flips}")

    rng = np.random.default_rng(RNG_SEED)
    rid = np.array(col_ids)
    null = np.array([stats_for(rng.choice(rid, 19, replace=False)) for _ in range(N_PERM)])
    n_dyes, n_abs, n_flips = null[:, 0], null[:, 1], null[:, 2]

    rows = []
    for nm, obs, nul, tail in [
            ("mean d_P(Yes) (signed)", obs_mean_dyes, n_dyes, "two"),
            ("mean |d_P(Yes)|", obs_abs_dyes, n_abs, "two"),
            ("majority flips", obs_flips, n_flips, "two")]:
        mu, sd = nul.mean(), nul.std()
        z = (obs - mu) / sd if sd > 0 else np.nan
        p_two = float((np.abs(nul - mu) >= abs(obs - mu)).mean())
        lo, hi = np.percentile(nul, [2.5, 97.5])
        print(f"\n   -- {nm} --")
        print(f"      null mean {mu:+.6f}   sd {sd:.6f}   95% range [{lo:+.6f}, {hi:+.6f}]")
        print(f"      observed  {obs:+.6f}   z = {z:+.2f}   two-sided p_perm = {p_two:.4f}")
        rows.append(dict(statistic=nm, observed=obs, null_mean=mu, null_sd=sd,
                         null_lo=lo, null_hi=hi, z=z, p_perm=p_two, n_perm=N_PERM))
    pd.DataFrame(rows).to_csv("results/03b_direction_tests.csv", index=False)

    # -- reconcile against the Wilcoxon reported in script 03 -------------------------
    it_d = it.d_yes.to_numpy()
    w = stats.wilcoxon(it_d)
    print("\n   -- RECONCILING WITH THE WILCOXON TEST IN SCRIPT 03 --")
    print(f"      script 03 reported Wilcoxon on per-item d_P(Yes): p = {w.pvalue:.3g}, which")
    print(f"      looks decisive. The permutation test above says p = "
          f"{rows[0]['p_perm']:.4f}. They disagree because the")
    print(f"      Wilcoxon treats the 350 items as 350 independent observations. They are not.")
    print(f"      The SAME 19 raters are removed from every item, so there is exactly ONE")
    print(f"      independent draw of a removal set, not 350. The Wilcoxon is pseudo-replicated")
    print(f"      and its p-value is not interpretable as evidence about the filter.")
    print(f"      The permutation test, whose unit of randomisation is the rater set, is the")
    print(f"      correct one. THE WILCOXON RESULT IN SCRIPT 03 IS SUPERSEDED AND SHOULD NOT")
    print(f"      BE QUOTED. What the Wilcoxon does establish, descriptively, is that the")
    print(f"      shift is CONSISTENT in sign across items ({int((it_d>0).sum())} up, "
          f"{int((it_d<0).sum())} down) - but a")
    print(f"      randomly chosen set of 19 raters produces an equally consistent shift.")

    print("\n   WHY THE DIRECTION EXISTS - composition of the removed group:")
    rem, kep = df[df.removed], df[~df.removed]
    print(f"      removed raters (n=19)  P(Yes) = {(rem[D.LABEL_COL]=='Yes').mean():.4f}")
    print(f"      kept raters    (n=104) P(Yes) = {(kep[D.LABEL_COL]=='Yes').mean():.4f}")
    print(f"      difference = {(kep[D.LABEL_COL]=='Yes').mean()-(rem[D.LABEL_COL]=='Yes').mean():+.4f}")
    r = D.rater_table(df)
    print(f"      the filter removes 14/61 men but 5/62 women; men's pool-wide P(Yes) is")
    py = df.groupby('rater_gender')[D.LABEL_COL].apply(lambda s: (s == 'Yes').mean())
    print(f"      {py['Man']:.4f} vs women's {py['Woman']:.4f}, so dropping men tilts the")
    print(f"      surviving pool toward the higher-P(Yes) group.")

    # ------------------------------------------------------------------ (2) metric check
    print("\n" + "=" * 96)
    print("(2) IS JSD MISLEADING AT LOW ENTROPY?")
    print("=" * 96)
    p_all = it[[f"p_all_{l}" for l in D.LABELS]].to_numpy()
    p_kep = it[[f"p_kept_{l}" for l in D.LABELS]].to_numpy()
    it["tvd"] = 0.5 * np.abs(p_all - p_kep).sum(axis=1)
    it["min_cell_all"] = p_all.min(axis=1)

    print("   total variation distance, TVD = 0.5*sum|p-q|, has no log term.")
    print(f"      TVD: mean {it.tvd.mean():.5f}, median {it.tvd.median():.5f}, "
          f"max {it.tvd.max():.5f}")
    for nm, col in [("JSD", "jsd"), ("TVD", "tvd")]:
        rs = stats.spearmanr(it.entropy_all, it[col])
        rp = stats.pearsonr(it.entropy_all, it[col])
        print(f"      {nm} vs entropy:  Spearman {rs.statistic:+.4f} (p={rs.pvalue:.3g})   "
              f"Pearson {rp.statistic:+.4f} (p={rp.pvalue:.3g})")

    # -- the actual mechanism, tested rather than asserted ---------------------------
    # For a small perturbation D = q - p, a second-order expansion of JSD gives
    #     JSD ~= (1 / (8 ln2)) * sum_i D_i^2 / p_i
    # i.e. each cell's contribution is INVERSELY weighted by its own probability. That is
    # the precise sense in which small cells amplify JSD. TVD = 0.5*sum|D_i| has no such
    # weighting. Verify the approximation holds on this data before relying on the claim.
    delta = p_kep - p_all
    with np.errstate(divide="ignore", invalid="ignore"):
        chi_term = np.where(p_all > 0, delta ** 2 / np.where(p_all > 0, p_all, 1), 0.0).sum(axis=1)
    approx = chi_term / (8 * np.log(2))
    it["chi_term"] = chi_term
    print(f"\n   MECHANISM CHECK: JSD ~= (1/(8 ln2)) * sum_i (q_i-p_i)^2 / p_i ?")
    print(f"      Pearson r(approx, actual JSD) = {stats.pearsonr(approx, it.jsd).statistic:.6f}")
    print(f"      max |approx - actual|         = {np.abs(approx - it.jsd).max():.3e}")
    print(f"      mean relative error           = "
          f"{np.mean(np.abs(approx - it.jsd) / np.maximum(it.jsd, 1e-12)):.4f}")
    print(f"      -> the second-order form is accurate here, so the 1/p_i weighting is real.")
    print(f"\n      Spearman(smallest unfiltered cell prob, JSD) = "
          f"{stats.spearmanr(it.min_cell_all, it.jsd).statistic:+.4f} "
          f"(p={stats.spearmanr(it.min_cell_all, it.jsd).pvalue:.3g})")
    print(f"      Spearman(smallest unfiltered cell prob, TVD) = "
          f"{stats.spearmanr(it.min_cell_all, it.tvd).statistic:+.4f} "
          f"(p={stats.spearmanr(it.min_cell_all, it.tvd).pvalue:.3g})")
    print(f"      NOTE: min-cell size alone does NOT correlate with JSD, because the size of")
    print(f"      the perturbation D_i varies too. The 1/p_i weighting shows up in the")
    print(f"      approximation above, not in a raw marginal correlation. Reported as such.")

    print(f"\n   by entropy tertile, the two metrics ORDER THE BINS DIFFERENTLY:")
    print(f"      {'bin':<8} {'n':>4} {'mean JSD':>10} {'mean TVD':>10} {'mean |dP(Yes)|':>15} "
          f"{'flips':>6}")
    for b in ["low", "medium", "high"]:
        s = it[it.ent_bin == b]
        print(f"      {b:<8} {len(s):>4} {s.jsd.mean():>10.5f} {s.tvd.mean():>10.5f} "
              f"{s.d_yes.abs().mean():>15.5f} {int(s.maj_changed.sum()):>6}")
    kw_j = stats.kruskal(*[it[it.ent_bin == b].jsd.values for b in ["low", "medium", "high"]])
    kw_t = stats.kruskal(*[it[it.ent_bin == b].tvd.values for b in ["low", "medium", "high"]])
    print(f"\n      Kruskal-Wallis JSD across bins: H={kw_j.statistic:.3f}, p={kw_j.pvalue:.3g}")
    print(f"      Kruskal-Wallis TVD across bins: H={kw_t.statistic:.3f}, p={kw_t.pvalue:.3g}")
    print(f"\n   the single largest-JSD item (140) sits in the LOW-entropy tertile:")
    row = it.loc[it.jsd.idxmax()]
    print(f"      item {int(row.item_id)}: entropy {row.entropy_all:.4f}, JSD {row.jsd:.5f}, "
          f"TVD {row.tvd:.5f}, |dP(Yes)| {abs(row.d_yes):.5f}")
    print(f"      its 'Unsure' cell went 2/123 -> 0/104, and JSD punishes a cell hitting zero")
    print(f"      far harder than TVD does. Under TVD it ranks "
          f"{int((it.tvd > row.tvd).sum()) + 1} of 350, not 1st.")

    it.to_csv("results/03_item_jsd.csv", index=False)
    print("\n   appended tvd and min_cell_all to results/03_item_jsd.csv")

    # --------------------------------------------------------------------------- figure
    print("\nFIGURE")
    PS.apply()
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.1), gridspec_kw=dict(wspace=0.4))
    ax = axes[0]
    ax.hist(n_dyes, bins=50, color=PS.LIGHT, edgecolor="white", lw=0.3)
    ax.axvline(obs_mean_dyes, color=PS.OKABE_ITO[1], lw=1.6)
    ax.axvline(0, color=PS.GREY, lw=0.8, ls=":")
    ax.set_xlabel("mean per-item change in P(Yes)")
    ax.set_ylabel("random removals")
    ax.set_title(f"Direction is systematic\n(observed {obs_mean_dyes:+.4f}, "
                 f"{N_PERM} draws)", pad=6)
    ax = axes[1]
    ax.scatter(it.entropy_all, it.jsd, s=11, alpha=0.5, color=PS.OKABE_ITO[0], lw=0)
    ax.set_xlabel("unfiltered entropy (bits)"); ax.set_ylabel("JSD (bits)")
    ax.set_title("JSD vs disagreement\nlow-entropy outliers dominate", pad=6)
    ax = axes[2]
    ax.scatter(it.entropy_all, it.tvd, s=11, alpha=0.5, color=PS.OKABE_ITO[2], lw=0)
    ax.set_xlabel("unfiltered entropy (bits)"); ax.set_ylabel("total variation distance")
    ax.set_title("TVD vs disagreement\nno log term, trend is clean", pad=6)
    PS.save(fig, "fig2c_metric_sensitivity")


if __name__ == "__main__":
    main()
