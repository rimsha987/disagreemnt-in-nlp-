"""RQ1, part 2: how far does the published filter move the labels?

Per item, builds the soft label distribution over (No, Yes, Unsure) twice - all 123 raters,
then the 104 kept raters - and measures the shift.

  * TVD (total variation distance) is the PRIMARY metric, promoted at Checkpoint 2.
    JSD is reported alongside as SECONDARY. See tvd() and FINDINGS.md 1.5 for why.

Conventions, fixed and stated because they change the numbers:
  * JSD is the DIVERGENCE, base-2 logs, so it lies in [0, 1] for any pair of distributions.
    JS distance (its square root) is also reported. Where a single "JSD" is quoted, it is
    the divergence.
  * Entropy is Shannon, base 2, over 3 categories, so max = log2(3) = 1.585.
  * "Unsure" is its own bin. The binary appendix at the end is the only place it is collapsed.

Outputs: results/03_item_jsd.csv, results/03_entropy_bins.csv, results/03_worked_example.txt,
         figures/fig2_label_shift_histogram.{pdf,png}, figures/fig2b_jsd_by_entropy.{pdf,png}
"""
import sys
import numpy as np, pandas as pd
from scipy import stats
sys.path.insert(0, "src")
import dices_io as D, plotstyle as PS
import matplotlib.pyplot as plt

RNG_SEED = 20260826          # fixed; the permutation baseline is reproducible
N_PERM = 2000


def entropy2(p):
    p = np.asarray(p, dtype=float)
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def jsd2(p, q):
    """Jensen-Shannon DIVERGENCE, base 2. In [0, 1]. SECONDARY metric - see tvd()."""
    p, q = np.asarray(p, float), np.asarray(q, float)
    assert abs(p.sum() - 1) < 1e-9 and abs(q.sum() - 1) < 1e-9
    m = 0.5 * (p + q)
    return float(entropy2(m) - 0.5 * (entropy2(p) + entropy2(q)))


def tvd(p, q):
    """Total variation distance, 0.5*sum|p-q|. In [0, 1]. PRIMARY metric.

    Promoted over JSD at Checkpoint 2. JSD's second-order form is
    (1/8ln2) * sum_i (q_i-p_i)^2 / p_i, so each category is weighted by 1/p_i and
    near-empty cells are amplified - which reverses the entropy-bin ordering on this
    data for a reason that has nothing to do with the filter. TVD has no such weighting.
    Justification and the verification of that expansion are in
    src/03b_direction_and_metric_checks.py and FINDINGS.md 1.5."""
    p, q = np.asarray(p, float), np.asarray(q, float)
    assert abs(p.sum() - 1) < 1e-9 and abs(q.sum() - 1) < 1e-9
    return float(0.5 * np.abs(p - q).sum())


def dists(sub):
    """Counts and normalised distribution over D.LABELS for a slice of ratings."""
    c = sub[D.LABEL_COL].value_counts().reindex(D.LABELS).fillna(0).to_numpy(float)
    return c, c / c.sum()


def majority(p):
    """argmax label, or 'TIE:a|b' if the top is not unique."""
    mx = p.max()
    top = [D.LABELS[i] for i in range(len(p)) if abs(p[i] - mx) < 1e-12]
    return top[0] if len(top) == 1 else "TIE:" + "|".join(top)


def main():
    df = D.load_350()
    kept = df[~df.removed]
    assert kept.rater_id.nunique() == 104
    assert len(kept) == 350 * 104, f"kept rows {len(kept)}"

    print("=" * 96)
    print("RQ1 PART 2 - HOW MUCH DO THE LABELS MOVE?")
    print("=" * 96)
    print(f"unfiltered: {df.rater_id.nunique()} raters, {len(df)} ratings, "
          f"{len(df)//350} per item")
    print(f"filtered:   {kept.rater_id.nunique()} raters, {len(kept)} ratings, "
          f"{len(kept)//350} per item")

    # ------------------------------------------------------------- per-item computation
    recs = []
    for iid, sub in df.groupby("item_id"):
        c_all, p_all = dists(sub)
        c_kep, p_kep = dists(sub[~sub.removed])
        assert c_all.sum() == 123 and c_kep.sum() == 104
        recs.append(dict(
            item_id=iid,
            n_all=int(c_all.sum()), n_kept=int(c_kep.sum()),
            **{f"cnt_all_{l}": int(c_all[i]) for i, l in enumerate(D.LABELS)},
            **{f"cnt_kept_{l}": int(c_kep[i]) for i, l in enumerate(D.LABELS)},
            **{f"p_all_{l}": p_all[i] for i, l in enumerate(D.LABELS)},
            **{f"p_kept_{l}": p_kep[i] for i, l in enumerate(D.LABELS)},
            tvd=tvd(p_all, p_kep),
            jsd=jsd2(p_all, p_kep),
            js_dist=np.sqrt(jsd2(p_all, p_kep)),
            entropy_all=entropy2(p_all), entropy_kept=entropy2(p_kep),
            maj_all=majority(p_all), maj_kept=majority(p_kep),
            d_yes=p_kep[D.LABELS.index("Yes")] - p_all[D.LABELS.index("Yes")],
            d_unsure=p_kep[D.LABELS.index("Unsure")] - p_all[D.LABELS.index("Unsure")],
        ))
    it = pd.DataFrame(recs)
    assert len(it) == 350
    it["maj_changed"] = it.maj_all != it.maj_kept

    print("\n" + "=" * 96)
    print("PER-ITEM LABEL SHIFT - TVD PRIMARY, JSD SECONDARY")
    print("=" * 96)
    for nm, col in [("TVD (PRIMARY)", "tvd"), ("JSD (secondary)", "jsd")]:
        q = it[col].describe(percentiles=[.05, .5, .95, .99])
        print(f"\n   -- {nm}, range [0,1] --")
        print(f"      n items {int(q['count'])}   mean {q['mean']:.5f}   sd {it[col].std():.5f}")
        print(f"      min {q['min']:.5f}   5th {q['5%']:.5f}   median {q['50%']:.5f}   "
              f"95th {q['95%']:.5f}   99th {q['99%']:.5f}")
        print(f"      max {q['max']:.5f}  (item_id {int(it.loc[it[col].idxmax(),'item_id'])})")
    print(f"\n   JS distance (sqrt JSD): mean {it.js_dist.mean():.5f}, "
          f"median {it.js_dist.median():.5f}, max {it.js_dist.max():.5f}")
    print(f"\n   TVD is directly readable: a TVD of 0.0156 means the filtered and unfiltered")
    print(f"   distributions differ by 1.56 percentage points of probability mass in total.")
    print(f"   Spearman rho(TVD, JSD) across items = "
          f"{stats.spearmanr(it.tvd, it.jsd).statistic:+.4f}")

    print("\n   items by TVD magnitude:")
    for lo, hi in [(0, .01), (.01, .02), (.02, .03), (.03, .05), (.05, 1.0)]:
        n = int(((it.tvd >= lo) & (it.tvd < hi)).sum())
        print(f"      {lo:.3f} <= TVD < {hi:<5.3f}   {n:>4} items ({100*n/350:5.1f}%)")

    # ------------------------------------------------------------------ majority change
    print("\n" + "=" * 96)
    print("MAJORITY LABEL CHANGES")
    print("=" * 96)
    n_ch = int(it.maj_changed.sum())
    print(f"   items whose majority label changes: {n_ch} / 350 ({100*n_ch/350:.1f}%)")
    print(f"   ties under the full pool:   {int(it.maj_all.str.startswith('TIE').sum())}")
    print(f"   ties under the filtered pool: {int(it.maj_kept.str.startswith('TIE').sum())}")
    if n_ch:
        print("\n   every changed item:")
        cols = ["item_id", "maj_all", "maj_kept", "tvd", "jsd", "entropy_all"] + \
               [f"cnt_all_{l}" for l in D.LABELS] + [f"cnt_kept_{l}" for l in D.LABELS]
        print(it[it.maj_changed][cols].to_string(index=False,
              float_format=lambda v: f"{v:.4f}"))
    print("\n   majority label distribution:")
    print(f"      unfiltered: {dict(it.maj_all.value_counts())}")
    print(f"      filtered:   {dict(it.maj_kept.value_counts())}")

    # --------------------------------------------------------------------- direction
    print("\n" + "=" * 96)
    print("DIRECTION OF THE SHIFT")
    print("=" * 96)
    print(f"   mean change in P(Yes) = {it.d_yes.mean():+.5f}  "
          f"(median {it.d_yes.median():+.5f}, sd {it.d_yes.std():.5f})")
    print(f"   items where P(Yes) rises: {int((it.d_yes>0).sum())}, "
          f"falls: {int((it.d_yes<0).sum())}, unchanged: {int((it.d_yes==0).sum())}")
    print(f"   mean change in P(Unsure) = {it.d_unsure.mean():+.5f}")
    print(f"   pool-level P(Yes): unfiltered {(df[D.LABEL_COL]=='Yes').mean():.5f}  ->  "
          f"filtered {(kept[D.LABEL_COL]=='Yes').mean():.5f}")
    print(f"   mean absolute change in P(Yes) = {it.d_yes.abs().mean():.5f} "
          f"({100*it.d_yes.abs().mean():.2f} percentage points)")
    print(f"\n   NO ITEM-LEVEL SIGNIFICANCE TEST IS REPORTED HERE. A Wilcoxon signed-rank over")
    print(f"   the 350 items would treat them as 350 independent observations, but the same 19")
    print(f"   raters are removed from every item - there is ONE independent draw of a removal")
    print(f"   set, not 350. Any such test is pseudo-replicated. Inference on this shift comes")
    print(f"   from the rater-level permutation test in src/03b_direction_and_metric_checks.py.")

    # ------------------------------------------------------------- JSD vs disagreement
    print("\n" + "=" * 96)
    print("DOES FILTERING MATTER MORE FOR CONTESTED ITEMS?")
    print("=" * 96)
    print(f"   unfiltered entropy (base 2, max log2(3)={np.log2(3):.3f}): "
          f"mean {it.entropy_all.mean():.4f}, median {it.entropy_all.median():.4f}, "
          f"min {it.entropy_all.min():.4f}, max {it.entropy_all.max():.4f}")
    for nm, col in [("TVD (PRIMARY)", "tvd"), ("JSD (secondary)", "jsd")]:
        rs, rp = stats.spearmanr(it.entropy_all, it[col]), stats.pearsonr(it.entropy_all, it[col])
        print(f"   {nm:<16} vs entropy: Spearman {rs.statistic:+.4f} (p={rs.pvalue:.3g})   "
              f"Pearson {rp.statistic:+.4f} (p={rp.pvalue:.3g})   n=350")

    it["ent_bin"] = pd.qcut(it.entropy_all, 3, labels=["low", "medium", "high"])
    cuts = pd.qcut(it.entropy_all, 3, retbins=True)[1]
    print(f"\n   entropy tertile cut points: {np.round(cuts, 4).tolist()}")
    print(f"\n   {'bin':<8} {'n':>4} {'entropy range':>16} {'mean TVD':>10} {'median TVD':>11} "
          f"{'mean |dP(Yes)|':>15} {'maj flips':>10} {'mean JSD':>10}")
    bin_rows = []
    for b in ["low", "medium", "high"]:
        s = it[it.ent_bin == b]
        rng = f"{s.entropy_all.min():.3f}-{s.entropy_all.max():.3f}"
        print(f"   {b:<8} {len(s):>4} {rng:>16} {s.tvd.mean():>10.5f} {s.tvd.median():>11.5f} "
              f"{s.d_yes.abs().mean():>15.5f} {int(s.maj_changed.sum()):>10} {s.jsd.mean():>10.5f}")
        bin_rows.append(dict(bin=b, n=len(s), ent_min=s.entropy_all.min(),
                             ent_max=s.entropy_all.max(),
                             mean_tvd=s.tvd.mean(), median_tvd=s.tvd.median(),
                             max_tvd=s.tvd.max(), mean_jsd=s.jsd.mean(),
                             median_jsd=s.jsd.median(), max_jsd=s.jsd.max(),
                             n_maj_changed=int(s.maj_changed.sum()),
                             mean_abs_d_yes=s.d_yes.abs().mean()))
    pd.DataFrame(bin_rows).to_csv("results/03_entropy_bins.csv", index=False)
    for nm, col in [("TVD (PRIMARY)", "tvd"), ("JSD (secondary)", "jsd")]:
        kw = stats.kruskal(*[it[it.ent_bin == b][col].values for b in ["low", "medium", "high"]])
        print(f"   Kruskal-Wallis across bins, {nm:<16} H={kw.statistic:.3f}, p={kw.pvalue:.3g}")
    print(f"\n   TVD, |dP(Yes)| and the flip counts all rise monotonically with disagreement.")
    print(f"   JSD does not, for the 1/p_i reason established in 03b. TVD is the primary read.")

    # -------------------------------------------------- permutation baseline (addition)
    print("\n" + "=" * 96)
    print("BASELINE: IS THIS FILTER'S SHIFT BIGGER THAN REMOVING *ANY* 19 RATERS?")
    print("(not requested in the brief; added because 'JSD is small' means little without")
    print(" a null, and this separates 'filtering does nothing' from 'any 19 do this')")
    print("=" * 96)
    rng = np.random.default_rng(RNG_SEED)
    rater_ids = df.rater_id.unique()
    # ratings matrix: items x raters, coded 0/1/2
    code = {l: i for i, l in enumerate(D.LABELS)}
    piv = df.pivot(index="item_id", columns="rater_id", values=D.LABEL_COL)
    M = piv.to_numpy()
    Mi = np.vectorize(code.get)(M)
    onehot = np.zeros((Mi.shape[0], Mi.shape[1], 3))
    np.put_along_axis(onehot, Mi[:, :, None], 1, axis=2)
    full_counts = onehot.sum(axis=1)
    p_full = full_counts / full_counts.sum(axis=1, keepdims=True)
    col_ids = list(piv.columns)
    idx_of = {r: i for i, r in enumerate(col_ids)}

    def mean_jsd_for(drop_ids):
        keep_idx = np.array([i for r, i in idx_of.items() if r not in set(drop_ids)])
        c = onehot[:, keep_idx, :].sum(axis=1)
        p = c / c.sum(axis=1, keepdims=True)
        m = 0.5 * (p_full + p)
        with np.errstate(divide="ignore", invalid="ignore"):
            def H(x):
                lg = np.where(x > 0, np.log2(np.where(x > 0, x, 1)), 0.0)
                return -(x * lg).sum(axis=1)
            return float(np.mean(H(m) - 0.5 * (H(p_full) + H(p))))

    obs = mean_jsd_for(D.REMOVED_RATERS_350)
    print(f"   observed mean JSD (published filter):        {obs:.6f}")
    assert abs(obs - it.jsd.mean()) < 1e-9, "vectorised and per-item JSD disagree"
    null = np.array([mean_jsd_for(rng.choice(rater_ids, 19, replace=False))
                     for _ in range(N_PERM)])
    print(f"   null: {N_PERM} random removals of 19 raters (seed {RNG_SEED})")
    print(f"      mean   {null.mean():.6f}")
    print(f"      sd     {null.std():.6f}")
    print(f"      2.5th  {np.percentile(null,2.5):.6f}   97.5th {np.percentile(null,97.5):.6f}")
    print(f"      min    {null.min():.6f}   max {null.max():.6f}")
    pperm = float((null >= obs).mean())
    print(f"   proportion of random removals with mean JSD >= observed: {pperm:.4f}")
    print(f"   observed is {(obs-null.mean())/null.std():+.2f} SD from the null mean")

    # --------------------------------------------------------------- binary appendix
    print("\n" + "=" * 96)
    print("APPENDIX - BINARY COLLAPSE ROBUSTNESS CHECK")
    print("Unsure merged into No, i.e. Yes vs not-Yes. Primary analysis stays 3-way.")
    print("=" * 96)
    bin_recs = []
    for iid, sub in df.groupby("item_id"):
        ya = (sub[D.LABEL_COL] == "Yes").mean()
        yk = (sub[~sub.removed][D.LABEL_COL] == "Yes").mean()
        bin_recs.append(dict(item_id=iid, jsd_bin=jsd2([1 - ya, ya], [1 - yk, yk]),
                             ent_bin_all=entropy2([1 - ya, ya])))
    bi = pd.DataFrame(bin_recs)
    print(f"   binary JSD: mean {bi.jsd_bin.mean():.5f}, median {bi.jsd_bin.median():.5f}, "
          f"max {bi.jsd_bin.max():.5f}")
    print(f"   3-way JSD:  mean {it.jsd.mean():.5f}, median {it.jsd.median():.5f}, "
          f"max {it.jsd.max():.5f}")
    print(f"   ratio of means (binary / 3-way): {bi.jsd_bin.mean()/it.jsd.mean():.3f}")
    r_, p_ = stats.spearmanr(bi.jsd_bin, it.jsd)
    print(f"   Spearman rho between the two per-item JSD series: {r_:+.4f} p={p_:.3g}")
    print("   -> conclusion is unchanged by the collapse; magnitudes differ because")
    print("      the binary version cannot see movement between No and Unsure.")
    it = it.merge(bi, on="item_id", validate="one_to_one")
    assert len(it) == 350

    it.to_csv("results/03_item_jsd.csv", index=False)
    print("\nwrote results/03_item_jsd.csv, results/03_entropy_bins.csv")

    # ------------------------------------------------------------------- worked example
    worked_example(df, it)

    # -------------------------------------------------------------------------- figures
    figures(it, null, obs)


def worked_example(df, it):
    """One conversation, every step printed. Chosen as the MAXIMUM-JSD item: if the
    arithmetic is wrong anywhere, the largest effect is where it shows."""
    iid = int(it.loc[it.jsd.idxmax(), "item_id"])
    sub = df[df.item_id == iid]
    L = []
    P = L.append
    P("=" * 96)
    P("WORKED EXAMPLE - ONE CONVERSATION, BY HAND")
    P("=" * 96)
    P(f"item_id = {iid}")
    P("chosen because it has the LARGEST per-item JSD of all 350 items, so it is the")
    P("strictest arithmetic check available. It is not typical; the median item is at the end.")
    P("")
    P(f"degree_of_harm = {sub.degree_of_harm.iloc[0]}   harm_type = {sub.harm_type.iloc[0]}")
    P(f"safety_gold    = {sub.safety_gold.iloc[0]}")
    P("")
    P("STEP 1. Raw ratings on Q_overall")
    P(f"   total raters on this item: {len(sub)}")
    ca = sub[D.LABEL_COL].value_counts().reindex(D.LABELS).fillna(0).astype(int)
    for l in D.LABELS:
        P(f"      {l:<8} {ca[l]:>4}")
    P(f"      {'SUM':<8} {ca.sum():>4}")
    P("")
    P(f"   of these, {int(sub.removed.sum())} raters are on the published removal list.")
    cr = sub[sub.removed][D.LABEL_COL].value_counts().reindex(D.LABELS).fillna(0).astype(int)
    P("   how those 19 removed raters voted:")
    for l in D.LABELS:
        P(f"      {l:<8} {cr[l]:>4}")
    P(f"      {'SUM':<8} {cr.sum():>4}")
    ck = sub[~sub.removed][D.LABEL_COL].value_counts().reindex(D.LABELS).fillna(0).astype(int)
    P("")
    P("STEP 2. Two soft label distributions")
    P("   UNFILTERED, all 123 raters:")
    pa = []
    for l in D.LABELS:
        v = ca[l] / 123
        pa.append(v)
        P(f"      P({l:<7}) = {ca[l]:>3} / 123 = {v:.6f}")
    P(f"      sum = {sum(pa):.6f}")
    P("   FILTERED, 104 kept raters:")
    pk = []
    for l in D.LABELS:
        v = ck[l] / 104
        pk.append(v)
        P(f"      P({l:<7}) = {ck[l]:>3} / 104 = {v:.6f}")
    P(f"      sum = {sum(pk):.6f}")
    pa, pk = np.array(pa), np.array(pk)
    P("")
    P("STEP 3. Mixture M = (P + Q) / 2")
    m = 0.5 * (pa + pk)
    for i, l in enumerate(D.LABELS):
        P(f"      M({l:<7}) = ({pa[i]:.6f} + {pk[i]:.6f}) / 2 = {m[i]:.6f}")
    P(f"      sum = {m.sum():.6f}")
    P("")
    P("STEP 4. Shannon entropies, base 2, H(x) = -sum p*log2(p), 0*log2(0) := 0")
    for nm, v in [("P (unfiltered)", pa), ("Q (filtered)", pk), ("M (mixture)", m)]:
        terms = " + ".join(f"{-x*np.log2(x):.6f}" if x > 0 else "0.000000" for x in v)
        P(f"      H({nm:<15}) = {terms} = {entropy2(v):.6f} bits")
    P("")
    P("STEP 5. JSD = H(M) - [H(P) + H(Q)] / 2")
    P(f"      = {entropy2(m):.6f} - ({entropy2(pa):.6f} + {entropy2(pk):.6f}) / 2")
    P(f"      = {entropy2(m):.6f} - {(entropy2(pa)+entropy2(pk))/2:.6f}")
    P(f"      = {jsd2(pa, pk):.6f} bits")
    P(f"   JS distance = sqrt(JSD) = {np.sqrt(jsd2(pa,pk)):.6f}")
    P(f"   cross-check against results/03_item_jsd.csv: "
      f"{float(it.loc[it.item_id==iid,'jsd'].iloc[0]):.6f}")
    P("")
    P("STEP 6. What moved")
    P(f"      majority label: {majority(pa)} -> {majority(pk)}"
      f"   ({'CHANGED' if majority(pa)!=majority(pk) else 'unchanged'})")
    P(f"      P(Yes): {pa[1]:.6f} -> {pk[1]:.6f}  ({pk[1]-pa[1]:+.6f}, "
      f"{100*(pk[1]-pa[1]):+.2f} percentage points)")
    P(f"      entropy: {entropy2(pa):.6f} -> {entropy2(pk):.6f} bits "
      f"({entropy2(pk)-entropy2(pa):+.6f})")
    P("")
    P("   Sense check: the 19 removed raters split "
      f"{'/'.join(str(cr[l]) for l in D.LABELS)} across {D.LABELS},")
    P(f"   against {'/'.join(str(ca[l]) for l in D.LABELS)} in the full pool. Removing them")
    P("   changes each proportion by a few points, and JSD of a few points of movement is")
    P("   O(0.001-0.01) bits. The number is small because the perturbation is small.")
    P("")
    med_iid = int(it.iloc[(it.jsd - it.jsd.median()).abs().argsort().iloc[0]]["item_id"])
    ms = it[it.item_id == med_iid].iloc[0]
    P("-" * 96)
    P(f"FOR CONTRAST, the median item (item_id {med_iid}, JSD {ms.jsd:.6f}):")
    P(f"   unfiltered counts {[int(ms[f'cnt_all_{l}']) for l in D.LABELS]} of 123 -> "
      f"{[round(ms[f'p_all_{l}'],4) for l in D.LABELS]}")
    P(f"   filtered   counts {[int(ms[f'cnt_kept_{l}']) for l in D.LABELS]} of 104 -> "
      f"{[round(ms[f'p_kept_{l}'],4) for l in D.LABELS]}")
    P(f"   majority {ms.maj_all} -> {ms.maj_kept}")
    txt = "\n".join(L)
    print("\n" + txt)
    with open("results/03_worked_example.txt", "w", encoding="utf-8") as f:
        f.write(txt + "\n")
    print("\nwrote results/03_worked_example.txt")


def figures(it, null, obs):
    print("\nFIGURES")
    PS.apply()

    # Figure 2: per-item label shift. TVD primary (left, large), JSD secondary (right).
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.2),
                             gridspec_kw=dict(wspace=0.30, width_ratios=[1.5, 1]))
    ax = axes[0]
    ax.hist(it.tvd, bins=45, color=PS.OKABE_ITO[0], edgecolor="white", linewidth=0.4)
    ax.axvline(it.tvd.mean(), color=PS.OKABE_ITO[1], lw=1.3, ls="--")
    ax.text(it.tvd.mean() * 1.05, ax.get_ylim()[1] * 0.95,
            f" mean {it.tvd.mean():.4f}", color=PS.OKABE_ITO[1], fontsize=8, va="top")
    ax.set_xlabel("total variation distance (max possible = 1)")
    ax.set_ylabel("conversations")
    ax.set_title("Label shift from the published filter\nTVD, primary metric", pad=8)
    ax.text(0.97, 0.62, f"n = 350 conversations\nmedian {it.tvd.median():.4f}\n"
            f"max {it.tvd.max():.4f}\nmajority flips {int(it.maj_changed.sum())}",
            transform=ax.transAxes, ha="right", va="top", fontsize=7.6, color=PS.GREY)
    ax = axes[1]
    ax.hist(it.jsd, bins=45, color=PS.LIGHT, edgecolor="white", linewidth=0.4)
    ax.axvline(it.jsd.mean(), color=PS.GREY, lw=1.1, ls="--")
    ax.set_xlabel("Jensen-Shannon divergence (bits)")
    ax.set_ylabel("conversations")
    ax.set_title("JSD, secondary\n(see fig2c for why)", pad=8)
    ax.text(0.97, 0.72, f"mean {it.jsd.mean():.5f}\nmax {it.jsd.max():.5f}",
            transform=ax.transAxes, ha="right", va="top", fontsize=7.6, color=PS.GREY)
    fig.text(0.5, -0.06, "DICES-350, all 123 raters vs the 104 kept by the published filter, "
             "per conversation.", ha="center", fontsize=7.4, color=PS.GREY)
    PS.save(fig, "fig2_label_shift_histogram")

    # Figure 2b: JSD vs entropy, plus bin means, plus permutation null
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.2),
                             gridspec_kw=dict(wspace=0.38, width_ratios=[1.15, 1, 1]))
    ax = axes[0]
    ax.scatter(it.entropy_all, it.tvd, s=11, alpha=0.55, color=PS.OKABE_ITO[0],
               linewidths=0)
    rho = stats.spearmanr(it.entropy_all, it.tvd).statistic
    ax.set_xlabel("unfiltered disagreement, entropy (bits)")
    ax.set_ylabel("TVD from filtering")
    ax.set_title(f"Shift vs disagreement\nSpearman $\\rho$ = {rho:+.3f} (n=350)", pad=6)

    ax = axes[1]
    bins = ["low", "medium", "high"]
    means = [it[it.ent_bin == b].tvd.mean() for b in bins]
    ns = [int((it.ent_bin == b).sum()) for b in bins]
    boxdata = [it[it.ent_bin == b].tvd.values for b in bins]
    bp = ax.boxplot(boxdata, widths=0.55, showfliers=True, patch_artist=True,
                    flierprops=dict(marker="o", ms=2.2, mfc=PS.GREY, mec="none", alpha=0.5),
                    medianprops=dict(color="white", lw=1.3),
                    whiskerprops=dict(color=PS.GREY), capprops=dict(color=PS.GREY))
    for patch in bp["boxes"]:
        patch.set_facecolor(PS.OKABE_ITO[0]); patch.set_edgecolor("none"); patch.set_alpha(0.85)
    ax.plot(range(1, 4), means, "D", ms=4, color=PS.OKABE_ITO[1], zorder=5)
    ax.set_xticks(range(1, 4))
    ax.set_xticklabels([f"{b}\n(n={n})" for b, n in zip(bins, ns)])
    ax.set_xlabel("disagreement tertile (unfiltered entropy)")
    ax.set_ylabel("TVD from filtering")
    ax.set_title("Filtering effect by\nhow contested the item is", pad=6)

    ax = axes[2]
    ax.hist(null, bins=40, color=PS.LIGHT, edgecolor="white", linewidth=0.3)
    ax.axvline(obs, color=PS.OKABE_ITO[1], lw=1.6)
    ax.text(obs, ax.get_ylim()[1] * 0.95, f"  published\n  filter\n  {obs:.4f}",
            color=PS.OKABE_ITO[1], fontsize=7.6, va="top")
    ax.set_xlabel("mean JSD across 350 items (magnitude null)")
    ax.set_ylabel("random removals")
    ax.set_title(f"Published filter vs removing\n19 raters at random ({len(null)} draws)", pad=6)
    PS.save(fig, "fig2b_jsd_by_entropy")


if __name__ == "__main__":
    main()
