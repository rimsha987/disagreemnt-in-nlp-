"""RQ1, part 1: who gets removed by the published DICES quality filter?

Verifies computed removal counts against the targets supplied at Checkpoint 2 (derived
from the DICES paper's kept-104 pool). Reports Fisher's exact tests with odds ratios and
confidence intervals, Clopper-Pearson intervals on every proportion, and Cramer's V.
Every percentage is printed with its N.

Outputs: results/02_removal_by_group.csv, results/02_removal_tests.csv,
         figures/fig1_removal_rate_by_group.{pdf,png}
"""
import sys, itertools, math
import numpy as np, pandas as pd
from scipy import stats
from scipy.stats.contingency import odds_ratio
sys.path.insert(0, "src")
import dices_io as D, plotstyle as PS
import matplotlib.pyplot as plt

# ---------------------------------------------------------------- targets (Checkpoint 2)
# Supplied by the human from the DICES paper's kept-104 pool. NOT adjustable.
TARGET_REMOVED = {
    "rater_gender": {"Woman": 5, "Man": 14},
    "rater_age": {"gen z": 7, "millenial": 8, "gen x+": 4},
    "rater_race": {"White": 5, "Black/African American": 6, "Asian/Asian subcontinent": 5,
                   "LatinX, Latino, Hispanic or Spanish Origin": 0, "Multiracial": 3},
}
TARGET_KEPT = {
    "rater_gender": {"Woman": 57, "Man": 47},
    "rater_age": {"gen z": 49, "millenial": 28, "gen x+": 27},
    "rater_race": {"White": 25, "Black/African American": 23, "Asian/Asian subcontinent": 21,
                   "LatinX, Latino, Hispanic or Spanish Origin": 22, "Multiracial": 13},
}


def clopper_pearson(k, n, alpha=0.05):
    """Exact binomial CI. Matches the exactness of Fisher's test."""
    lo = 0.0 if k == 0 else stats.beta.ppf(alpha / 2, k, n - k + 1)
    hi = 1.0 if k == n else stats.beta.ppf(1 - alpha / 2, k + 1, n - k)
    return float(lo), float(hi)


def _log_table_prob(rows, cols, cells, logfac):
    """log P of a contingency table under fixed margins (multiple hypergeometric)."""
    N = sum(rows)
    return (sum(logfac[r] for r in rows) + sum(logfac[c] for c in cols)
            - logfac[N] - sum(logfac[x] for x in cells))


def freeman_halton(table):
    """Exact Fisher test for a 2 x C table, by full enumeration over fixed margins.
    Returns (p_value, n_tables_enumerated). Two-sided in the standard sense:
    sums probability of every table no more likely than the observed one."""
    table = np.asarray(table, dtype=int)
    assert table.shape[0] == 2
    rows = table.sum(axis=1).tolist()
    cols = table.sum(axis=0).tolist()
    N = int(table.sum())
    logfac = np.concatenate(([0.0], np.cumsum(np.log(np.arange(1, N + 1)))))

    obs = _log_table_prob(rows, cols, table.ravel().tolist(), logfac)
    r0 = rows[0]
    total, count = 0.0, 0
    # enumerate the top row; the bottom row is then determined
    ranges = [range(0, min(c, r0) + 1) for c in cols]
    for top in itertools.product(*ranges):
        if sum(top) != r0:
            continue
        bottom = [cols[j] - top[j] for j in range(len(cols))]
        if any(b < 0 for b in bottom):
            continue
        lp = _log_table_prob(rows, cols, list(top) + bottom, logfac)
        count += 1
        if lp <= obs + 1e-9:
            total += math.exp(lp)
    return float(min(total, 1.0)), count


def cramers_v(table):
    table = np.asarray(table, dtype=float)
    chi2 = stats.chi2_contingency(table, correction=False)[0]
    n = table.sum()
    return float(np.sqrt(chi2 / (n * min(table.shape[0] - 1, table.shape[1] - 1))))


def main():
    df = D.load_350()
    r = D.rater_table(df)
    n_rm, n_kept = int(r.removed.sum()), int((~r.removed).sum())
    print("=" * 96)
    print("RQ1 PART 1 - WHO DOES THE PUBLISHED FILTER REMOVE?")
    print("=" * 96)
    print(f"rater pool: {len(r)}   removed: {n_rm}   kept: {n_kept}")
    print(f"rows removed: {int(df.removed.sum())} / {len(df)} "
          f"({100 * df.removed.mean():.1f}%)")
    per_item = df[df.removed].groupby("item_id").size()
    print(f"ratings removed per item: min={per_item.min()} max={per_item.max()} "
          f"(uniform across all {per_item.size} items: {per_item.nunique() == 1})")

    # ------------------------------------------------------- verification against targets
    print("\n" + "=" * 96)
    print("VERIFICATION AGAINST CHECKPOINT-2 TARGETS (computed vs required)")
    print("=" * 96)
    all_ok = True
    for var, target in TARGET_REMOVED.items():
        print(f"\n-- {D.PRETTY_VAR[var]} --")
        print(f"   {'group':<44} {'pool':>5} {'removed':>8} {'target':>7} {'kept':>5} "
              f"{'t.kept':>7}  status")
        for g in D.GROUP_ORDER[var]:
            pool = int((r[var] == g).sum())
            rem = int(((r[var] == g) & r.removed).sum())
            kept = pool - rem
            t_rem, t_kept = target[g], TARGET_KEPT[var][g]
            ok = (rem == t_rem) and (kept == t_kept)
            all_ok &= ok
            print(f"   {D.SHORT.get(g, g):<44} {pool:>5} {rem:>8} {t_rem:>7} {kept:>5} "
                  f"{t_kept:>7}  {'MATCH' if ok else '*** MISMATCH ***'}")
        s_rem = int(((r[var].isin(target)) & r.removed).sum())
        ok_sum = s_rem == 19
        all_ok &= ok_sum
        print(f"   {'column sum':<44} {len(r):>5} {s_rem:>8} {19:>7} "
              f"{'':>5} {'':>7}  {'MATCH' if ok_sum else '*** MISMATCH ***'}")
    print(f"\n   OVERALL VERIFICATION: {'PASS - all cells match' if all_ok else 'FAIL'}")
    if not all_ok:
        print("   STOPPING per Checkpoint-2 instruction: computed numbers differ from target.")
        sys.exit(1)

    # ------------------------------------------------------------------ per-group rates
    print("\n" + "=" * 96)
    print("REMOVAL RATE BY GROUP, with exact (Clopper-Pearson) 95% CI")
    print("=" * 96)
    rows_out = []
    for var in D.DEMOGRAPHICS:
        print(f"\n-- {D.PRETTY_VAR[var]} --")
        print(f"   {'group':<28} {'n_pool':>6} {'n_removed':>9} {'rate':>8}  {'95% CI':>18}")
        for g in D.GROUP_ORDER[var]:
            pool = int((r[var] == g).sum())
            rem = int(((r[var] == g) & r.removed).sum())
            lo, hi = clopper_pearson(rem, pool)
            print(f"   {D.SHORT.get(g, g):<28} {pool:>6} {rem:>9} {100*rem/pool:>7.1f}% "
                  f"  [{100*lo:>5.1f}, {100*hi:>5.1f}]")
            rows_out.append(dict(variable=var, group=g, n_pool=pool, n_removed=rem,
                                 n_kept=pool - rem, removal_rate=rem / pool,
                                 ci_lo=lo, ci_hi=hi))
    lo, hi = clopper_pearson(n_rm, len(r))
    print(f"\n   {'ALL RATERS':<28} {len(r):>6} {n_rm:>9} {100*n_rm/len(r):>7.1f}% "
          f"  [{100*lo:>5.1f}, {100*hi:>5.1f}]")
    rows_out.append(dict(variable="ALL", group="all raters", n_pool=len(r), n_removed=n_rm,
                         n_kept=n_kept, removal_rate=n_rm / len(r), ci_lo=lo, ci_hi=hi))
    pd.DataFrame(rows_out).to_csv("results/02_removal_by_group.csv", index=False)

    # ------------------------------------------------------------------------- testing
    print("\n" + "=" * 96)
    print("IS REMOVAL INDEPENDENT OF GROUP MEMBERSHIP?")
    print("Fisher's exact throughout. Omnibus = Freeman-Halton exact for 2xC.")
    print("=" * 96)
    tests = []
    for var in D.DEMOGRAPHICS:
        groups = D.GROUP_ORDER[var]
        tab = np.array([[int(((r[var] == g) & r.removed).sum()) for g in groups],
                        [int(((r[var] == g) & ~r.removed).sum()) for g in groups]])
        p_om, n_tab = freeman_halton(tab)
        v = cramers_v(tab)
        print(f"\n-- {D.PRETTY_VAR[var]} --")
        print(f"   table (row0=removed, row1=kept), columns {[D.SHORT.get(g,g) for g in groups]}:")
        print(f"     removed {tab[0].tolist()}")
        print(f"     kept    {tab[1].tolist()}")
        print(f"   Freeman-Halton exact p = {p_om:.4f}   ({n_tab} tables enumerated)")
        print(f"   Cramer's V = {v:.3f}   (n = {int(tab.sum())} raters)")
        tests.append(dict(variable=var, group="OMNIBUS", n_pool=int(tab.sum()),
                          n_removed=int(tab[0].sum()), test="Freeman-Halton exact",
                          p_value=p_om, odds_ratio=np.nan, or_ci_lo=np.nan,
                          or_ci_hi=np.nan, cramers_v=v))

        print(f"\n   per group vs all other raters (2x2 Fisher's exact, conditional MLE OR):")
        print(f"   {'group':<28} {'a/b (rm/kept)':>16} {'OR':>8} {'95% CI':>20} {'p':>9}")
        for g in groups:
            a = int(((r[var] == g) & r.removed).sum())
            b = int(((r[var] == g) & ~r.removed).sum())
            c = int(((r[var] != g) & r.removed).sum())
            d = int(((r[var] != g) & ~r.removed).sum())
            t2 = np.array([[a, b], [c, d]])
            p = stats.fisher_exact(t2, alternative="two-sided").pvalue
            res = odds_ratio(t2, kind="conditional")
            or_hat = res.statistic
            ci = res.confidence_interval(confidence_level=0.95)
            or_s = "0" if or_hat == 0 else ("inf" if not np.isfinite(or_hat) else f"{or_hat:.2f}")
            ci_s = f"[{ci.low:.2f}, {'inf' if not np.isfinite(ci.high) else f'{ci.high:.2f}'}]"
            print(f"   {D.SHORT.get(g, g):<28} {f'{a}/{b}':>16} {or_s:>8} {ci_s:>20} {p:>9.4f}")
            tests.append(dict(variable=var, group=g, n_pool=a + b, n_removed=a,
                              test="Fisher exact 2x2 (group vs rest)", p_value=p,
                              odds_ratio=or_hat, or_ci_lo=ci.low, or_ci_hi=ci.high,
                              cramers_v=np.nan))
    pd.DataFrame(tests).to_csv("results/02_removal_tests.csv", index=False)

    print("\n" + "=" * 96)
    print("POWER NOTE")
    print("=" * 96)
    print(f"   The whole test rests on {len(r)} raters and {n_rm} removals. The smallest")
    print(f"   race cell holds 16 raters; Latine/x has 0 removals, so its odds ratio is 0")
    print(f"   with an unbounded-below interval. Any group-level rate here carries a CI")
    print(f"   roughly +/-15 points wide. These tests are underpowered by construction and")
    print(f"   no null result should be read as evidence of no effect.")

    # --------------------------------------------------------------------------- figure
    print("\nFIGURE 1")
    PS.apply()
    plot_vars = D.DEMOGRAPHICS
    # panel width proportional to number of groups, so tick labels cannot collide
    widths = [len(D.GROUP_ORDER[v]) for v in plot_vars]
    fig, axes = plt.subplots(1, len(plot_vars), figsize=(11.0, 3.4),
                             gridspec_kw=dict(wspace=0.30, width_ratios=widths))
    overall = n_rm / len(r)
    for ax, var in zip(axes, plot_vars):
        groups = D.GROUP_ORDER[var]
        ys, los, his, ns = [], [], [], []
        for g in groups:
            pool = int((r[var] == g).sum())
            rem = int(((r[var] == g) & r.removed).sum())
            lo_, hi_ = clopper_pearson(rem, pool)
            ys.append(100 * rem / pool); los.append(100 * lo_); his.append(100 * hi_)
            ns.append(pool)
        x = np.arange(len(groups))
        ax.axhline(100 * overall, color=PS.LIGHT, lw=1.0, zorder=0)
        err = np.array([np.array(ys) - np.array(los), np.array(his) - np.array(ys)])
        ax.errorbar(x, ys, yerr=err, fmt="o", ms=4.5, capsize=3, lw=1.2,
                    color=PS.OKABE_ITO[0], ecolor=PS.GREY, mfc=PS.OKABE_ITO[0], mec="white")
        ax.set_xticks(x)
        ax.set_xticklabels([f"{D.SHORT.get(g, g)}\n(n={n})" for g, n in zip(groups, ns)],
                           rotation=45, ha="right", rotation_mode="anchor", fontsize=7.6)
        ax.set_xlim(-0.6, len(groups) - 0.4)
        ax.set_ylim(-2, 62)
        ax.set_title(D.PRETTY_VAR[var], pad=6)
        if ax is axes[0]:
            ax.set_ylabel("raters removed (%)")
        else:
            ax.tick_params(labelleft=False)
    axes[0].text(-0.45, 100 * overall + 1.5, f"pool mean {100*overall:.1f}%",
                 fontsize=7.0, color=PS.GREY, va="bottom")
    fig.suptitle("Removal rate by demographic group, DICES-350 published filter "
                 "(19 of 123 raters)", y=1.04, fontsize=10.5)
    fig.text(0.5, -0.42, "Points are removal rates; bars are exact (Clopper-Pearson) 95% "
             "intervals. Grey line is the pool-wide rate (15.4%).\nGroup sizes are small, "
             "so intervals are wide; Latine/x has zero removals.",
             ha="center", fontsize=7.4, color=PS.GREY)
    PS.save(fig, "fig1_removal_rate_by_group")
    print("\nwrote results/02_removal_by_group.csv, results/02_removal_tests.csv")


if __name__ == "__main__":
    main()
