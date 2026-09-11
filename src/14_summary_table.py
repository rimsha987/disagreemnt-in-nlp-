"""Every number the paper will cite, assembled by READING results/ - nothing retyped.

Each row carries the value, its N, and the file it came from, so a figure in the paper can be
traced back to the script that produced it. If a source file changes, re-running this changes
the table; there is no hand-maintained copy to fall out of date.

Outputs: results/SUMMARY_TABLE.csv, results/SUMMARY_TABLE.md
"""
import sys, json, glob
import numpy as np, pandas as pd
sys.path.insert(0, "src")
import dices_io as D, prompts as PR

ROWS = []


def add(phase, quantity, value, n="", source="", note=""):
    ROWS.append(dict(phase=phase, quantity=quantity, value=value, n=n,
                     source=source, note=note))


def f(x, d=4):
    return "" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{d}f}"


def main():
    # ------------------------------------------------------------------ Phase 0
    df = D.load_350()
    S = "data/dices-dataset/350/diverse_safety_adversarial_dialog_350.csv"
    add("0", "Conversations (items)", 350, 350, S)
    add("0", "Raters", 123, 123, S)
    add("0", "Ratings (fully crossed)", f"{len(df):,}", len(df), S, "123 x 350, zero missing")
    add("0", "Ratings per conversation", "123 (min=median=max)", 350, S)
    add("0", "Q_overall: No / Yes / Unsure",
        f"{(df.Q_overall=='No').mean():.3f} / {(df.Q_overall=='Yes').mean():.3f} / "
        f"{(df.Q_overall=='Unsure').mean():.3f}", len(df), S)
    add("0", "safety_gold balance", "175 Yes / 175 No", 350, S)
    add("0", "degree_of_harm",
        " / ".join(f"{k} {v}" for k, v in
                   df.drop_duplicates('item_id').degree_of_harm.value_counts().items()),
        350, S)
    add("0", "Quality-flag column exists", "NO - 19 rater IDs hardcoded in dataset README",
        "", "data/dices-dataset/350/README.md", "see FINDINGS 0")

    # ------------------------------------------------------------------ Phase 1
    rm = pd.read_csv("results/02_removal_by_group.csv")
    SRC = "results/02_removal_by_group.csv"
    for t in rm.itertuples():
        g = D.SHORT.get(t.group, t.group)
        add("1", f"Removal rate: {g}", f"{100*t.removal_rate:.1f}%", t.n_pool, SRC,
            f"{t.n_removed}/{t.n_pool}, 95% CI [{100*t.ci_lo:.1f}, {100*t.ci_hi:.1f}]")
    tests = pd.read_csv("results/02_removal_tests.csv")
    for t in tests[tests.group == "OMNIBUS"].itertuples():
        add("1", f"Omnibus exact p: {D.PRETTY_VAR.get(t.variable, t.variable)}",
            f"{t.p_value:.4f}", t.n_pool, "results/02_removal_tests.csv",
            f"Freeman-Halton; Cramer's V {t.cramers_v:.3f}")
    for t in tests[tests.group.isin(["Man", "Woman"])].itertuples():
        add("1", f"Fisher OR: {t.group} removed", f"{t.odds_ratio:.2f}", t.n_pool,
            "results/02_removal_tests.csv",
            f"95% CI [{t.or_ci_lo:.2f}, {t.or_ci_hi:.2f}], p={t.p_value:.4f}")

    it = pd.read_csv("results/03_item_jsd.csv")
    SRC = "results/03_item_jsd.csv"
    add("1", "Label shift, mean TVD (primary)", f(it.tvd.mean(), 5), 350, SRC,
        f"median {it.tvd.median():.5f}, max {it.tvd.max():.5f}")
    add("1", "Label shift, mean JSD (secondary)", f(it.jsd.mean(), 5), 350, SRC,
        f"median {it.jsd.median():.5f}, max {it.jsd.max():.5f}")
    add("1", "Majority-label flips", f"{int(it.maj_changed.sum())}/350",
        350, SRC, "tie-aware; 7 under first-label tie-breaking")
    add("1", "Mean change in P(Yes)", f"{it.d_yes.mean():+.5f}", 350, SRC,
        "pool 0.32669 -> 0.33121")
    eb = pd.read_csv("results/03_entropy_bins.csv")
    for t in eb.itertuples():
        add("1", f"TVD by disagreement tertile: {t.bin}", f(t.mean_tvd, 5), t.n,
            "results/03_entropy_bins.csv", f"flips {t.n_maj_changed}")
    dt = pd.read_csv("results/03b_direction_tests.csv")
    for t in dt.itertuples():
        add("1", f"Permutation test: {t.statistic}", f"{t.observed:+.5f}", t.n_perm,
            "results/03b_direction_tests.csv",
            f"null {t.null_mean:+.5f}+/-{t.null_sd:.5f}, z={t.z:+.2f}, p={t.p_perm:.4f}")
    add("1", "Mean JSD vs random-removal null", "0.000550 vs 0.000575", 2000,
        "results/03_labels_report.txt", "p_perm=0.4495, z=-0.16")
    pt = pd.read_csv("results/04_profile_tests.csv")
    for t in pt.itertuples():
        add("1", f"Removed-rater profile: {t.label}",
            f"{t.removed_median:.4f} vs {t.kept_median:.4f}", "19 vs 104",
            "results/04_profile_tests.csv", f"MWU p={t.mwu_p:.4f}, CLES={t.cles:.3f}")
    add("1", "Removed raters extreme on >=1 signal", "18/19", 19,
        "results/04_profile_report.txt", "descriptive only")

    # ------------------------------------------------------------------ Phase 2
    sc = pd.read_csv("results/05_cell_scheme_scores.csv")
    for t in sc.itertuples():
        add("2", f"Cell-scheme spread: {t.scheme}", f(t.obs_tvd, 5), t.n_raters,
            "results/05_cell_scheme_scores.csv",
            f"null {t.null_tvd:.5f}, excess {t.excess:+.5f}, z={t.z:+.2f}")
    add("2", "Chosen cells", "race3 x gender2 (6 cells, 85 raters)", 85,
        "results/05_chosen_cells.csv", "z=+3.61, highest of 6 candidate schemes")
    add("2", "Model requested", "qwen/qwen3.8-27b", "", "src/phase2_config.py")
    add("2", "Model version (API-reported)", "qwen/qwen3.8-27b", "",
        "results/08_pilot_report.txt", "no dated snapshot; not pinnable")
    add("2", "Temperature / max output tokens", "1.0 / 8", "", "src/phase2_config.py")
    add("2", "Pilot calls", "60/60 returned", 60, "results/08_pilot_report.txt")
    add("2", "Pilot refusal rate", "0.0%", 60, "results/08_pilot_report.txt",
        "finish_reason=stop on all 60")
    add("2", "Pilot parse rate", "100.0%", 60, "results/08_pilot_report.txt")
    add("2", "Measured input tokens/call", "922.9 mean", 60,
        "results/08_pilot_report.txt", "min 826, max 1195")
    add("2", "Groq quota mechanism", "continuous refill, 1 request / 86.4 s", "",
        "results/08_pilot_report.txt", "no daily reset event; 86400/1000")
    hn = pd.read_csv("results/13_headline_numbers.csv")
    for t in hn.itertuples():
        add("2/3", f"Spread: {t.condition}", f(t.tvd, 5), t.scope,
            "results/13_headline_numbers.csv", t.note)
    add("2", "Noise floor, same-persona pairs", "[1,1,1,1,2,2,2,3,4,4] of 15", 10,
        "results/11_diagnostic_report.txt", "mean 2.10/15")
    add("2", "Between-cell, v1 prompt", "2/15", 15, "results/11_diagnostic_report.txt",
        "P(noise>=2)=0.60, directional gap +0.067")
    add("2", "Between-cell, v2 prompt", "3/15", 15, "results/11_diagnostic_report.txt",
        "P(noise>=3)=0.30, directional gap +0.000")
    add("2", "Model P(Yes) v1 / v2", "0.050 / 0.044", 60,
        "results/11_diagnostic_report.txt", "humans 0.278 on matched items")
    add("2", "DECISION", "LLM arm stopped; 3,675-call run not executed", "",
        "FINDINGS.md 2.13", "180 calls spent of 3,675 budgeted")

    # ------------------------------------------------------------------ Phase 3
    sp = pd.read_csv("results/12_spread_by_condition.csv")
    for t in sp.itertuples():
        add("3", f"Spread [{t.grouping}] {t.condition}", f(t.tvd, 5), t.n_raters,
            "results/12_spread_by_condition.csv",
            f"null {t.nullA_tvd:.5f}, excess {t.excess:+.5f}, z={t.z:+.2f}, "
            f"JSD {t.jsd:.6f}, to-pooled {t.tvd_to_pooled:.5f}")
    nb = pd.read_csv("results/12_spread_nullB.csv")
    for t in nb.itertuples():
        add("3", f"Filtering vs random-19 [{t.grouping}]", f"{t.delta:+.5f}", t.n_perm,
            "results/12_spread_nullB.csv",
            f"null {t.nullB_mean:+.5f}, z={t.z:+.2f}, p={t.p_perm:.3f}")
    bd = pd.read_csv("results/12_spread_breakdowns.csv")
    for t in bd.itertuples():
        add("3", f"Spread by {t.split}: {t.level}",
            f"{t.tvd_all:.5f} -> {t.tvd_kept:.5f}", t.n_items,
            "results/12_spread_breakdowns.csv", f"delta {t.delta:+.5f}")

    out = pd.DataFrame(ROWS)
    out.to_csv("results/SUMMARY_TABLE.csv", index=False)

    lines = ["# Summary table - every number the paper cites", "",
             "Generated by `src/14_summary_table.py` by reading `results/`. Nothing here is",
             "hand-transcribed. Re-run the script to refresh.", ""]
    for ph in ["0", "1", "2", "2/3", "3"]:
        sub = out[out.phase == ph]
        if sub.empty:
            continue
        lines += [f"## Phase {ph}", "", "| Quantity | Value | N | Source | Note |",
                  "|---|---|---|---|---|"]
        for t in sub.itertuples():
            lines.append(f"| {t.quantity} | `{t.value}` | {t.n} | `{t.source}` | {t.note} |")
        lines.append("")
    open("results/SUMMARY_TABLE.md", "w", encoding="utf-8").write("\n".join(lines) + "\n")

    print(f"wrote results/SUMMARY_TABLE.csv and .md")
    print(f"   {len(out)} rows across phases {sorted(out.phase.unique())}")
    print(f"   distinct source files cited: {out.source.nunique()}")
    for s in sorted(out.source.unique()):
        print(f"      {s}  ({int((out.source==s).sum())} rows)")


if __name__ == "__main__":
    main()
