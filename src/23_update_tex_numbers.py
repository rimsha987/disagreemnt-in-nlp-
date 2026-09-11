"""Propagate the converged 5,000-permutation results into the LaTeX.

Null A was raised 500 -> 5,000 and all six schemes now come from ONE run, so Tables 6 and 9
agree and the "two independent permutation runs" footnote can be deleted. Null B was raised
1,000 -> 5,000 because at 1,000 the gender p-value moved 0.026 -> 0.035 on RNG ordering alone.

Every substitution is verified present before it is applied; a miss is reported, never silent.
"""
import sys, re
import pandas as pd

P = "tex/main.tex"

SUBS = [
    # ---------------------------------------------------------------- Table 6
    (r"race3 $\times$ gender2 & 6 & 10 & 0.20315 & 0.16559 & $+0.03756$ & $+3.61$ \\",
     r"race3 $\times$ gender2 & 6 & 10 & 0.20315 & 0.16549 & $+0.03766$ & $+3.78$ \\"),
    (r"gender2 $\times$ age3 & 6 & 14 & 0.16156 & 0.13959 & $+0.02197$ & $+2.78$ \\",
     r"gender2 $\times$ age3 & 6 & 14 & 0.16156 & 0.13989 & $+0.02167$ & $+2.67$ \\"),
    (r"gender2 & 2 & 61 & 0.09170 & 0.07794 & $+0.01377$ & $+1.47$ \\",
     r"gender2 & 2 & 61 & 0.09170 & 0.07869 & $+0.01301$ & $+1.26$ \\"),
    (r"race3 & 3 & 26 & 0.12887 & 0.11503 & $+0.01384$ & $+1.23$ \\",
     r"race3 & 3 & 26 & 0.12887 & 0.11538 & $+0.01349$ & $+1.21$ \\"),
    (r"race5 & 5 & 16 & 0.13835 & 0.12749 & $+0.01086$ & $+1.21$ \\",
     r"race5 & 5 & 16 & 0.13835 & 0.12676 & $+0.01159$ & $+1.40$ \\"),
    (r"age3 & 3 & 31 & 0.09603 & 0.09949 & $-0.00346$ & $-0.37$ \\",
     r"age3 & 3 & 31 & 0.09603 & 0.09913 & $-0.00310$ & $-0.34$ \\"),
    # ---------------------------------------------------------------- Table 9
    (r"race3 $\times$ gender2 & All 123 & 0.20315 & 0.16479 & $+0.03836$ & $+4.00$ \\",
     r"race3 $\times$ gender2 & All 123 & 0.20315 & 0.16549 & $+0.03766$ & $+3.78$ \\"),
    (r"race3 $\times$ gender2 & 104 kept & 0.20900 & 0.18375 & $+0.02526$ & $+2.89$ \\",
     r"race3 $\times$ gender2 & 104 kept & 0.20900 & 0.18352 & $+0.02548$ & $+2.75$ \\"),
    (r"gender2 & All 123 & 0.09170 & 0.07823 & $+0.01347$ & $+1.40$ \\",
     r"gender2 & All 123 & 0.09170 & 0.07869 & $+0.01301$ & $+1.26$ \\"),
    (r"gender2 & 104 kept & 0.08186 & 0.08574 & $-0.00388$ & $-0.42$ \\",
     r"gender2 & 104 kept & 0.08186 & 0.08508 & $-0.00322$ & $-0.36$ \\"),
    (r"race5 & All 123 & 0.13835 & 0.12683 & $+0.01152$ & $+1.38$ \\",
     r"race5 & All 123 & 0.13835 & 0.12676 & $+0.01159$ & $+1.40$ \\"),
    (r"race5 & 104 kept & 0.15419 & 0.13660 & $+0.01758$ & $+2.25$ \\",
     r"race5 & 104 kept & 0.15419 & 0.13669 & $+0.01750$ & $+2.30$ \\"),
    (r"age3 & All 123 & 0.09603 & 0.09863 & $-0.00260$ & $-0.31$ \\",
     r"age3 & All 123 & 0.09603 & 0.09913 & $-0.00310$ & $-0.34$ \\"),
    (r"age3 & 104 kept & 0.10715 & 0.10810 & $-0.00096$ & $-0.10$ \\",
     r"age3 & 104 kept & 0.10715 & 0.10727 & $-0.00012$ & $-0.01$ \\"),
    # ---------------------------------------------------------------- Table 10
    (r"race3 $\times$ gender2 & $+0.00585$ & $+0.01103$ & $-0.41$ & 0.625 \\",
     r"race3 $\times$ gender2 & $+0.00585$ & $+0.01092$ & $-0.42$ & 0.642 \\"),
    (r"gender2 & $-0.00984$ & $+0.00591$ & $-2.16$ & 0.026 \\",
     r"gender2 & $-0.00984$ & $+0.00606$ & $-2.10$ & 0.037 \\"),
    (r"race5 & $+0.01584$ & $+0.01037$ & $+0.87$ & 0.376 \\",
     r"race5 & $+0.01584$ & $+0.01031$ & $+0.86$ & 0.380 \\"),
    (r"age3 & $+0.01112$ & $+0.00911$ & $+0.54$ & 0.573 \\",
     r"age3 & $+0.01112$ & $+0.00895$ & $+0.57$ & 0.553 \\"),
    # ---------------------------------------------------------------- prose: 4.4, 4.5
    ("The analysis samples random sets of 19 raters,\nwith between 2,000 and 10,000 draws depending on the statistic, and compares the published filter\nwith that empirical reference distribution.",
     "The analysis samples 5,000 random sets of 19 raters and compares the published filter with that\nempirical reference distribution. Null A uses 5,000 label permutations, and every null in the\npaper is computed from a single run so that repeated quantities agree exactly."),
    ("excess over its size-matched null, with observed spread 0.20315, null 0.16559, excess $+0.03756$,\nand $z = +3.61$. Age3, by contrast, fell below its null ($z = -0.37$)",
     "excess over its size-matched null, with observed spread 0.20315, null 0.16549, excess $+0.03766$,\nand $z = +3.78$. Age3, by contrast, fell below its null ($z = -0.34$)"),
    # ---------------------------------------------------------------- prose: 5.2.1
    ("age does not show a human between-group signal in this dataset: its observed spread is slightly\nbelow the size-matched null, with $z = -0.37$.",
     "age does not show a human between-group signal in this dataset: its observed spread is slightly\nbelow the size-matched null, with $z = -0.34$."),
    ("null, with $z = +3.61$. Within the selected cells, Asian-Woman raters return 47.8\\% Yes",
     "null, with $z = +3.78$. Within the selected cells, Asian-Woman raters return 47.8\\% Yes"),
    # ---------------------------------------------------------------- prose: 5.3.1
    ("Age3 has $z = -0.31$ with all 123 raters and $z = -0.10$ after\nfiltering. In contrast, the primary race3 $\\times$ gender2 grouping has $z = +4.00$ before\nfiltering and $+2.89$ after filtering.",
     "Age3 has $z = -0.34$ with all 123 raters and $z = -0.01$ after\nfiltering. In contrast, the primary race3 $\\times$ gender2 grouping has $z = +3.78$ before\nfiltering and $+2.75$ after filtering."),
    ("Yet excess spread falls from\n$+0.03836$ to $+0.02526$ because the smaller post-filter cells are noisier.",
     "Yet excess spread falls from\n$+0.03766$ to $+0.02548$ because the smaller post-filter cells are noisier."),
    # ---------------------------------------------------------------- prose: 5.3.2
    ("The gender excess moves from $+0.01347$ ($z = +1.40$) before filtering\nto $-0.00388$ ($z = -0.42$) after, and the mean group-to-pooled distance falls from 0.04585 to\n0.04093.",
     "The gender excess moves from $+0.01301$ ($z = +1.26$) before filtering\nto $-0.00322$ ($z = -0.36$) after, and the mean group-to-pooled distance falls from 0.04585 to\n0.04093."),
    ("itself not statistically significant ($z = +1.40$, $p \\approx 0.16$).",
     "itself not statistically significant ($z = +1.26$, $p \\approx 0.21$)."),
    ("discount: a Bonferroni adjustment raises $p = 0.026$ to approximately 0.10, and it is reported as\nsuggestive without further defence.",
     "discount: a Bonferroni adjustment raises $p = 0.037$ to approximately 0.15, and it is reported as\nsuggestive without further defence."),
    # ---------------------------------------------------------------- prose: 6.1
    ("race3 $\\times$ gender2 grouping still has positive excess spread, $+0.02526$, with $z = +2.89$.",
     "race3 $\\times$ gender2 grouping still has positive excess spread, $+0.02548$, with $z = +2.75$."),
    ("between-group spread falls relative to random removal, moving from $z = +1.40$ to $z = -0.42$,",
     "between-group spread falls relative to random removal, moving from $z = +1.26$ to $z = -0.36$,"),
    # ---------------------------------------------------------------- prose: 7.3, abstract
    ("The compression\ncomparison (Section~\\ref{sec:compress}, $p = 0.026$) is \\emph{post-hoc}",
     "The compression\ncomparison (Section~\\ref{sec:compress}, $p = 0.037$) is \\emph{post-hoc}"),
    ("the only axis on which between-group spread falls relative to random removal ($p = 0.026$), but\nthat comparison was selected after inspecting four groupings",
     "the only axis on which between-group spread falls relative to random removal ($p = 0.037$), but\nthat comparison was selected after inspecting four groupings"),
    # ---------------------------------------------------------------- 8 conclusion
    ("($+0.02526$, $z = +2.89$). At this replication depth the filter does not produce a general",
     "($+0.02548$, $z = +2.75$). At this replication depth the filter does not produce a general"),
]

FOOTNOTE = """
\\vspace{0.5em}
{\\footnotesize\\raggedright Null values are estimated by Monte Carlo permutation and differ
slightly between Tables~\\ref{tab:schemes} and~\\ref{tab:spread} because each is computed from an
independent set of 500 permutations. The substantive ordering is unchanged.\\par}
"""


def main():
    s = open(P, encoding="utf-8").read()
    hit = miss = 0
    for old, new in SUBS:
        if old in s:
            s = s.replace(old, new, 1)
            hit += 1
        else:
            miss += 1
            print(f"  MISS: {old[:78]}")
    n_fn = s.count(FOOTNOTE.strip())
    s = s.replace(FOOTNOTE, "\n")
    open(P, "w", encoding="utf-8").write(s)
    print(f"\napplied {hit}/{len(SUBS)} substitutions ({miss} missed)")
    print(f"removed {n_fn} permutation-mismatch footnote(s) - both tables now share one run")

    left = re.findall(r"independent set of 500 permutations", s)
    print(f"footnote text remaining: {len(left)}")
    sys.exit(0 if miss == 0 and not left else 1)


if __name__ == "__main__":
    main()
