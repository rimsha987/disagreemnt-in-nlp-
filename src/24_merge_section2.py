"""Merge the expanded Section 2 into main.tex and add the five new references.

The expansion is grounded in the five PDFs supplied and extracted to papers/. It replaces the
one-paragraph 2.2, extends 2.3 with Sang & Stanton, and adds 2.5 and 2.6.
"""
import sys, re

MAIN = "tex/main.tex"
NEW = "tex/section2_expanded.tex"

# ---- the old 2.2, replaced wholesale -------------------------------------------------
OLD_22_START = r"\subsection{Taxonomies of disagreement}"
OLD_22_END = r"\subsection{Rater quality and filtering}"

# ---- 2.3: add Sang & Stanton back, now that it is actually read ----------------------
OLD_23 = """Homan et al.\\ (2024) provides additional motivation for examining intersectional and individual
differences rather than treating demographic groups as interchangeable."""
NEW_23 = """Homan et al.\\ (2024) and Sang and Stanton (2022) provide additional motivation for examining
intersectional and individual differences rather than treating demographic groups as
interchangeable."""

# ---- label the filter subsection so 2.2 can point at it ------------------------------
OLD_LBL = r"\subsection{The published rater filter}"
NEW_LBL = "\\subsection{The published rater filter}\n\\label{sec:published-filter}"

# ---- references ----------------------------------------------------------------------
ANCHOR = r"\item Fleisig, E., et al. (2025). [FULL BIBLIOGRAPHIC DETAILS NEEDED]."
NEW_REFS = r"""\item Fleisig, E., et al. (2025). [FULL BIBLIOGRAPHIC DETAILS NEEDED].
\item Gordon, M. L., Lam, M. S., Park, J. S., Patel, K., Hancock, J. T., Hashimoto, T.,
  \& Bernstein, M. S. (2022). Jury Learning: Integrating Dissenting Voices into Machine
  Learning Models. In \emph{Proceedings of the 2022 CHI Conference on Human Factors in
  Computing Systems (CHI '22)}. ACM. \url{https://doi.org/10.1145/3491102.3502004}"""

REF_AFTER_HOMAN = r"\item Hu, Z., \& Collier, N. (2024). [FULL BIBLIOGRAPHIC DETAILS NEEDED]."
REF_JIANG = r"""\item Hu, Z., \& Collier, N. (2024). [FULL BIBLIOGRAPHIC DETAILS NEEDED].
\item Jiang, N.-J., \& de Marneffe, M.-C. (2022). Investigating Reasons for Disagreement in
  Natural Language Inference. \emph{Transactions of the Association for Computational
  Linguistics}, 10, 1357--1374. \url{https://doi.org/10.1162/tacl_a_00523}"""

REF_AFTER_PAV = r"\item Pavlovic, M., \& Poesio, M. (2024). [FULL BIBLIOGRAPHIC DETAILS NEEDED]."
REF_ROMBERG_SANG = r"""\item Pavlovic, M., \& Poesio, M. (2024). [FULL BIBLIOGRAPHIC DETAILS NEEDED].
\item Romberg, J. (2022). Is Your Perspective Also My Perspective? Enriching Prediction with
  Subjectivity. In \emph{Proceedings of the 9th Workshop on Argument Mining}, 115--125.
\item Sang, Y., \& Stanton, J. (2022). The Origin and Value of Disagreement Among Data Labelers:
  A Case Study of Individual Differences in Hate Speech Annotation. arXiv:2112.04030.
  [CHECK: a peer-reviewed version may exist; the supplied PDF is the preprint.]"""

REF_AFTER_SOR = r"\item Sorensen, T., et al. (2024). [FULL BIBLIOGRAPHIC DETAILS NEEDED]."
REF_WEBER = r"""\item Sorensen, T., et al. (2024). [FULL BIBLIOGRAPHIC DETAILS NEEDED].
\item Weber-Genzel, L., Peng, S., de Marneffe, M.-C., \& Plank, B. (2024). VariErr NLI:
  Separating Annotation Error from Human Label Variation. In \emph{Proceedings of the 62nd
  Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)},
  2256--2269."""


def main():
    s = open(MAIN, encoding="utf-8").read()
    new_sec = open(NEW, encoding="utf-8").read()
    # drop the file's own header comment block
    new_sec = new_sec[new_sec.index(r"\subsection{Taxonomies of disagreement}"):].rstrip() + "\n\n"

    i, j = s.find(OLD_22_START), s.find(OLD_22_END)
    if i < 0 or j < 0 or j < i:
        sys.exit("could not locate the old 2.2 block")
    old_words = len(s[i:j].split())
    s = s[:i] + new_sec + s[j:]
    print(f"replaced 2.2: {old_words} words -> {len(new_sec.split())} words")

    for name, old, new in [("2.3 Sang & Stanton", OLD_23, NEW_23),
                           ("filter label", OLD_LBL, NEW_LBL),
                           ("ref Gordon", ANCHOR, NEW_REFS),
                           ("ref Jiang", REF_AFTER_HOMAN, REF_JIANG),
                           ("ref Romberg+Sang", REF_AFTER_PAV, REF_ROMBERG_SANG),
                           ("ref Weber-Genzel", REF_AFTER_SOR, REF_WEBER)]:
        if old in s:
            s = s.replace(old, new, 1)
            print(f"  applied: {name}")
        else:
            print(f"  MISS: {name}")

    if "\\usepackage{url}" not in s and "\\url{" in s:
        s = s.replace("\\usepackage[hidelinks]{hyperref}",
                      "\\usepackage[hidelinks]{hyperref}   % hyperref provides \\url")
    open(MAIN, "w", encoding="utf-8").write(s)

    body = "".join(l for l in open(MAIN, encoding="utf-8") if not l.lstrip().startswith("%"))
    m = re.search(r"\\section\{Background and Related Work\}(.*?)\\section\{Data\}", body, re.S)
    print(f"\nSection 2 now {len(m.group(1).split()):,} words "
          f"(~{len(m.group(1).split())/480:.1f} pages)")
    n_refs = body.count(r"\item ")
    print(f"reference entries: {len([l for l in body.split(chr(10)) if l.strip().startswith(chr(92)+'item') and '(' in l])}")


if __name__ == "__main__":
    main()
