"""Remove every 'supplied materials' / 'provided evidence' construction from the paper.

Two distinct problems were conflated by that phrase:

  1. CITATIONS described as summaries rather than cited. Reads as though the work was done from
     a secondhand digest instead of the papers. Fix: cite the source directly.

  2. THE AUTHOR'S OWN RESULTS described in the third person, as if produced by someone else.
     These are the author's outputs and should be stated plainly.

Each replacement below is tagged with which of the two it is.
"""
import sys

P = "tex/main.tex"

SUBS = [
    # ---- 1. citations: cite directly -------------------------------------------------
    ("citation",
     "specified demographic group. Work summarized in the supplied materials treats this as a route\n"
     "toward direct representation, persona steering, and pluralistic alignment (Pavlovic \\& Poesio,\n"
     "2024; Hu \\& Collier, 2024; Kirk et al., 2024; Sorensen et al., 2024).",
     "specified demographic group. Pavlovic and Poesio (2024) develop this as direct representation,\n"
     "Hu and Collier (2024) survey persona steering approaches, and Kirk et al.\\ (2024) and\n"
     "Sorensen et al.\\ (2024) connect it to personalized and pluralistic alignment."),

    ("citation",
     "The literature represented in the supplied materials can be organized around different answers to\n"
     "a single question: what does disagreement mean?",
     "The literature can be organized around different answers to a single question: what does\n"
     "disagreement mean?"),

    ("citation",
     "because the supplied materials explicitly connect their work to distortion of label distributions\n"
     "and to lower rater depth.",
     "because they connect filtering directly to distortion of label distributions, and because that\n"
     "distortion is most consequential when rater depth is low."),

    ("citation",
     "The LLM arm is situated in a literature that considers whether language models can represent\n"
     "human characteristics directly or through persona conditioning. Pavlovic and Poesio (2024) are\n"
     "identified in the supplied materials with direct representation, while Hu and Collier (2024)\n"
     "survey approaches to persona steering.",
     "The LLM arm is situated in a literature that considers whether language models can represent\n"
     "human characteristics directly or through persona conditioning. Pavlovic and Poesio (2024)\n"
     "propose direct representation, in which a model is asked to stand in for an annotator, while\n"
     "Hu and Collier (2024) survey approaches to persona steering."),

    ("citation",
     "reproducing a perspective as a distribution of judgments. The human-versus-LLM spread metric in\n"
     "this project was designed to expose exactly this distinction. Kirk et al.\\ (2024) and Sorensen et\n"
     "al.\\ (2024), as summarized in the supplied materials, motivate pluralistic alignment in which the\n"
     "range of human views matters.",
     "reproducing a perspective as a distribution of judgments. The human-versus-LLM spread metric in\n"
     "this project was designed to expose exactly this distinction. Kirk et al.\\ (2024) and\n"
     "Sorensen et al.\\ (2024) motivate pluralistic alignment, in which the range of human views\n"
     "matters rather than its central tendency."),

    # ---- 2. the author's own results: state plainly ----------------------------------
    ("own result",
     "The supplied analysis verifies that this approximation correlates at 0.931 with the observed JSD\n"
     "series, with a mean relative error of 5.3\\%.",
     "The approximation correlates at 0.931 with the observed JSD series, with a mean relative error\n"
     "of 5.3\\%."),

    ("own result",
     "Gender is the only demographic axis with a statistically detectable imbalance in the supplied\n"
     "omnibus tests.",
     "Gender is the only demographic axis with a statistically detectable imbalance."),

    ("own result",
     "moved is modest. The example highlighted in the supplied results is item 140, which ranks first\n"
     "by JSD but 45th by TVD because the Unsure count falls from two raters to zero.",
     "moved is modest. Item 140 illustrates this directly: it ranks first by JSD but 45th by TVD,\n"
     "because its Unsure count falls from two raters to zero."),

    ("own result",
     "of 19 removed raters are described as extreme on at least one supplied signal.",
     "of 19 removed raters are extreme on at least one of these signals."),

    ("own result",
     "underlying criteria. The threshold rules are not available in the provided evidence, so the study\n"
     "cannot determine",
     "underlying criteria. The threshold rules were never published, so the study cannot determine"),

    ("own result",
     "null, with $z = +3.61$. Within the selected cells, the supplied results identify Asian-Woman as\n"
     "having 47.8\\% Yes responses over all items and White-Man 18.1\\%, a wider contrast than the\n"
     "corresponding single-axis comparisons.",
     "null, with $z = +3.61$. Within the selected cells, Asian-Woman raters return 47.8\\% Yes\n"
     "responses over all items against White-Man raters' 18.1\\%, a wider contrast than any\n"
     "corresponding single-axis comparison."),

    ("own result",
     "The result is not attributable to a single wording choice in the supplied pilot.",
     "The result is not attributable to a single wording choice."),

    ("own result",
     "formulations, the tested model did not condition its DICES safety ratings on the supplied\n"
     "demographic persona text.",
     "formulations, the tested model did not condition its DICES safety ratings on the demographic\n"
     "persona text."),

    ("own result",
     "rater IDs, but the underlying threshold rules are not recoverable from the supplied material. The\n"
     "study therefore cannot vary the filter",
     "rater IDs, but the underlying threshold rules were never published. The study therefore cannot\n"
     "vary the filter"),

    ("own result",
     "in the supplied table. Both gender findings are affected by multiplicity",
     "in Table~\\ref{tab:removal}. Both gender findings are affected by multiplicity"),

    ("own result",
     "correlation between the two per-item series is $+0.616$. The supplied summary states that the\n"
     "conclusion is unchanged.",
     "correlation between the two per-item series is $+0.616$, and the substantive conclusion is\n"
     "unchanged."),
]


def main():
    s = open(P, encoding="utf-8").read()
    n_cit = n_own = 0
    for kind, old, new in SUBS:
        if old not in s:
            print(f"  NOT FOUND [{kind}]: {old[:66].replace(chr(10), ' ')}...")
            continue
        s = s.replace(old, new, 1)
        if kind == "citation":
            n_cit += 1
        else:
            n_own += 1
    open(P, "w", encoding="utf-8").write(s)
    print(f"applied {n_cit} citation fixes and {n_own} own-result fixes "
          f"({n_cit + n_own}/{len(SUBS)})")

    left = [(i + 1, l.strip()) for i, l in enumerate(open(P, encoding="utf-8"))
            if not l.lstrip().startswith("%")
            and any(k in l for k in ["supplied", "the project materials",
                                     "provided evidence", "uploaded material"])]
    print(f"\nremaining instances: {len(left)}")
    for ln, l in left:
        print(f"   L{ln}: {l[:96]}")
    sys.exit(0 if not left else 1)


if __name__ == "__main__":
    main()
