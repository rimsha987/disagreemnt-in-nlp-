"""Generate tex/appendix_content.tex from the source of truth.

Appendices B, C and D document the pipeline: the harm_type grouping, both prompt versions
verbatim, and all 19 persona condition texts. Generating them rather than transcribing keeps
them in sync with src/harm_groups.py, src/prompts.py, src/prompts_v2.py and src/personas.py.
"""
import sys, textwrap
sys.path.insert(0, "src")
import pandas as pd
import personas as P, prompts as PR, prompts_v2 as PR2, harm_groups as HG

SPECIAL = [("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"), ("$", r"\$"),
           ("#", r"\#"), ("_", r"\_"), ("{", r"\{"), ("}", r"\}"),
           ("~", r"\textasciitilde{}"), ("^", r"\textasciicircum{}")]


def esc(s):
    s = str(s)
    for a, b in SPECIAL:
        s = s.replace(a, b)
    return s


def verb(text, width=92):
    """verbatim block, hard-wrapped so nothing overflows the text block."""
    out = []
    for para in text.split("\n"):
        out.extend(textwrap.wrap(para, width) or [""])
    return "\\begin{verbatim}\n" + "\n".join(out) + "\n\\end{verbatim}"


def main():
    L = []
    A = L.append
    items = pd.read_csv("results/item_harm_groups.csv")
    fc, cc = items.fine.value_counts(), items.coarse.value_counts()

    # ---------------------------------------------------------------- Appendix B
    A(r"\section{Harm-type grouping}")
    A("The \\texttt{harm\\_type} field contains 71 comma-joined values built from 25 atomic "
      "tags; 268 of 350 items carry one tag, 78 carry two, and 4 carry three. Items were "
      "assigned to a group by a first-tag rule fixed before any results were inspected: an "
      "item's topic is the group of its first comma-separated tag. The dataset's own tag order "
      "is used because it varies between items in a way that implies it is meaningful rather "
      "than canonical.")
    A("")
    A("This scheme was built for the persona arm's topic breakdown. \\textbf{That arm was "
      "stopped, so no result reported in this paper depends on it.} It is documented here for "
      "reproducibility.")
    A("")
    A(r"\begin{table}[htbp]\centering")
    A(r"\caption{Level 1: ten fine groups, covering all 25 atomic tags.}")
    A(r"\small\begin{tabularx}{\textwidth}{lXr}\toprule")
    A(r"\textbf{Fine group} & \textbf{Atomic tags} & \textbf{Items} \\ \midrule")
    for g, tags in HG.FINE_GROUPS.items():
        A(f"{esc(g)} & {esc(', '.join(tags))} & {int(fc.get(g, 0))} " + r"\\")
    A(r"\midrule \textbf{Total} & 25 tags & \textbf{350} \\ \bottomrule")
    A(r"\end{tabularx}\end{table}")
    A("")
    A(r"\begin{table}[htbp]\centering")
    A(r"\caption{Level 2: four coarse groups.}")
    A(r"\small\begin{tabularx}{\textwidth}{lXr}\toprule")
    A(r"\textbf{Coarse group} & \textbf{Fine groups} & \textbf{Items} \\ \midrule")
    for g, fs in HG.COARSE_GROUPS.items():
        A(f"{esc(g)} & {esc('; '.join(fs))} & {int(cc.get(g, 0))} " + r"\\")
    A(r"\midrule \textbf{Total} & & \textbf{350} \\ \bottomrule")
    A(r"\end{tabularx}\end{table}")
    A("")
    A("A reporting rule was fixed in advance: any cell with fewer than 20 items is reported "
      "with its N but not interpreted.")
    A("")

    # ---------------------------------------------------------------- Appendix C
    A(r"\section{Prompt texts}")
    A("Both prompts force a single-word answer so that parsing cannot drift. The list of "
      "unsafe-reasons, the three rating options, and the output format are identical across "
      "versions; those are the instrument. Version v2 differs in two respects: the one-sided "
      "provocation sentence becomes two-sided, and the persona block moves out of the system "
      "prompt into the user message immediately before the rating question.")
    A("")
    A(r"\subsection{Prompt v1, system}")
    A(verb(PR.RUBRIC + "\n\n[PERSONA BLOCK - see Appendix D]"))
    A(r"\subsection{Prompt v1, user template}")
    A(verb(PR.USER_TEMPLATE))
    A(r"\subsection{Prompt v2, system}")
    A(verb(PR2.RUBRIC_V2))
    A(r"\subsection{Prompt v2, user template}")
    A(verb(PR2.USER_TEMPLATE_V2))
    A("")

    # ---------------------------------------------------------------- Appendix D
    A(r"\section{Persona definitions}")
    A("Six cells times three distinct personas, plus one no-persona baseline: 19 condition "
      "texts. Within a cell the three personas differ on age and education, taken from the "
      "three most common (age, education) profiles actually present among real DICES raters in "
      "that cell.")
    A("")
    prof = pd.read_csv("results/05b_persona_profiles.csv")
    A(r"\begin{table}[htbp]\centering")
    A(r"\caption{Persona profiles, grounded in the real rater pool.}")
    A(r"\small\begin{tabular}{llllr}\toprule")
    A(r"\textbf{Cell} & \textbf{Persona} & \textbf{Age} & \textbf{Education} & "
      r"\textbf{Real raters} \\ \midrule")
    for t in prof.itertuples():
        A(f"{esc(t.cell)} & {esc(t.persona)} & {esc(t.rater_age)} & "
          f"{esc(t.rater_education)} & {t.n_real_raters} " + r"\\")
    A(r"\bottomrule\end{tabular}\end{table}")
    A("")
    A(r"\subsection{Verbatim condition texts}")
    for cell in P.CELLS:
        for k in range(3):
            A(r"\noindent\textbf{" + esc(P.persona_id(cell, k)) + "}")
            A(verb(P.persona_text(cell, k)))
    A(r"\noindent\textbf{baseline} (identical framing, demographic sentences removed)")
    A(verb(P.baseline_text()))

    txt = "\n".join(L) + "\n"
    open("tex/appendix_content.tex", "w", encoding="utf-8").write(txt)
    print(f"wrote tex/appendix_content.tex  ({len(txt):,} chars)")
    print(f"   fine groups {len(HG.FINE_GROUPS)}, coarse {len(HG.COARSE_GROUPS)}, "
          f"persona rows {len(prof)}")


if __name__ == "__main__":
    main()
