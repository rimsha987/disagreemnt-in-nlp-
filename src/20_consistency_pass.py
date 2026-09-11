"""Document-wide consistency pass.

A section-by-section merge risks leaving the abstract claiming one thing and Section 5.3.2
claiming another. This greps the WHOLE corpus - main text, appendices, and the summary table -
for the framings that were withdrawn, and separately verifies that no reported number moved.

Usage:  .venv/Scripts/python.exe src/20_consistency_pass.py
"""
import re, sys, os

FILES = ["tex/main.tex", "tex/appendix_content.tex", "results/SUMMARY_TABLE.md",
         "FINDINGS.md"]

# Framings withdrawn when the gender claim was downgraded. Each is only a problem in
# a gender/spread context, so hits are printed with their line for judgement.
WITHDRAWN = {
    "disappear": r"\bdisappear\w*",
    "erase": r"\berase\w*|\berasure\b",
    "independent": r"\bindependent\w*",
    "align": r"\balign\w*",
    "clear imbalance": r"clear\s+\w*\s*imbalance",
    "targets the axis": r"target\w*\s+(the\s+)?\w*\s*axis",
    "corroborate": r"\bcorroborat\w*",
}

# Numbers that must NOT move. (label, regex that must appear at least once)
FROZEN = {
    "pooled movement 0.04585 -> 0.04093": r"0\.04585.{0,40}0\.04093",
    "gender delta -0.00984":              r"-?0\.00984",
    "gender z = -2.16":                   r"-2\.16",
    "gender p = 0.026":                   r"0\.026\b",
    "Null B race3xgender2 +0.01103":      r"\+?0\.01103",
    "Null B race5 +0.01037":              r"\+?0\.01037",
    "Null B age3 +0.00911":               r"\+?0\.00911",
    "Table 10 delta race3xgender2":       r"\+?0\.00585",
    "Table 10 delta race5":               r"\+?0\.01584",
    "Table 10 delta age3":                r"\+?0\.01112",
    "RQ1 gender p = 0.0262":              r"0\.0262",
    "RQ1 odds ratio 3.36":                r"3\.36",
    "gender excess +0.01347":             r"\+?0\.01347",
    "gender excess -0.00388":             r"-0\.00388",
    "gender z +1.40":                     r"\+?1\.40",
    "gender z -0.42":                     r"-0\.42",
    "primary excess +0.02526":            r"\+?0\.02526",
    "primary z +2.89":                    r"\+?2\.89",
}


def strip_comments_tex(s):
    out = []
    for line in s.split("\n"):
        i, esc = None, False
        for k, ch in enumerate(line):
            if ch == "\\":
                esc = not esc
                continue
            if ch == "%" and not esc:
                i = k
                break
            esc = False
        out.append(line if i is None else line[:i])
    return "\n".join(out)


def section_of(lines, idx):
    """Nearest preceding \\section or \\subsection heading."""
    for j in range(idx, -1, -1):
        m = re.search(r"\\(?:sub)*section\*?\{([^}]+)\}", lines[j])
        if m:
            return m.group(1)[:44]
        m = re.match(r"#+\s+(.*)", lines[j])
        if m:
            return m.group(1)[:44]
    return "(before first heading)"


def main():
    print("=" * 94)
    print("CONSISTENCY PASS - withdrawn framings across the whole corpus")
    print("=" * 94)
    total = 0
    for f in FILES:
        if not os.path.exists(f):
            print(f"\n  {f}: NOT FOUND")
            continue
        raw = open(f, encoding="utf-8").read()
        body = strip_comments_tex(raw) if f.endswith(".tex") else raw
        lines = body.split("\n")
        hits = []
        for name, pat in WITHDRAWN.items():
            for i, line in enumerate(lines):
                for m in re.finditer(pat, line, re.I):
                    hits.append((i + 1, name, m.group(0), line.strip()))
        print(f"\n--- {f}  ({len(hits)} hits) ---")
        for ln, name, tok, line in sorted(hits):
            total += 1
            print(f"  L{ln:<5} [{name}] '{tok}'")
            print(f"        sec: {section_of(lines, ln-1)}")
            print(f"        {line[:104]}")
    print(f"\n  TOTAL HITS TO ADJUDICATE: {total}")

    print("\n" + "=" * 94)
    print("FROZEN NUMBERS - none of these may move")
    print("=" * 94)
    corpus = ""
    for f in FILES:
        if os.path.exists(f):
            corpus += open(f, encoding="utf-8").read() + "\n"
    ok = True
    for label, pat in FROZEN.items():
        n = len(re.findall(pat, corpus, re.S))
        good = n > 0
        ok &= good
        print(f"  {label:<38} {n:>3} occurrence(s)  {'OK' if good else '*** MISSING ***'}")
    print(f"\n  {'ALL FROZEN NUMBERS PRESENT' if ok else 'A NUMBER WENT MISSING - INVESTIGATE'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
