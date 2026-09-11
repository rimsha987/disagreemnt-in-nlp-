"""Structural sanity check on the LaTeX sources before uploading to Overleaf.

Not a compiler. Checks the failure modes that actually bite: unbalanced environments,
unbalanced braces, missing figure files, and leftover placeholders.
"""
import re, collections, os, sys

BEGIN = re.compile(r"\\begin\{(\w+\*?)\}")
END = re.compile(r"\\end\{(\w+\*?)\}")
GRAPHIC = re.compile(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}")
LABEL = re.compile(r"\\label\{([^}]+)\}")
REF = re.compile(r"\\ref\{([^}]+)\}")


def strip_comments(s):
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


def main():
    files = ["tex/main.tex", "tex/appendix_content.tex"]
    ok = True
    all_labels, all_refs, all_graphics = set(), set(), []
    for f in files:
        if not os.path.exists(f):
            print(f"MISSING FILE {f}")
            ok = False
            continue
        raw = open(f, encoding="utf-8").read()
        live = strip_comments(raw)
        b, e = collections.Counter(BEGIN.findall(live)), collections.Counter(END.findall(live))
        print(f"--- {f} ---")
        for env in sorted(set(b) | set(e)):
            good = b[env] == e[env]
            ok &= good
            print(f"   {env:<14} begin {b[env]:>3}  end {e[env]:>3}  "
                  f"{'OK' if good else '*** MISMATCH ***'}")
        depth = 0
        for ch in live:
            depth += (ch == "{") - (ch == "}")
        ok &= depth == 0
        print(f"   braces: net {depth}  {'OK' if depth == 0 else '*** UNBALANCED ***'}")
        all_labels |= set(LABEL.findall(live))
        all_refs |= set(REF.findall(live))
        all_graphics += [(f, g) for g in GRAPHIC.findall(live)]
        print()

    print("--- figures ---")
    for src, g in all_graphics:
        p = os.path.join("tex", g)
        good = os.path.exists(p)
        ok &= good
        print(f"   {g:<45} {'found' if good else '*** MISSING ***'}")

    print("\n--- cross-references ---")
    dangling = sorted(all_refs - all_labels)
    print(f"   labels {len(all_labels)}, refs {len(all_refs)}, "
          f"dangling {dangling if dangling else 'none'}")
    ok &= not dangling

    print("\n--- remaining placeholders (expected: these are yours to fill) ---")
    for f in files:
        for i, line in enumerate(open(f, encoding="utf-8"), 1):
            if any(k in line for k in ["INFORMATION NEEDED", "[SURNAME]", "[INSTITUTION]",
                                       "[SEMESTER]", "BIBLIOGRAPHIC DETAILS NEEDED"]):
                print(f"   {f}:{i}  {line.strip()[:78]}")

    print(f"\n{'STRUCTURE OK - ready to upload' if ok else 'PROBLEMS FOUND - see above'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
