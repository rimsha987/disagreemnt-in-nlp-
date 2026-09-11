"""Confirm results/SUMMARY_TABLE.md still agrees with the numbers printed in the LaTeX.

SUMMARY_TABLE is generated from results/; the LaTeX tables were transcribed by hand from the
same results. If a merge silently changed a table cell, this catches it. Interpretation is out
of scope here - this checks numbers only.
"""
import re, sys
import pandas as pd

TEX = "tex/main.tex"
SUM = "results/SUMMARY_TABLE.csv"


def tex_body():
    out = []
    for line in open(TEX, encoding="utf-8"):
        if line.lstrip().startswith("%"):
            continue
        out.append(line)
    return "".join(out)


# (label, value as printed in the paper, source row substring in SUMMARY_TABLE)
CHECKS = [
    ("RQ1 gender omnibus p",      "0.0262",  "Omnibus exact p: Gender"),
    ("RQ1 gender odds ratio",     "3.36",    "Fisher OR: Man removed"),
    ("RQ1 race omnibus p",        "0.1667",  "Omnibus exact p: Race/ethnicity"),
    ("RQ1 age omnibus p",         "0.4523",  "Omnibus exact p: Age group"),
    ("RQ1 education omnibus p",   "0.5907",  "Omnibus exact p: Education"),
    ("mean TVD label shift",      "0.01556", "Label shift, mean TVD"),
    ("mean JSD label shift",      "0.00055", "Label shift, mean JSD"),
    ("tertile low TVD",           "0.01281", "TVD by disagreement tertile: low"),
    ("tertile medium TVD",        "0.01580", "TVD by disagreement tertile: medium"),
    ("tertile high TVD",          "0.01810", "TVD by disagreement tertile: high"),
    ("spread all123 primary",     "0.20315", "Spread [race3 x gender2 (6 chosen cells)] All 123"),
    ("spread kept primary",       "0.20900", "Spread [race3 x gender2 (6 chosen cells)] 104 kept"),
    ("spread all123 gender2",     "0.09170", "Spread [gender2] All 123"),
    ("spread kept gender2",       "0.08186", "Spread [gender2] 104 kept"),
    ("nullB gender delta",        "-0.00984", "Filtering vs random-19 [gender2]"),
    ("nullB primary delta",       "0.00585", "Filtering vs random-19 [race3 x gender2"),
    ("nullB race5 delta",         "0.01584", "Filtering vs random-19 [race5]"),
    ("nullB age3 delta",          "0.01112", "Filtering vs random-19 [age3"),
    ("headline humans n=1",       "0.524",   "Spread: Humans, 1 rater/group"),
    ("headline LLM v1",           "0.133",   "Spread: LLM persona v1"),
    ("headline LLM v2",           "0.200",   "Spread: LLM persona v2"),
    ("headline noise floor",      "0.140",   "Spread: Same-persona noise floor"),
]


def main():
    body = tex_body()
    st = pd.read_csv(SUM)
    st["blob"] = (st.quantity.astype(str) + " | " + st.value.astype(str) + " | "
                  + st.note.fillna("").astype(str))

    print("=" * 92)
    print("SUMMARY_TABLE.md  vs  the numbers printed in the paper")
    print("=" * 92)
    print(f"  {'quantity':<26} {'paper':>10}  in tex  in summary")
    ok = True
    for label, val, key in CHECKS:
        in_tex = val.lstrip("+-") in body or val in body
        rows = st[st.quantity.str.contains(re.escape(key), case=False, na=False)]
        in_sum = bool(len(rows)) and any(val.lstrip("+-") in b for b in rows.blob)
        good = in_tex and in_sum
        ok &= good
        print(f"  {label:<26} {val:>10}  {'yes' if in_tex else 'NO ':>6}  "
              f"{'yes' if in_sum else 'NO':>10}  {'' if good else '  <-- CHECK'}")

    print(f"\n  summary table rows: {len(st)}, source files: {st.source.nunique()}")
    print(f"\n  {'CONSISTENT' if ok else 'DISCREPANCY FOUND - investigate above'}")

    # the summary table holds numbers, not interpretation - confirm it makes no claims
    claims = st[st.blob.str.contains(
        r"erase|disappear|independent observ|clear imbalance", case=False, na=False)]
    print(f"\n  interpretive claims in SUMMARY_TABLE: {len(claims)} "
          f"({'none, as intended' if len(claims) == 0 else 'REVIEW THESE'})")
    for c in claims.itertuples():
        print(f"     {c.quantity}: {c.note}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
