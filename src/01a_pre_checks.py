"""Pre-analysis inspection only. No results computed here.
(a) enumerate every harm_type value, to build the grouping scheme
(b) hunt for 'Prefer not to answer' / non-response categories in all demographic columns
"""
import pandas as pd
df = pd.read_csv("data/dices-dataset/350/diverse_safety_adversarial_dialog_350.csv",
                 dtype={"rater_id": str})
items = df.drop_duplicates("item_id")

print("="*90)
print("(a) ALL harm_type VALUES, counted at ITEM level (n=350)")
print("="*90)
vc = items.harm_type.value_counts()
print(f"{len(vc)} distinct raw strings\n")
for k, v in vc.items():
    print(f"   {v:>4}  {k}")

print("\n--- split on comma: atomic tag frequency (an item may carry several) ---")
tags = items.harm_type.str.split(",").explode().str.strip()
tvc = tags.value_counts()
print(f"{len(tvc)} distinct atomic tags\n")
for k, v in tvc.items():
    print(f"   {v:>4}  {k}")
print(f"\nitems with 1 tag: {(items.harm_type.str.count(',')==0).sum()}, "
      f"2 tags: {(items.harm_type.str.count(',')==1).sum()}, "
      f"3+ tags: {(items.harm_type.str.count(',')>=2).sum()}")

print("\n"+"="*90)
print("(b) NON-RESPONSE / 'PREFER NOT TO ANSWER' CHECK")
print("="*90)
NEEDLES = ["prefer","not to answer","decline","no answer","n/a","na","none","unknown",
           "other","unspecified","not say","rather not","blank",""]
demo_cols = ["rater_gender","rater_race","rater_raw_race","rater_age","rater_education"]
raters = df.drop_duplicates("rater_id")
for c in demo_cols:
    print(f"\n-- {c}  ({raters[c].nunique()} levels, {raters[c].isna().sum()} null) --")
    for k, v in raters[c].value_counts(dropna=False).items():
        hit = [n for n in NEEDLES if n and n in str(k).lower()]
        mark = f"   <== matches {hit}" if hit else ""
        print(f"   {str(k)[:88]:<90} {v:>4}{mark}")
    # empty strings / whitespace-only
    blanks = raters[c].astype(str).str.strip().eq("").sum()
    print(f"   [whitespace-only or empty string values: {blanks}]")
