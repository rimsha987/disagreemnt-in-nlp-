import pandas as pd
CSV = "data/dices-dataset/350/diverse_safety_adversarial_dialog_350.csv"
df = pd.read_csv(CSV, dtype={"rater_id": str})
items = df.drop_duplicates("item_id")[["item_id","degree_of_harm","harm_type","safety_gold","context","response"]]
print(f"ITEM-LEVEL (n={len(items)} conversations)")
for c in ["degree_of_harm","safety_gold"]:
    print(f"\n-- {c} (per item) --")
    for k,v in items[c].value_counts().items(): print(f"   {k:<15} {v:>4} items ({100*v/len(items):.1f}%)")
print("\n-- harm_type per item, top 15 --")
for k,v in items.harm_type.value_counts().head(15).items(): print(f"   {str(k)[:50]:<52} {v:>4}")

print("\n=== CONVERSATION TEXT DUPLICATION ===")
print(f"unique context strings:            {items.context.nunique()} / 350")
print(f"unique response strings:           {items.response.nunique()} / 350")
print(f"unique (context,response) pairs:   {items.drop_duplicates(['context','response']).shape[0]} / 350")
dup = items[items.duplicated(['context','response'], keep=False)].sort_values(['context','response'])
print(f"items involved in a duplicate pair: {len(dup)}")
if len(dup):
    g = dup.groupby(['context','response']).item_id.apply(list)
    print(f"duplicate groups: {len(g)}; example item_id groups: {[x for x in g.head(5)]}")
    # do duplicated items get the same gold label?
    gg = dup.groupby(['context','response']).agg(n=('item_id','size'),
            golds=('safety_gold', lambda s: s.nunique()), harms=('degree_of_harm', lambda s: s.nunique()))
    print(f"duplicate groups where safety_gold differs:     {(gg.golds>1).sum()}")
    print(f"duplicate groups where degree_of_harm differs:  {(gg.harms>1).sum()}")

print("\n=== RATER CROSS-TABS (for persona cell planning) ===")
r = df.drop_duplicates("rater_id")
print("\ngender x race:"); print(pd.crosstab(r.rater_race, r.rater_gender, margins=True))
print("\nage x race:");    print(pd.crosstab(r.rater_race, r.rater_age, margins=True))
print("\nage x gender:");  print(pd.crosstab(r.rater_age, r.rater_gender, margins=True))
print("\nraw_race detail:")
for k,v in r.rater_raw_race.value_counts().items(): print(f"   {str(k)[:95]:<97} {v:>3}")

print("\n=== Q_overall 'Yes' rate by demographic group (raw, all 123 raters) ===")
df["yes"] = (df.Q_overall=="Yes").astype(int)
for c in ["rater_gender","rater_race","rater_age","rater_education"]:
    print(f"\n-- {c} --")
    t = df.groupby(c).agg(n_raters=("rater_id","nunique"), n_ratings=("yes","size"), pct_unsafe=("yes","mean"))
    t["pct_unsafe"] = (100*t.pct_unsafe).round(2)
    print(t.to_string())
