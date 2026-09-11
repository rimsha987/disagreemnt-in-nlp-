"""Phase 0: schema + counts for DICES-350. Read-only. Prints, writes nothing to data/."""
import pandas as pd, numpy as np, json, os

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 100)

CSV = "data/dices-dataset/350/diverse_safety_adversarial_dialog_350.csv"
OUT = "results"
os.makedirs(OUT, exist_ok=True)

# rater IDs the DICES README says were removed for failing quality checks
REMOVED_RATERS_350 = ['297514565398139','297515609163939','297515750682315','297515617432733',
 '297541515566649','297541515769980','297515629971478','297059995361243','297541522412126',
 '297540556928761','297541321453321','297540562350921','297540983991638','297060365288109',
 '297514543980607','297515729806999','297541271027233','296709611112092','296709543131761']

df = pd.read_csv(CSV, dtype={"rater_id": str})
print("=" * 100)
print(f"FILE: {CSV}")
print(f"SHAPE: {df.shape[0]} rows x {df.shape[1]} columns")
print("=" * 100)

print("\n### FULL SCHEMA ###")
rows = []
for c in df.columns:
    s = df[c]
    nun = s.nunique(dropna=True)
    # show values only when the column is genuinely categorical
    if nun <= 12:
        vals = " | ".join(sorted(str(v) for v in s.dropna().unique()))
        vals = vals[:110] + ("..." if len(vals) > 110 else "")
    else:
        vals = f"<{nun} distinct>"
    rows.append({"column": c, "dtype": str(s.dtype), "n_unique": nun,
                 "n_null": int(s.isna().sum()), "values_or_count": vals})
schema = pd.DataFrame(rows)
for _, r in schema.iterrows():
    print(f"{r['column']:<62} {r['dtype']:<8} uniq={r['n_unique']:<7} null={r['n_null']:<6} {r['values_or_count']}")
schema.to_csv(f"{OUT}/00_schema_350.csv", index=False)

print("\n" + "=" * 100)
print("### CORE COUNTS ###")
print(f"conversations (unique item_id): {df.item_id.nunique()}")
print(f"raters (unique rater_id):       {df.rater_id.nunique()}")
print(f"rows (rater x conversation):    {len(df)}")
print(f"duplicate (rater_id,item_id) pairs: {df.duplicated(['rater_id','item_id']).sum()}")

rpc = df.groupby("item_id").rater_id.nunique()
print(f"\nratings per conversation: min={rpc.min()} median={rpc.median()} max={rpc.max()} mean={rpc.mean():.2f}")
ipr = df.groupby("rater_id").item_id.nunique()
print(f"items per rater:          min={ipr.min()} median={ipr.median()} max={ipr.max()} mean={ipr.mean():.2f}")

print("\n" + "=" * 100)
print("### RATER POOL DEMOGRAPHICS (one row per rater) ###")
raters = df.drop_duplicates("rater_id")[
    ["rater_id","rater_gender","rater_race","rater_age","rater_education","phase"]].copy()
print(f"unique raters: {len(raters)}")
for col in ["rater_gender","rater_race","rater_age","rater_education","phase"]:
    print(f"\n-- {col} --")
    vc = raters[col].value_counts(dropna=False)
    for k, v in vc.items():
        print(f"   {str(k):<70} {v:>4}  ({100*v/len(raters):5.1f}%)")

print("\n" + "=" * 100)
print("### QUALITY FLAG CHECK ###")
flag_like = [c for c in df.columns if any(k in c.lower() for k in
             ["qual","flag","remov","exclud","valid","attention","trap","check"])]
print(f"columns whose name suggests a quality flag: {flag_like if flag_like else 'NONE'}")
data_ids = set(df.rater_id.unique())
rm = set(REMOVED_RATERS_350)
print(f"README removed-rater list: {len(rm)} ids ({len(REMOVED_RATERS_350)} entries, {len(rm)} unique)")
print(f"  of these, present in data: {len(rm & data_ids)}")
print(f"  NOT found in data:         {sorted(rm - data_ids)}")
print(f"raters kept if we apply it:  {len(data_ids - rm)}")
print(f"rows removed if we apply it: {df.rater_id.isin(rm).sum()} / {len(df)} "
      f"({100*df.rater_id.isin(rm).sum()/len(df):.1f}%)")

print("\n" + "=" * 100)
print("### CANDIDATE LABEL COLUMNS ###")
for c in ["Q_overall","Q2_harmful_content_overall","Q3_bias_overall",
          "Q6_policy_guidelines_overall","Q1_whole_conversation_evaluation",
          "safety_gold","degree_of_harm"]:
    if c in df.columns:
        print(f"\n-- {c} --")
        vc = df[c].value_counts(dropna=False)
        for k, v in vc.items():
            print(f"   {str(k)[:80]:<82} {v:>7}  ({100*v/len(df):5.1f}%)")

print("\n-- harm_type (top 30) --")
for k, v in df.harm_type.value_counts(dropna=False).head(30).items():
    print(f"   {str(k)[:80]:<82} {v:>7}")
print(f"   ... {df.harm_type.nunique()} distinct harm_type values total")

print("\n### PER-ITEM metadata consistency (must be 1 value per item) ###")
for c in ["degree_of_harm","harm_type","safety_gold","context","response"]:
    n = df.groupby("item_id")[c].nunique(dropna=False).max()
    print(f"   {c:<20} max distinct values within an item_id = {n}")
