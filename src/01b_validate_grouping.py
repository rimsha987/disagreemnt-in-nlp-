"""Validate the fixed harm_type grouping: full coverage, no overlap, resulting cell sizes.
Item counts only. Contains no rating information, so it cannot bias any RQ result."""
import pandas as pd, sys
sys.path.insert(0, "src")
import harm_groups as HG

df = pd.read_csv("data/dices-dataset/350/diverse_safety_adversarial_dialog_350.csv")
items = df.drop_duplicates("item_id")[["item_id","harm_type","degree_of_harm","safety_gold"]].copy()
assert len(items) == 350, len(items)

# --- coverage / disjointness -----------------------------------------------------
data_tags = set(t for h in items.harm_type for t in HG.atomic_tags(h))
scheme_tags = [t for tags in HG.FINE_GROUPS.values() for t in tags]
print("SCHEME VALIDATION")
print(f"  atomic tags in data:            {len(data_tags)}")
print(f"  atomic tags in scheme:          {len(scheme_tags)} ({len(set(scheme_tags))} unique)")
print(f"  in data but NOT in scheme:      {sorted(data_tags - set(scheme_tags)) or 'none'}")
print(f"  in scheme but NOT in data:      {sorted(set(scheme_tags) - data_tags) or 'none'}")
print(f"  duplicated across fine groups:  "
      f"{[t for t in set(scheme_tags) if scheme_tags.count(t) > 1] or 'none'}")
fine_in_coarse = [f for fs in HG.COARSE_GROUPS.values() for f in fs]
print(f"  fine groups not in a coarse group: {sorted(set(HG.FINE_GROUPS) - set(fine_in_coarse)) or 'none'}")
print(f"  fine groups in >1 coarse group:    "
      f"{[f for f in set(fine_in_coarse) if fine_in_coarse.count(f) > 1] or 'none'}")

items["fine"]   = items.harm_type.map(HG.fine_group)
items["coarse"] = items.harm_type.map(HG.coarse_group)
assert items.fine.notna().all() and items.coarse.notna().all()

# --- resulting sizes -------------------------------------------------------------
print("\nFINE GROUPS (primary = first-listed tag; each item counted once)")
fc = items.fine.value_counts().reindex(HG.FINE_ORDER).fillna(0).astype(int)
for g, n in fc.items():
    flag = "   << below MIN_CELL_N, will not be interpreted" if n < HG.MIN_CELL_N else ""
    print(f"   {g:<32} {n:>4} items ({100*n/350:5.1f}%){flag}")
print(f"   {'TOTAL':<32} {fc.sum():>4}")

print("\nCOARSE GROUPS")
cc = items.coarse.value_counts().reindex(HG.COARSE_ORDER).fillna(0).astype(int)
for g, n in cc.items():
    flag = "   << below MIN_CELL_N" if n < HG.MIN_CELL_N else ""
    print(f"   {g:<32} {n:>4} items ({100*n/350:5.1f}%){flag}")
print(f"   {'TOTAL':<32} {cc.sum():>4}")

print("\nCOARSE x degree_of_harm  (the Phase 2 breakdown cells)")
ct = pd.crosstab(items.coarse, items.degree_of_harm).reindex(HG.COARSE_ORDER)
ct = ct[[c for c in ["Benign","Debatable","Moderate","Extreme"] if c in ct.columns]]
ct["ALL"] = ct.sum(axis=1)
print(ct.to_string())
print(f"\n   cells below MIN_CELL_N={HG.MIN_CELL_N}: "
      f"{int((ct.drop(columns='ALL') < HG.MIN_CELL_N).sum().sum())} of "
      f"{ct.drop(columns='ALL').size}")

print("\nMULTI-LABEL ROBUSTNESS VARIANT (item counted in every group it touches)")
ml = items.harm_type.map(HG.all_fine_groups).explode().value_counts().reindex(HG.FINE_ORDER).fillna(0).astype(int)
for g, n in ml.items():
    print(f"   {g:<32} {n:>4}")
print(f"   {'TOTAL (>350 by construction)':<32} {ml.sum():>4}")

print("\nHOW MANY ITEMS DOES FIRST-TAG vs MULTI-LABEL DISAGREE ON?")
n_multi = (items.harm_type.map(lambda h: len(set(HG.all_fine_groups(h)))) > 1).sum()
print(f"   items whose tags span >1 fine group: {n_multi} / 350 ({100*n_multi/350:.1f}%)")

items[["item_id","harm_type","fine","coarse","degree_of_harm","safety_gold"]].to_csv(
    "results/item_harm_groups.csv", index=False)
print("\nwrote results/item_harm_groups.csv")
