"""Freeze the Phase 2 item sample. Draws ONCE and refuses to redraw.

DECISION (Checkpoint 3): reduce to 175 items, keep all 21 conditions. 175 x 21 = 3,675 calls.
Cutting items rather than personas is deliberate: Phase 3 metrics are means over items, so 175
still leaves ~58 per entropy tertile, whereas within-group spread estimated from 2 personas is
not a usable variance estimate. The cheap dimension is the one that got cut.

STRATIFICATION: entropy tertile (low/medium/high) x degree_of_harm. Benign (31 items) and
Debatable (22) are collapsed into one stratum; Extreme (199) and Moderate (98) stand alone, as
instructed. That gives 3 x 3 = 9 strata rather than 3 x 4 = 12, avoiding cells of 2-3 items
where a 50% draw is meaningless. Allocation is proportional at 50% of each stratum, with
largest-remainder rounding to land on exactly 175, so the joint distribution of the sample
matches the full 350.

FORCED INCLUSION: the 15 pilot items are forced into block 1. They were drawn earlier,
stratified by entropy, with a fixed seed and before any response was seen, so including them
introduces no selection on outcome. The benefit is that all 60 pilot calls become reusable as
part of the main run instead of being thrown away.

BLOCKS, not a discard. Every one of the 350 items is assigned a block:
   block 1 = the 175 to run now
   block 2 = the queued continuation, same item-keyed store
If block 1 finishes clean with time in hand, block 2 is an extension, not a restart: the
runner is keyed by (item_id, condition), so adding block 2 adds keys and re-runs nothing.

Output: results/09_item_blocks.csv   FROZEN. This script will not overwrite it.
"""
import sys, os
import numpy as np, pandas as pd
sys.path.insert(0, "src")
import dices_io as D, personas as P

OUT = "results/09_item_blocks.csv"
SEED = 20260826
N_BLOCK1 = 175
HARM_COLLAPSE = {"Extreme": "Extreme", "Moderate": "Moderate",
                 "Benign": "Benign/Debatable", "Debatable": "Benign/Debatable"}
ENT_ORDER = ["low", "medium", "high"]
HARM_ORDER = ["Extreme", "Moderate", "Benign/Debatable"]


def verify(path):
    """Re-check a frozen file instead of redrawing it."""
    b = pd.read_csv(path)
    print(f"FROZEN FILE ALREADY EXISTS: {path}")
    print("Not redrawing. Verifying it instead.\n")
    assert len(b) == 350, len(b)
    assert b.item_id.is_unique
    assert set(b.block) == {1, 2}
    n1 = int((b.block == 1).sum())
    print(f"   items            {len(b)}")
    print(f"   block 1          {n1}")
    print(f"   block 2          {int((b.block == 2).sum())}")
    assert n1 == N_BLOCK1, n1
    print(f"\n   block 1 strata:")
    t = pd.crosstab(b[b.block == 1].ent_bin, b[b.block == 1].harm_stratum)
    print(t.reindex(ENT_ORDER)[[c for c in HARM_ORDER if c in t.columns]].to_string())
    print(f"\n   per entropy tertile in block 1: "
          f"{dict(b[b.block == 1].ent_bin.value_counts())}")
    print(f"\n   VERIFIED. {n1} x {len(P.all_conditions())} = "
          f"{n1 * len(P.all_conditions()):,} calls in block 1.")
    return b


def main():
    if os.path.exists(OUT):
        verify(OUT)
        return

    df = D.load_350()
    items = df.drop_duplicates("item_id")[["item_id", "degree_of_harm", "harm_type"]].copy()
    ent = pd.read_csv("results/03_item_jsd.csv")[["item_id", "entropy_all", "ent_bin"]]
    items = items.merge(ent, on="item_id", validate="one_to_one")
    assert len(items) == 350
    items["harm_stratum"] = items.degree_of_harm.map(HARM_COLLAPSE)
    assert items.harm_stratum.notna().all()
    items["stratum"] = items.ent_bin.astype(str) + " | " + items.harm_stratum

    pilot_items = sorted(set(pd.read_csv("results/06_pilot_plan.csv").item_id.astype(int)))
    print(f"forced inclusions (the 15 pilot items): {pilot_items}")

    print("\n" + "=" * 92)
    print("STRATA IN THE FULL 350")
    print("=" * 92)
    full_ct = pd.crosstab(items.ent_bin, items.harm_stratum).reindex(ENT_ORDER)[HARM_ORDER]
    print(full_ct.to_string())
    print(f"\n   4-level degree_of_harm before collapsing: "
          f"{dict(items.degree_of_harm.value_counts())}")

    # ---- proportional allocation, largest remainder, exactly N_BLOCK1 ------------------
    sizes = items.groupby("stratum").size()
    exact = sizes * (N_BLOCK1 / 350.0)
    take = np.floor(exact).astype(int)
    rem = (exact - take).sort_values(ascending=False)
    short = N_BLOCK1 - int(take.sum())
    for s in rem.index[:short]:
        take[s] += 1
    assert take.sum() == N_BLOCK1, take.sum()

    # forced inclusions must not overflow their stratum quota
    forced = items[items.item_id.isin(pilot_items)]
    for s, n_forced in forced.groupby("stratum").size().items():
        if n_forced > take[s]:
            raise SystemExit(f"stratum {s!r} has {n_forced} forced pilot items but a quota of "
                             f"{take[s]}; the pilot draw and this allocation are incompatible")

    rng = np.random.default_rng(SEED)
    chosen = list(pilot_items)
    print("\n" + "=" * 92)
    print("ALLOCATION (proportional 50%, largest-remainder, forced inclusions honoured)")
    print("=" * 92)
    print(f"   {'stratum':<28} {'pool':>5} {'quota':>6} {'forced':>7} {'drawn':>6}")
    for s in sorted(sizes.index):
        pool = items[items.stratum == s]
        f_ids = [i for i in chosen if i in set(pool.item_id)]
        need = int(take[s]) - len(f_ids)
        cand = sorted(set(pool.item_id) - set(f_ids))
        drawn = sorted(rng.choice(cand, need, replace=False).tolist()) if need > 0 else []
        chosen += drawn
        print(f"   {s:<28} {len(pool):>5} {int(take[s]):>6} {len(f_ids):>7} {len(drawn):>6}")
    chosen = sorted(set(chosen))
    assert len(chosen) == N_BLOCK1, len(chosen)
    assert set(pilot_items) <= set(chosen), "a forced pilot item did not make block 1"

    items["block"] = np.where(items.item_id.isin(chosen), 1, 2)
    items["is_pilot_item"] = items.item_id.isin(pilot_items)
    items = items.sort_values(["block", "item_id"])

    print("\n" + "=" * 92)
    print("RESULT")
    print("=" * 92)
    b1 = items[items.block == 1]
    print(f"   block 1: {len(b1)} items -> {len(b1)*len(P.all_conditions()):,} calls")
    print(f"   block 2: {int((items.block==2).sum())} items (queued continuation)")
    print(f"\n   block 1 strata:")
    print(pd.crosstab(b1.ent_bin, b1.harm_stratum).reindex(ENT_ORDER)[HARM_ORDER].to_string())
    print(f"\n   entropy tertile balance: {dict(b1.ent_bin.value_counts())}")
    print(f"   harm balance:            {dict(b1.harm_stratum.value_counts())}")
    print(f"\n   proportions preserved (block 1 % vs full 350 %):")
    for col in ["ent_bin", "harm_stratum", "degree_of_harm"]:
        a = (b1[col].value_counts(normalize=True) * 100).round(1)
        f = (items[col].value_counts(normalize=True) * 100).round(1)
        for k in f.index:
            print(f"      {col:<15} {str(k):<18} block1 {a.get(k,0.0):>5.1f}%   "
                  f"full {f[k]:>5.1f}%")

    items[["item_id", "block", "ent_bin", "entropy_all", "degree_of_harm", "harm_stratum",
           "harm_type", "is_pilot_item"]].to_csv(OUT, index=False)
    print(f"\n   wrote {OUT}")
    print(f"   THIS FILE IS FROZEN. Re-running this script verifies it and will not redraw.")


if __name__ == "__main__":
    main()
