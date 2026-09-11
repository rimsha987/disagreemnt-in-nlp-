"""Phase 2 design: choose the 6 persona cells from the real DICES rater pool.

The choice must not be made by eye. RQ2 asks whether a persona lands closest to the human
group it imitates; if the human groups chosen barely differ from each other, RQ2 has no
signal to detect regardless of how the model behaves. So candidate cell schemes are scored
by the quantity Phase 3 will measure: BETWEEN-GROUP SPREAD.

  spread(scheme) = mean over items of [ mean pairwise TVD between the groups'
                                        (No, Yes, Unsure) distributions on that item ]

Small groups produce noisy per-item distributions, and noise inflates pairwise distance. So
raw spread is not comparable across schemes with different cell sizes. Each scheme is scored
against a permutation null that randomly reassigns raters to groups while PRESERVING CELL
SIZES; the reported effect is observed minus null, and z = (observed - null) / null sd.
That isolates real demographic signal from small-cell noise.

Primary metric TVD (Checkpoint 2); JSD reported alongside.

Outputs: results/05_cell_scheme_scores.csv, results/05_chosen_cells.csv
"""
import sys, itertools
import numpy as np, pandas as pd
sys.path.insert(0, "src")
import dices_io as D

RNG_SEED, N_PERM = 20260826, 500


def build_tensor(df):
    """items x raters x 3 one-hot, plus the rater id order."""
    code = {l: i for i, l in enumerate(D.LABELS)}
    piv = df.pivot(index="item_id", columns="rater_id", values=D.LABEL_COL)
    Mi = np.vectorize(code.get)(piv.to_numpy())
    oh = np.zeros((*Mi.shape, 3))
    np.put_along_axis(oh, Mi[:, :, None], 1, axis=2)
    return oh, list(piv.columns), list(piv.index)


def spread(oh, group_idx):
    """Mean pairwise TVD and JSD between group distributions, averaged over items.
    group_idx: list of arrays of rater column indices, one per group."""
    P = []
    for idx in group_idx:
        c = oh[:, idx, :].sum(axis=1)
        P.append(c / c.sum(axis=1, keepdims=True))
    P = np.stack(P)                                   # groups x items x 3
    tv, js = [], []
    for a, b in itertools.combinations(range(len(P)), 2):
        tv.append(0.5 * np.abs(P[a] - P[b]).sum(axis=2 - 1))
        m = 0.5 * (P[a] + P[b])
        H = lambda x: -(np.where(x > 0, x * np.log2(np.where(x > 0, x, 1)), 0.0)).sum(axis=1)
        js.append(H(m) - 0.5 * (H(P[a]) + H(P[b])))
    return float(np.mean(tv)), float(np.mean(js))


def main():
    df = D.load_350()
    oh, rater_ids, item_ids = build_tensor(df)
    r = D.rater_table(df).set_index("rater_id").loc[rater_ids].reset_index()
    assert list(r.rater_id) == rater_ids and len(r) == 123
    rng = np.random.default_rng(RNG_SEED)

    RACE3 = ["White", "Black/African American", "Asian/Asian subcontinent"]

    def cells_from(labels):
        """labels: array of group name per rater (None = excluded). Returns (names, idx)."""
        names = [g for g in pd.unique(labels) if g is not None]
        names = sorted(names)
        return names, [np.where(labels == g)[0] for g in names]

    gen, race, age = r.rater_gender.to_numpy(), r.rater_race.to_numpy(), r.rater_age.to_numpy()
    SCHEMES = {
        "race5 (all race groups)":
            np.array([x for x in race], dtype=object),
        "gender2":
            np.array([x for x in gen], dtype=object),
        "age3":
            np.array([x for x in age], dtype=object),
        "race3 x gender2 (White/Black/Asian)":
            np.array([f"{D.SHORT[a]}-{b}" if a in RACE3 else None
                      for a, b in zip(race, gen)], dtype=object),
        "gender2 x age3":
            np.array([f"{b}-{a}" for a, b in zip(age, gen)], dtype=object),
        "race5 + gender2 marginal (7 cells)":
            None,   # handled separately below, not a partition
        "race3 (White/Black/Asian only)":
            np.array([D.SHORT[a] if a in RACE3 else None for a in race], dtype=object),
    }

    rows = []
    print("=" * 100)
    print("CANDIDATE CELL SCHEMES, SCORED BY BETWEEN-GROUP SPREAD")
    print("all 123 raters, 350 items, unfiltered pool")
    print("=" * 100)
    for name, labels in SCHEMES.items():
        if labels is None:
            continue
        names, idx = cells_from(labels)
        sizes = [len(i) for i in idx]
        obs_t, obs_j = spread(oh, idx)

        # permutation null: reshuffle rater -> group, preserving cell sizes
        assigned = np.concatenate(idx)
        null_t = np.empty(N_PERM)
        for k in range(N_PERM):
            perm = rng.permutation(assigned)
            cuts, s = [], 0
            for n in sizes:
                cuts.append(perm[s:s + n]); s += n
            null_t[k] = spread(oh, cuts)[0]
        z = (obs_t - null_t.mean()) / null_t.std()
        print(f"\n-- {name} --")
        print(f"   {len(names)} cells, sizes {sizes} (min {min(sizes)}, total {sum(sizes)} raters)")
        print(f"   observed mean pairwise TVD  {obs_t:.5f}   (JSD {obs_j:.6f})")
        print(f"   size-matched null           {null_t.mean():.5f} +/- {null_t.std():.5f}")
        print(f"   excess over null            {obs_t-null_t.mean():+.5f}   z = {z:+.2f}")
        rows.append(dict(scheme=name, n_cells=len(names), min_cell=min(sizes),
                         n_raters=sum(sizes), obs_tvd=obs_t, obs_jsd=obs_j,
                         null_tvd=null_t.mean(), null_sd=null_t.std(),
                         excess=obs_t - null_t.mean(), z=z, cells="|".join(names)))
    out = pd.DataFrame(rows).sort_values("z", ascending=False)
    out.to_csv("results/05_cell_scheme_scores.csv", index=False)

    print("\n" + "=" * 100)
    print("RANKED BY z (real demographic signal above size-matched noise)")
    print("=" * 100)
    print(out[["scheme", "n_cells", "min_cell", "obs_tvd", "null_tvd", "excess", "z"]]
          .to_string(index=False, float_format=lambda v: f"{v:.5f}"))

    # ------------------------------------------------------------------ the chosen scheme
    CHOSEN = "race3 x gender2 (White/Black/Asian)"
    labels = SCHEMES[CHOSEN]
    names, idx = cells_from(labels)
    print("\n" + "=" * 100)
    print(f"CHOSEN: {CHOSEN}  -- 6 cells, a clean 2x3 factorial")
    print("=" * 100)
    sel = r[[l is not None for l in labels]]
    print(f"   covers {len(sel)} of 123 raters "
          f"({100*len(sel)/123:.0f}%); excludes Latine/x (22) and Multiracial (16)")
    ch = []
    print(f"\n   {'cell':<16} {'n raters':>9} {'n removed':>10} {'% Yes (all)':>12} "
          f"{'% Yes (kept)':>13} {'ratings/item':>13}")
    for g, i in zip(names, idx):
        ids = set(np.array(rater_ids)[i])
        sub = df[df.rater_id.isin(ids)]
        nrem = int(r.iloc[i].removed.sum())
        pk = sub[~sub.removed]
        print(f"   {g:<16} {len(i):>9} {nrem:>10} {100*(sub[D.LABEL_COL]=='Yes').mean():>11.2f}% "
              f"{100*(pk[D.LABEL_COL]=='Yes').mean():>12.2f}% {len(i):>13}")
        ch.append(dict(cell=g, n_raters=len(i), n_removed=nrem, n_kept=len(i) - nrem,
                       pct_yes_all=(sub[D.LABEL_COL] == "Yes").mean(),
                       pct_yes_kept=(pk[D.LABEL_COL] == "Yes").mean(),
                       pct_no_all=(sub[D.LABEL_COL] == "No").mean(),
                       pct_unsure_all=(sub[D.LABEL_COL] == "Unsure").mean(),
                       rater_ids="|".join(sorted(ids))))
    pd.DataFrame(ch).to_csv("results/05_chosen_cells.csv", index=False)

    print(f"\n   pairwise human TVD between the 6 chosen cells, averaged over 350 items:")
    P = []
    for i in idx:
        c = oh[:, i, :].sum(axis=1)
        P.append(c / c.sum(axis=1, keepdims=True))
    print(f"   {'':<16}" + "".join(f"{g:>16}" for g in names))
    for a in range(len(names)):
        line = f"   {names[a]:<16}"
        for b in range(len(names)):
            line += ("{:>16.4f}".format(np.mean(0.5 * np.abs(P[a] - P[b]).sum(axis=1)))
                     if a != b else f"{'-':>16}")
        print(line)

    print(f"\n   WHY THIS SCHEME, in one place:")
    print(f"   * race carries the real human spread (Phase 0: 27.4%-38.2% Yes across race,")
    print(f"     versus ~3 points across gender and ~3 across age), so RQ2 has something to")
    print(f"     detect. A gender x age scheme would ask the model to reproduce differences")
    print(f"     the humans barely show.")
    print(f"   * crossing with gender ties Phase 2 to Phase 1: gender is the axis the")
    print(f"     published filter is actually biased on (men removed at 3.4x, p=0.026).")
    print(f"   * it is a balanced 2x3 factorial, so main effects and the interaction are")
    print(f"     separable, which a ragged 6-cell pick would not allow.")
    print(f"   * smallest cell is 10 raters. Thin, and reported as such everywhere.")
    print(f"\n   COST: Latine/x and Multiracial raters are excluded from the persona")
    print(f"   comparison. Multiracial has the HIGHEST %Yes in the pool (38.2), so the")
    print(f"   chosen cells span a narrower range than the full pool does. This understates")
    print(f"   human diversity and must be stated as a limitation, not buried. Both groups")
    print(f"   remain in the Phase 3 human conditions, which use all 123 raters.")


if __name__ == "__main__":
    main()
