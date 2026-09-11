"""Phase 3 headline figure: between-group spread across conditions, with the LLM arm shown
as measured and marked against the same-persona noise floor.

THE COMPARABILITY PROBLEM, AND HOW IT IS HANDLED

Human between-group spread is estimated from 10-19 raters per group, which gives smooth
distributions. The LLM conditions have ONE sample per group, so each "distribution" is a point
mass and TVD between two groups is 0 or 1 on every item. Those are different estimators: the
n=1 version is far noisier and biased upward. Plotting them on one axis without saying so would
be misleading, and the misleading direction would FLATTER the LLM.

So the figure carries both:
  * the full-precision human values (all 350 items, all raters per group), and
  * a LIKE-FOR-LIKE human value: one rater drawn per group, on the same 15 items, 2000 times.
    That is the same estimator the LLM conditions get, so the LLM can be compared to it fairly.

The noise floor from Diagnostic A (same persona, one draw vs one draw) is drawn as a band.
A condition inside that band has not demonstrated any between-group structure at all.

Outputs: figures/fig6_between_group_spread.{pdf,png}, results/13_headline_numbers.csv
"""
import sys, json, glob, itertools
import numpy as np, pandas as pd
sys.path.insert(0, "src")
import dices_io as D, prompts as PR, plotstyle as PS
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

SEED, N_BOOT = 20260826, 2000
RACE3 = {"White": "White", "Black/African American": "Black",
         "Asian/Asian subcontinent": "Asian"}


def llm_tables():
    """Per-item ratings for the pilot (v1) and diagnostic (v2/A) runs."""
    def rd(pat):
        rows = []
        for f in sorted(glob.glob(pat)):
            for line in open(f, encoding="utf-8"):
                if line.strip():
                    r = json.loads(line)
                    r["rating"] = PR.parse_rating(r.get("text"))
                    rows.append(r)
        return pd.DataFrame(rows).drop_duplicates("key", keep="first")
    pil = rd("raw_responses/phase2/pilot_*.jsonl")
    dia = rd("raw_responses/phase2/diag_*.jsonl")
    return pil, dia


def main():
    rng = np.random.default_rng(SEED)
    PS.apply()

    spread_tbl = pd.read_csv("results/12_spread_by_condition.csv")
    PRIM = "race3 x gender2 (6 chosen cells)"
    h_all = spread_tbl[(spread_tbl.grouping == PRIM) &
                       (spread_tbl.condition == "All 123 human raters")].iloc[0]
    h_kept = spread_tbl[(spread_tbl.grouping == PRIM) &
                        (spread_tbl.condition == "104 kept (filtered)")].iloc[0]

    pil, dia = llm_tables()
    items = sorted(set(pil.item_id))
    A = dia[dia.diagnostic == "A"]
    B = dia[dia.diagnostic == "B"]

    # ---- LLM spread: AW#1 vs WM#1, one sample each -> mean TVD = fraction of items differing
    pp = pil.pivot_table(index="item_id", columns="condition", values="rating", aggfunc="first")
    pb = B.pivot_table(index="item_id", columns="condition", values="rating", aggfunc="first")
    llm_v1 = float((pp["Asian-Woman#1"] != pp["White-Man#1"]).mean())
    llm_v2 = float((pb["Asian-Woman#1"] != pb["White-Man#1"]).mean())

    # ---- noise floor: same persona, one draw vs one draw
    pv = A.pivot_table(index="item_id", columns="rep", values="rating", aggfunc="first")
    floor = np.array([float((pv[a] != pv[b]).mean())
                      for a, b in itertools.combinations(sorted(pv.columns), 2)])

    # ---- like-for-like human: ONE rater per group, same 15 items, same estimator
    df = D.load_350()
    r = df[df.rater_race.isin(RACE3)].copy()
    r["cell"] = r.rater_race.map(RACE3) + "-" + r.rater_gender
    r = r[r.item_id.isin(items) & r.cell.isin(["Asian-Woman", "White-Man"])]
    wide = r.pivot_table(index="item_id", columns=["cell", "rater_id"],
                         values=D.LABEL_COL, aggfunc="first")
    aw = wide["Asian-Woman"].to_numpy()
    wm = wide["White-Man"].to_numpy()
    boot = np.empty(N_BOOT)
    for k in range(N_BOOT):
        a = aw[np.arange(len(aw)), rng.integers(0, aw.shape[1], len(aw))]
        b = wm[np.arange(len(wm)), rng.integers(0, wm.shape[1], len(wm))]
        boot[k] = float(np.mean(a != b))
    # ---- full-precision human 2-group value on the same 15 items, for reference
    def dist(sub):
        c = sub[D.LABEL_COL].value_counts().reindex(D.LABELS).fillna(0).to_numpy(float)
        return c / c.sum()
    tv15 = [0.5 * np.abs(dist(r[(r.item_id == i) & (r.cell == "Asian-Woman")]) -
                         dist(r[(r.item_id == i) & (r.cell == "White-Man")])).sum()
            for i in items]
    human15_full = float(np.mean(tv15))

    print("=" * 96)
    print("HEADLINE NUMBERS")
    print("=" * 96)
    print(f"   FULL-PRECISION, all 350 items, 6 cells, all raters per group")
    print(f"      all 123 human raters      TVD {h_all.tvd:.5f}   excess over null "
          f"{h_all.excess:+.5f} (z {h_all.z:+.2f})")
    print(f"      104 kept (filtered)       TVD {h_kept.tvd:.5f}   excess over null "
          f"{h_kept.excess:+.5f} (z {h_kept.z:+.2f})")
    print(f"\n   LIKE-FOR-LIKE, 15 items, 2 groups, ONE draw per group (same estimator as LLM)")
    print(f"      humans, full groups       TVD {human15_full:.5f}  (10 and 11 raters)")
    print(f"      humans, 1 rater per group TVD {boot.mean():.5f}  "
          f"[95% {np.percentile(boot,2.5):.5f}, {np.percentile(boot,97.5):.5f}], {N_BOOT} draws")
    print(f"      LLM persona v1            TVD {llm_v1:.5f}")
    print(f"      LLM persona v2            TVD {llm_v2:.5f}")
    print(f"      same-persona NOISE FLOOR  TVD {floor.mean():.5f}  "
          f"[{floor.min():.5f}, {floor.max():.5f}], {len(floor)} pairs")
    p_v1 = float((floor >= llm_v1).mean())
    p_v2 = float((floor >= llm_v2).mean())
    p_h = float((floor >= boot.mean()).mean())
    print(f"\n   P(noise >= condition):  v1 {p_v1:.2f}   v2 {p_v2:.2f}   "
          f"humans@n=1 {p_h:.2f}")
    print(f"   humans at n=1 sit {'ABOVE' if boot.mean() > floor.max() else 'inside'} "
          f"the floor's observed range; both LLM conditions sit inside it.")

    pd.DataFrame([
        dict(condition="Humans, all 123", scope="350 items, 6 cells, full groups",
             tvd=h_all.tvd, excess=h_all.excess, z=h_all.z, note="full precision"),
        dict(condition="Humans, 104 kept", scope="350 items, 6 cells, full groups",
             tvd=h_kept.tvd, excess=h_kept.excess, z=h_kept.z, note="full precision"),
        dict(condition="Humans, full groups", scope="15 items, 2 groups",
             tvd=human15_full, excess=np.nan, z=np.nan, note="reference"),
        dict(condition="Humans, 1 rater/group", scope="15 items, 2 groups, n=1",
             tvd=boot.mean(), excess=np.nan, z=np.nan,
             note=f"like-for-like; 95% [{np.percentile(boot,2.5):.4f}, {np.percentile(boot,97.5):.4f}]"),
        dict(condition="LLM persona v1", scope="15 items, 2 groups, n=1",
             tvd=llm_v1, excess=np.nan, z=np.nan, note=f"P(noise>=obs)={p_v1:.2f}"),
        dict(condition="LLM persona v2", scope="15 items, 2 groups, n=1",
             tvd=llm_v2, excess=np.nan, z=np.nan, note=f"P(noise>=obs)={p_v2:.2f}"),
        dict(condition="Same-persona noise floor", scope="15 items, n=1", tvd=floor.mean(),
             excess=np.nan, z=np.nan, note=f"range [{floor.min():.4f}, {floor.max():.4f}]"),
    ]).to_csv("results/13_headline_numbers.csv", index=False)

    # ------------------------------------------------------------------------- figure
    fig, axes = plt.subplots(1, 2, figsize=(9.8, 3.9),
                             gridspec_kw=dict(wspace=0.34, width_ratios=[1, 1.22]))

    # LEFT: full-precision human conditions, observed vs size-matched null
    ax = axes[0]
    labs = ["All 123\nhuman raters", "104 kept\n(filtered)"]
    obs = [h_all.tvd, h_kept.tvd]
    nul = [h_all.nullA_tvd, h_kept.nullA_tvd]
    x = np.arange(2)
    ax.bar(x - 0.19, obs, 0.36, color=PS.OKABE_ITO[0], label="observed spread")
    ax.bar(x + 0.19, nul, 0.36, color=PS.LIGHT, label="size-matched null")
    for i in range(2):
        ax.annotate("", xy=(i - 0.19, obs[i]), xytext=(i - 0.19, nul[i]),
                    arrowprops=dict(arrowstyle="<->", color=PS.OKABE_ITO[1], lw=1.1))
        ax.text(i + 0.02, (obs[i] + nul[i]) / 2, f"  excess\n  {obs[i]-nul[i]:+.4f}\n  z={[h_all.z,h_kept.z][i]:+.2f}",
                fontsize=7.2, color=PS.OKABE_ITO[1], va="center")
    ax.set_xticks(x); ax.set_xticklabels(labs)
    ax.set_ylabel("mean pairwise TVD between groups")
    ax.set_ylim(0, 0.28)
    ax.set_title("Human conditions, full precision\n350 items, 6 demographic cells", pad=7)
    ax.legend(loc="upper right", fontsize=7.4)

    # RIGHT: like-for-like, everything at one draw per group
    ax = axes[1]
    ax.axhspan(floor.min(), floor.max(), color=PS.LIGHT, alpha=0.55, zorder=0)
    ax.axhline(floor.mean(), color=PS.GREY, lw=1.0, ls="--", zorder=1)
    names = ["Humans\n1 rater/group", "LLM persona\nv1", "LLM persona\nv2"]
    vals = [boot.mean(), llm_v1, llm_v2]
    cols = [PS.OKABE_ITO[0], PS.OKABE_ITO[1], PS.OKABE_ITO[1]]
    xs = np.arange(3)
    ax.bar(xs, vals, 0.5, color=cols, zorder=3)
    ax.errorbar([0], [boot.mean()],
                yerr=[[boot.mean() - np.percentile(boot, 2.5)],
                      [np.percentile(boot, 97.5) - boot.mean()]],
                fmt="none", ecolor=PS.GREY, capsize=4, lw=1.2, zorder=4)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.012, f"{v:.3f}", ha="center", fontsize=8)
    ax.text(2.52, floor.mean(), " same-persona\n noise floor", va="center",
            fontsize=7.4, color=PS.GREY)
    ax.set_xticks(xs); ax.set_xticklabels(names)
    ax.set_ylabel("mean pairwise TVD between groups")
    ax.set_ylim(0, 0.62)
    ax.set_xlim(-0.6, 3.3)
    ax.set_title("Like-for-like: one draw per group\n15 items, Asian-Woman vs White-Man", pad=7)
    ax.legend(handles=[Patch(facecolor=PS.LIGHT, alpha=0.55,
                             label="range of same-persona noise")],
              loc="upper right", fontsize=7.4)

    fig.suptitle("Between-group spread: real human diversity against a synthetic condition "
                 "that never left the noise floor", y=1.045, fontsize=10.5)
    fig.text(0.5, -0.13,
             "Left: full-precision human spread on all 350 items, against a null that shuffles "
             "rater-to-group labels at fixed group sizes.\nRight: every condition reduced to the "
             "LLM's own estimator (one draw per group) so they are comparable. Both LLM "
             "conditions fall inside\nthe band of pure sampling noise measured from repeating a "
             "single persona; the human condition does not.",
             ha="center", fontsize=7.3, color=PS.GREY)
    PS.save(fig, "fig6_between_group_spread")
    print("\n   wrote results/13_headline_numbers.csv")


if __name__ == "__main__":
    main()
