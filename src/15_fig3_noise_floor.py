"""Replacement for the report's Figure 3.

The version in the submitted draft plots the same-persona noise floor as a single bar at its
MEAN (2.10) beside v1 (2.00) and v2 (3.00). Visually v2's bar is taller than the noise bar,
which invites exactly the opposite of the finding: it reads as "v2 beat the noise".

The floor is a DISTRIBUTION, [1,1,1,1,2,2,2,3,4,4] out of 15, range 1-4. The observed 3/15 sits
at the 70th percentile INSIDE it, with P(noise >= 3) = 0.30. Drawing the floor at its mean
destroys the only thing the figure exists to show. This is the same error the analysis script
originally made (comparing to the mean rather than the tail) and it must not survive into the
paper.

This version shows every one of the 10 same-persona values, the band they span, and the two
observed conditions as points inside it.

Output: figures/fig3_noise_floor.{pdf,png}
"""
import sys, json, glob, itertools
import numpy as np, pandas as pd
sys.path.insert(0, "src")
import prompts as PR, plotstyle as PS
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


def load(pat):
    rows = []
    for f in sorted(glob.glob(pat)):
        for line in open(f, encoding="utf-8"):
            if line.strip():
                r = json.loads(line)
                r["rating"] = PR.parse_rating(r.get("text"))
                rows.append(r)
    return pd.DataFrame(rows).drop_duplicates("key", keep="first")


def main():
    PS.apply()
    pil, dia = load("raw_responses/phase2/pilot_*.jsonl"), load("raw_responses/phase2/diag_*.jsonl")
    A, B = dia[dia.diagnostic == "A"], dia[dia.diagnostic == "B"]
    pv = A.pivot_table(index="item_id", columns="rep", values="rating", aggfunc="first")
    floor = np.array([int((pv[a] != pv[b]).sum())
                      for a, b in itertools.combinations(sorted(pv.columns), 2)])
    pp = pil.pivot_table(index="item_id", columns="condition", values="rating", aggfunc="first")
    pb = B.pivot_table(index="item_id", columns="condition", values="rating", aggfunc="first")
    v1 = int((pp["Asian-Woman#1"] != pp["White-Man#1"]).sum())
    v2 = int((pb["Asian-Woman#1"] != pb["White-Man#1"]).sum())
    within = int((pp["Asian-Woman#1"] != pp["Asian-Woman#2"]).sum())
    p1, p2 = float((floor >= v1).mean()), float((floor >= v2).mean())
    print(f"floor {sorted(floor.tolist())} mean {floor.mean():.2f} range {floor.min()}-{floor.max()}")
    print(f"v1 {v1}/15 P(noise>=)={p1:.2f}   v2 {v2}/15 P(noise>=)={p2:.2f}   within-cell {within}/15")

    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    ax.axhspan(floor.min(), floor.max(), color=PS.LIGHT, alpha=0.6, zorder=0)
    rng = np.random.default_rng(1)
    ax.scatter(np.full(len(floor), 0.0) + rng.uniform(-0.10, 0.10, len(floor)), floor,
               s=42, color=PS.GREY, alpha=0.85, zorder=3, label="10 same-persona pairs")
    ax.hlines(floor.mean(), -0.28, 0.28, color=PS.GREY, lw=1.6, ls="--", zorder=4)
    ax.text(0.32, floor.mean(), f"mean {floor.mean():.2f}", fontsize=7.6,
            color=PS.GREY, va="center")

    for x, val, lab, p in [(1.0, within, "within-cell\nAW#1 vs AW#2", None),
                           (2.0, v1, "between-cell\nprompt v1", p1),
                           (3.0, v2, "between-cell\nprompt v2", p2)]:
        col = PS.OKABE_ITO[1] if p is not None else PS.OKABE_ITO[4]
        ax.scatter([x], [val], s=130, marker="D", color=col, zorder=5, edgecolor="white")
        note = f"{val}/15" + (f"\nP(noise ≥ {val}) = {p:.2f}" if p is not None else "")
        ax.text(x, val + 0.30, note, ha="center", fontsize=7.6, color=col)

    ax.set_xticks([0, 1, 2, 3])
    ax.set_xticklabels(["same-persona\nnoise floor", "within-cell\n(same race+gender)",
                        "between-cell\nv1", "between-cell\nv2"])
    ax.set_ylabel("items differing, out of 15")
    ax.set_ylim(0, 5.4)
    ax.set_xlim(-0.55, 3.75)
    ax.set_title("Persona differences do not leave the same-persona sampling floor", pad=8)
    ax.legend(handles=[Patch(facecolor=PS.LIGHT, alpha=0.6,
                             label="range of pure sampling noise (1–4 of 15)")],
              loc="upper left", fontsize=7.6)
    fig.text(0.5, -0.22,
             "Each grey dot is one pair of repeated draws from the SAME persona, so it contains "
             "no demographic difference by construction.\nBoth between-cell values fall inside "
             "that range; the within-cell value (4/15) is larger than either. Plotting the floor "
             "as a single\nbar at its mean would make v2 appear to exceed it, which the tail "
             "probabilities show it does not.",
             ha="center", fontsize=7.2, color=PS.GREY)
    PS.save(fig, "fig3_noise_floor")


if __name__ == "__main__":
    main()
