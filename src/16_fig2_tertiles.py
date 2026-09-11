"""Clean version of the report's Figure 2: mean TVD by disagreement tertile.

The draft's version is correct but omits the group sizes and the test statistic. This adds
both, per the brief's rule that every reported quantity carries its N.

Output: figures/fig2_tertiles.{pdf,png}
"""
import sys
import numpy as np, pandas as pd
from scipy import stats
sys.path.insert(0, "src")
import plotstyle as PS
import matplotlib.pyplot as plt


def main():
    PS.apply()
    eb = pd.read_csv("results/03_entropy_bins.csv").set_index("bin").loc[["low", "medium", "high"]]
    it = pd.read_csv("results/03_item_jsd.csv")
    kw = stats.kruskal(*[it[it.ent_bin == b].tvd.values for b in ["low", "medium", "high"]])

    fig, ax = plt.subplots(figsize=(6.2, 3.5))
    x = np.arange(3)
    ax.bar(x, eb.mean_tvd, 0.55, color=PS.OKABE_ITO[0])
    for i, (v, n) in enumerate(zip(eb.mean_tvd, eb.n)):
        ax.text(i, v + 0.00035, f"{v:.5f}", ha="center", fontsize=8.5)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{b.capitalize()}\n(n = {int(n)})" for b, n in zip(eb.index, eb.n)])
    ax.set_ylabel("mean TVD, filtered vs unfiltered")
    ax.set_xlabel("disagreement tertile (entropy of the unfiltered distribution)")
    ax.set_ylim(0, 0.0215)
    ax.set_title("Filtering shifts labels more on contested items", pad=8)
    # Top-left. The previous position (bottom-right, 0.98/0.06) sat on top of the
    # "High" bar, which only became obvious once the figure was scaled down.
    ax.text(0.02, 0.97, f"Kruskal-Wallis H = {kw.statistic:.2f}, p = {kw.pvalue:.1e}\n"
                        f"majority flips: 0 / 0 / 9",
            transform=ax.transAxes, ha="left", va="top", fontsize=7.8, color=PS.GREY)
    PS.save(fig, "fig2_tertiles")
    print(f"   tertile means: {eb.mean_tvd.round(5).tolist()}, n = {eb.n.tolist()}")
    print(f"   Kruskal-Wallis H={kw.statistic:.3f} p={kw.pvalue:.3g}")


if __name__ == "__main__":
    main()
