"""Publication figure style: colourblind-safe, no chartjunk, readable at small size."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Okabe-Ito colourblind-safe qualitative palette.
OKABE_ITO = ["#0072B2", "#D55E00", "#009E73", "#CC79A7",
             "#E69F00", "#56B4E9", "#F0E442", "#000000"]
GREY = "#4D4D4D"
LIGHT = "#BFBFBF"

RC = {
    "figure.dpi": 130,
    "savefig.dpi": 400,
    "savefig.bbox": "tight",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "legend.fontsize": 8.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": GREY,
    "axes.labelcolor": "black",
    "axes.linewidth": 0.8,
    "xtick.color": GREY,
    "ytick.color": GREY,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "axes.grid": False,
    "legend.frameon": False,
    "lines.linewidth": 1.4,
    "figure.autolayout": False,
}


def apply():
    plt.rcParams.update(RC)


def save(fig, stem, outdir="figures"):
    """Write both PDF and PNG. Returns the two paths."""
    pdf, png = f"{outdir}/{stem}.pdf", f"{outdir}/{stem}.png"
    fig.savefig(pdf)
    fig.savefig(png)
    plt.close(fig)
    print(f"   wrote {pdf} and {png}")
    return pdf, png
