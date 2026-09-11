"""Shared loading and constants for DICES-350. Read-only; never writes to data/."""
import pandas as pd

CSV_350 = "data/dices-dataset/350/diverse_safety_adversarial_dialog_350.csv"

# The 19 rater IDs the DICES authors removed for failing their data quality checks.
# Copied verbatim from data/dices-dataset/350/README.md, section "Removed raters".
# This is the published filter. No proxy filter is constructed anywhere in this project.
REMOVED_RATERS_350 = [
    '297514565398139', '297515609163939', '297515750682315', '297515617432733',
    '297541515566649', '297541515769980', '297515629971478', '297059995361243',
    '297541522412126', '297540556928761', '297541321453321', '297540562350921',
    '297540983991638', '297060365288109', '297514543980607', '297515729806999',
    '297541271027233', '296709611112092', '296709543131761',
]

# Primary label. Three-way throughout; "Unsure" is a real category, never collapsed
# except in the explicitly-labelled binary appendix.
LABEL_COL = "Q_overall"
LABELS = ["No", "Yes", "Unsure"]          # fixed bin order for every distribution

DEMOGRAPHICS = ["rater_gender", "rater_race", "rater_age", "rater_education"]

# Fixed display order per demographic, so every table and figure matches.
GROUP_ORDER = {
    "rater_gender": ["Woman", "Man"],
    "rater_race": ["Asian/Asian subcontinent", "Black/African American",
                   "LatinX, Latino, Hispanic or Spanish Origin", "Multiracial", "White"],
    "rater_age": ["gen z", "millenial", "gen x+"],
    "rater_education": ["College degree or higher", "High school or below", "Other"],
}

# Short labels for figures.
SHORT = {
    "Asian/Asian subcontinent": "Asian",
    "Black/African American": "Black",
    "LatinX, Latino, Hispanic or Spanish Origin": "Latine/x",
    "Multiracial": "Multiracial",
    "White": "White",
    "College degree or higher": "College+",
    "High school or below": "HS or below",
    "Other": "Other",
    "gen z": "gen z", "millenial": "millennial", "gen x+": "gen x+",
    "Woman": "Woman", "Man": "Man",
}

PRETTY_VAR = {"rater_gender": "Gender", "rater_race": "Race/ethnicity",
              "rater_age": "Age group", "rater_education": "Education"}


def load_350():
    """Load DICES-350. Drops `phase` (constant 'Phase3', carries no information).
    Adds boolean `removed`. Asserts the shape we verified at Checkpoint 1."""
    df = pd.read_csv(CSV_350, dtype={"rater_id": str})
    assert df.shape == (43050, 41), f"unexpected shape {df.shape}"
    assert df.isna().sum().sum() == 0, "unexpected nulls"
    assert df["phase"].nunique() == 1, "phase is no longer constant; do not drop it"
    df = df.drop(columns=["phase"])

    rm = set(REMOVED_RATERS_350)
    assert rm <= set(df.rater_id.unique()), "removed-rater id not present in data"
    df["removed"] = df.rater_id.isin(rm)

    assert df.item_id.nunique() == 350
    assert df.rater_id.nunique() == 123
    assert df.duplicated(["rater_id", "item_id"]).sum() == 0
    assert set(df[LABEL_COL].unique()) == set(LABELS)
    return df


def rater_table(df):
    """One row per rater, with demographics and the removed flag."""
    r = df.drop_duplicates("rater_id")[
        ["rater_id", "removed"] + DEMOGRAPHICS + ["rater_raw_race"]].reset_index(drop=True)
    assert len(r) == 123
    assert r.removed.sum() == 19, f"expected 19 removed raters, got {r.removed.sum()}"
    return r
