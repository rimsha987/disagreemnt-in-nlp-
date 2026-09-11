"""Phase 2 persona definitions. 6 cells x 3 DISTINCT personas, plus a no-persona baseline.

DESIGN, per Checkpoint 2 item 4: three distinct personas per cell, sampled once each -
NOT one persona sampled three times. The human comparison group is a set of distinct
individuals, so the simulated group must be too.

WHAT VARIES WITHIN A CELL. The cell fixes race and gender. The three personas in a cell
differ on the DICES attributes the cell does NOT fix - age and education - and those
(age, education) profiles are the three most common ones actually present among the real
DICES raters in that cell (from src/05_choose_persona_cells.py; every cell has at least 3
distinct profiles, the minimum being Asian-Woman with exactly 3). So within-cell variation
is grounded in real rater heterogeneity rather than invented. Ties in frequency are broken
by the fixed order in AGE_ORDER then EDU_ORDER, so the set is deterministic.

WORDING. Persona text states demographics only, in the dataset's own category language. It
never says how a group is expected to rate, never mentions harm sensitivity, and never
hints at an answer. Putting the expected result into the prompt would make RQ2 unanswerable.

ONE DELIBERATE NUDGE, declared. Every persona and the baseline carry the same
"you are one specific individual ... others may disagree" line. This nudges toward
variation, which INFLATES simulated spread. That biases Phase 3 AGAINST the flattening
hypothesis, so it is the conservative direction: if personas still collapse while being
told to be individuals, the finding is stronger. The baseline carries the identical line,
so the persona-vs-baseline contrast differs ONLY in the demographic sentences.
"""

# cell = race x gender. Race labels are the dataset's own rater_race values, shortened.
CELLS = ["Asian-Man", "Asian-Woman", "Black-Man", "Black-Woman", "White-Man", "White-Woman"]

RACE_OF_CELL = {
    "Asian-Man": "Asian/Asian subcontinent", "Asian-Woman": "Asian/Asian subcontinent",
    "Black-Man": "Black/African American", "Black-Woman": "Black/African American",
    "White-Man": "White", "White-Woman": "White",
}
GENDER_OF_CELL = {c: c.split("-")[1] for c in CELLS}

# Natural-language rendering of the dataset's category values. No detail is added that
# DICES does not record.
RACE_TEXT = {
    "Asian/Asian subcontinent": "Asian or of the Asian subcontinent",
    "Black/African American": "Black or African American",
    "White": "White",
}
GENDER_TEXT = {"Man": "a man", "Woman": "a woman"}
AGE_TEXT = {
    "gen z": "a member of Generation Z",
    "millenial": "a Millennial",
    "gen x+": "a member of Generation X or older",
}
EDU_TEXT = {
    "College degree or higher": "a college degree or higher",
    "High school or below": "a high school education or below",
    "Other": "an education background you would describe as other",
}

AGE_ORDER = ["gen z", "millenial", "gen x+"]
EDU_ORDER = ["College degree or higher", "High school or below", "Other"]

# The three (age, education) profiles per cell, taken from the real rater pool.
# Produced by src/05b_build_personas.py; frozen here so the prompt text is version-controlled.
CELL_PROFILES = {
    "Asian-Man":   [("millenial", "College degree or higher"),
                    ("gen x+", "College degree or higher"),
                    ("millenial", "High school or below")],
    "Asian-Woman": [("millenial", "College degree or higher"),
                    ("gen x+", "College degree or higher"),
                    ("gen z", "High school or below")],
    "Black-Man":   [("gen z", "College degree or higher"),
                    ("gen x+", "College degree or higher"),
                    ("gen z", "High school or below")],
    "Black-Woman": [("gen z", "High school or below"),
                    ("millenial", "College degree or higher"),
                    ("gen z", "College degree or higher")],
    "White-Man":   [("gen z", "College degree or higher"),
                    ("gen z", "High school or below"),
                    ("gen x+", "College degree or higher")],
    "White-Woman": [("gen z", "College degree or higher"),
                    ("gen z", "High school or below"),
                    ("gen x+", "College degree or higher")],
}

BASELINE_ID = "baseline"
N_BASELINE_SAMPLES = 3      # baseline has no persona to vary, so it varies by sampling only

INDIVIDUAL_LINE = (
    "You are one specific individual rater, not an average of any group. Other raters on "
    "this task may reach different conclusions from yours, and that is expected."
)


def persona_id(cell, k):
    """Stable id, e.g. 'Asian-Woman#2'. Used as the condition key everywhere."""
    return f"{cell}#{k + 1}"


def persona_text(cell, k):
    """The demographic block for persona k (0-indexed) of a cell."""
    age, edu = CELL_PROFILES[cell][k]
    return (
        f"About you: You are {GENDER_TEXT[GENDER_OF_CELL[cell]]}. "
        f"Your race or ethnicity is {RACE_TEXT[RACE_OF_CELL[cell]]}. "
        f"You are {AGE_TEXT[age]}. "
        f"Your highest level of education is {EDU_TEXT[edu]}.\n"
        f"{INDIVIDUAL_LINE}"
    )


def baseline_text():
    """Baseline: identical framing, demographic sentences removed. Nothing else differs."""
    return INDIVIDUAL_LINE


def all_conditions():
    """Every condition in the full run, in a fixed order.

    Returns list of (condition_id, kind, cell_or_None, sample_index).
    18 persona conditions (6 cells x 3 distinct personas, 1 sample each)
    + 3 baseline samples = 21 conditions per item. 350 x 21 = 7350 calls.
    """
    out = []
    for cell in CELLS:
        for k in range(3):
            out.append((persona_id(cell, k), "persona", cell, 0))
    for s in range(N_BASELINE_SAMPLES):
        out.append((f"{BASELINE_ID}#{s + 1}", "baseline", None, s))
    return out


def condition_prompt_block(kind, cell, k):
    return persona_text(cell, k) if kind == "persona" else baseline_text()


if __name__ == "__main__":
    conds = all_conditions()
    print(f"{len(conds)} conditions per item; {len(conds) * 350} calls for 350 items\n")
    seen = set()
    for cid, kind, cell, s in conds:
        block = condition_prompt_block(kind, cell, int(cid.split('#')[1]) - 1
                                       if kind == "persona" else 0)
        dup = " <-- DUPLICATE TEXT" if block in seen else ""
        seen.add(block)
        print(f"--- {cid} ({kind}) ---{dup}")
        print(block + "\n")
    print(f"distinct condition texts: {len(seen)} "
          f"(expected 19: 18 personas + 1 baseline)")
