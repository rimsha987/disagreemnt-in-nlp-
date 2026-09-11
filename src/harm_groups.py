"""Canonical harm_type grouping scheme for DICES-350.

FIXED 2026-08-26, BEFORE any RQ result was computed. Documented in
results/harm_type_grouping.md. Do not edit after Phase 2 results exist; if a change
is needed, add a new named scheme alongside this one and report both.

The raw `harm_type` field holds 71 distinct strings, which are comma-joined
combinations of 25 atomic tags. 268/350 items carry one tag, 78 carry two, 4 carry
three.

PRIMARY ASSIGNMENT RULE: an item's topic is the group of its FIRST comma-separated
tag. The dataset's own tag order is used because it varies between items in a way
that implies it is meaningful, not canonical: both "Racial,Political" (6 items) and
"Political,Racial" (2 items) occur, as do "Misinformation,Political" (4) and
"Political,Misinformation" (4). Taking the first tag therefore uses the annotators'
own ordering and requires no judgement call from us. Every item lands in exactly one
FINE group and exactly one COARSE group.

ROBUSTNESS RULE (multi-label): an item counts once in every group any of its tags
map to. Group sizes then sum to more than 350. Used only as a robustness check.
"""

# 25 atomic tags -> 10 fine groups. Every atomic tag appears exactly once.
FINE_GROUPS = {
    "Race & ethnicity":            ["Racial"],
    "Gender & sexuality":          ["Gendered & Sexist", "LGBTQ+ & Homophobic"],
    "Other identity & bigotry":    ["Bigoted", "Religious", "National/regional",
                                    "Identity", "Ageism", "Ablist"],
    "Politics & ideology":         ["Political", "Abortion", "Environment/climate"],
    "Misinformation":              ["Misinformation"],
    "Health & substances":         ["Health", "Mental health/self harm", "Drugs/alcohol"],
    "Crime, law & regulated goods": ["Legal", "Criminal/carceral", "Regulated goods"],
    "Violence & sexual content":   ["Violent/Gory", "Aggressive", "Sexual"],
    "Personal & financial":        ["Personal", "Wealth/Finance"],
    "Miscellaneous":               ["Miscellaneous"],
}

# 10 fine groups -> 4 coarse groups, for cells that survive crossing with degree_of_harm.
COARSE_GROUPS = {
    "Identity-based harm":     ["Race & ethnicity", "Gender & sexuality",
                                "Other identity & bigotry"],
    "Politics & misinformation": ["Politics & ideology", "Misinformation"],
    "Health, safety & legality": ["Health & substances", "Crime, law & regulated goods",
                                  "Violence & sexual content"],
    "Other":                   ["Personal & financial", "Miscellaneous"],
}

# Reporting rule fixed in advance: any cell with fewer than this many items is
# reported with its N but not interpreted.
MIN_CELL_N = 20

# --- derived lookups -------------------------------------------------------------

TAG_TO_FINE = {tag: g for g, tags in FINE_GROUPS.items() for tag in tags}
FINE_TO_COARSE = {f: c for c, fines in COARSE_GROUPS.items() for f in fines}
TAG_TO_COARSE = {tag: FINE_TO_COARSE[f] for tag, f in TAG_TO_FINE.items()}

FINE_ORDER = list(FINE_GROUPS)
COARSE_ORDER = list(COARSE_GROUPS)


def atomic_tags(harm_type: str) -> list:
    """Split a raw harm_type string into its atomic tags, in dataset order."""
    return [t.strip() for t in str(harm_type).split(",") if t.strip()]


def primary_tag(harm_type: str) -> str:
    """The first-listed atomic tag."""
    return atomic_tags(harm_type)[0]


def fine_group(harm_type: str) -> str:
    """Fine topic group, via the first-listed tag. Raises on an unmapped tag."""
    t = primary_tag(harm_type)
    if t not in TAG_TO_FINE:
        raise KeyError(f"unmapped harm_type tag: {t!r}")
    return TAG_TO_FINE[t]


def coarse_group(harm_type: str) -> str:
    """Coarse topic group, via the first-listed tag."""
    return FINE_TO_COARSE[fine_group(harm_type)]


def all_fine_groups(harm_type: str) -> list:
    """Multi-label robustness variant: every fine group this item touches."""
    seen, out = set(), []
    for t in atomic_tags(harm_type):
        if t not in TAG_TO_FINE:
            raise KeyError(f"unmapped harm_type tag: {t!r}")
        g = TAG_TO_FINE[t]
        if g not in seen:
            seen.add(g)
            out.append(g)
    return out
