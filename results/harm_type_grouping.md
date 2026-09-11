# harm_type grouping scheme — FIXED before any result was computed

**Fixed:** 2026-08-26, after Checkpoint 1, before any RQ1/RQ2/RQ3 result existed.
**Machine-readable source of truth:** `src/harm_groups.py`. This document describes it.
**Validation output:** `results/01b_grouping_validation.txt`
**Per-item assignment:** `results/item_harm_groups.csv`

This scheme is used in **Phase 2 only** (the error breakdown by topic). Phase 1 does not
touch `harm_type` at all, so nothing in Phase 1 depends on it.

## The problem

`harm_type` holds **71 distinct strings**, which are comma-joined combinations of **25 atomic
tags**. Of 350 items: 268 carry one tag, 78 carry two, 4 carry three. Used raw, the modal cell
holds a single item, so no breakdown is possible.

## Assignment rule (primary)

> **An item's topic is the group of its FIRST comma-separated tag.**

The dataset's own tag order is used, because it varies between items in a way that implies it
is meaningful rather than canonical: both `Racial,Political` (6 items) and `Political,Racial`
(2 items) occur, as do `Misinformation,Political` (4) and `Political,Misinformation` (4). An
alphabetical or fixed-priority convention would not produce both orderings. Taking the first
tag therefore uses the annotators' own ordering and requires **no judgement call from us** —
which is the property that matters for a scheme that must be fixed in advance.

Every item lands in exactly one fine group and exactly one coarse group.

## Robustness rule (secondary)

> **Multi-label:** an item counts once in every group any of its tags map to.

Group sizes then sum to 421 rather than 350. Reported as a robustness check only. First-tag
and multi-label assignment differ for **68 / 350 items (19.4%)** — that is, 68 items carry
tags spanning more than one fine group.

## Level 1 — fine groups (10)

Every one of the 25 atomic tags appears in exactly one group. Verified: no tag missing, no
tag duplicated.

| Fine group | Atomic tags | Items | % |
|---|---|---|---|
| Race & ethnicity | Racial | 93 | 26.6 |
| Gender & sexuality | Gendered & Sexist, LGBTQ+ & Homophobic | 57 | 16.3 |
| Politics & ideology | Political, Abortion, Environment/climate | 53 | 15.1 |
| Other identity & bigotry | Bigoted, Religious, National/regional, Identity, Ageism, Ablist | 36 | 10.3 |
| Health & substances | Health, Mental health/self harm, Drugs/alcohol | 32 | 9.1 |
| Crime, law & regulated goods | Legal, Criminal/carceral, Regulated goods | 20 | 5.7 |
| Personal & financial | Personal, Wealth/Finance | 18 | 5.1 |
| Misinformation | Misinformation | 17 | 4.9 |
| Violence & sexual content | Violent/Gory, Aggressive, Sexual | 17 | 4.9 |
| Miscellaneous | Miscellaneous | 7 | 2.0 |
| **Total** | | **350** | |

## Level 2 — coarse groups (4)

| Coarse group | Fine groups | Items | % |
|---|---|---|---|
| Identity-based harm | Race & ethnicity, Gender & sexuality, Other identity & bigotry | 186 | 53.1 |
| Politics & misinformation | Politics & ideology, Misinformation | 70 | 20.0 |
| Health, safety & legality | Health & substances; Crime, law & regulated goods; Violence & sexual content | 69 | 19.7 |
| Other | Personal & financial, Miscellaneous | 25 | 7.1 |
| **Total** | | **350** | |

## Reporting rule, also fixed in advance

**`MIN_CELL_N = 20`.** Any cell with fewer than 20 items is reported **with its N** but is not
interpreted and carries no claim.

Consequences, known now:

- Four fine groups fall below the threshold: Misinformation (17), Violence & sexual content
  (17), Personal & financial (18), Miscellaneous (7). Descriptive reporting only.
- All four **coarse** groups clear it (25 to 186).
- **Coarse topic × `degree_of_harm` gives 10 of 16 cells below threshold.** So the Phase 2
  breakdown reports topic and harm-degree as **two separate marginal breakdowns**, not as a
  crossed table. The crossed table can be printed for completeness with all Ns visible, but no
  claim will rest on it.

| Coarse group | Benign | Debatable | Moderate | Extreme | All |
|---|---|---|---|---|---|
| Identity-based harm | 9 | 5 | 42 | 130 | 186 |
| Politics & misinformation | 13 | 6 | 23 | 28 | 70 |
| Health, safety & legality | 6 | 8 | 20 | 35 | 69 |
| Other | 3 | 3 | 13 | 6 | 25 |

## What this scheme cannot do

- It does not recover the annotators' intent for multi-tag items beyond trusting tag order.
- `Miscellaneous` (7 items) is a residual and is never interpreted.
- The topic distribution is heavily skewed toward identity-based harm (53%), so any Phase 2
  topic comparison has far more power on identity items than on anything else. This will be
  stated wherever a topic comparison is reported.
