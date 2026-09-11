# FINDINGS

Running record of every result. Numbers here are produced by scripts in `src/`; nothing is
estimated or filled in by hand. Where something could not be computed, it says so.

---

## Phase 0 — Setup and data verification

**Data source.** `data/dices-dataset/`, shallow clone of
`https://github.com/google-research-datasets/dices-dataset` (CC BY 4.0). Never modified.
Primary file: `data/dices-dataset/350/diverse_safety_adversarial_dialog_350.csv`.

**Scripts:** `src/00_inspect_schema.py`, `src/00b_extra_checks.py`
**Raw output:** `results/00_schema_report.txt`, `results/00b_extra_checks.txt`
**Schema table:** `results/00_schema_350.csv`

### Structure

DICES-350 is **one wide CSV**, not separate rating and rater files. Each row is one
(rater, conversation) pair, with the rater's demographics repeated on every one of their rows.
No join is required anywhere in Phase 1, so the silent-row-loss risk flagged in the brief does
not arise for the human-side analysis.

| Quantity | Value |
|---|---|
| Rows | 43,050 |
| Columns | 41 |
| Conversations (`item_id`) | 350 |
| Raters (`rater_id`) | 123 |
| Ratings per conversation | min 123, median 123, max 123 |
| Items per rater | min 350, median 350, max 350 |
| Duplicate (rater, item) pairs | 0 |
| Missing values | 0, in every column |

The design is **completely crossed and perfectly balanced**: 123 × 350 = 43,050. Every rater
rated every conversation. This is the strongest possible footing for distribution comparisons —
every per-item distribution rests on the same 123 raters, so differences between items are not
confounded with differences in who rated them.

### Column identification

| Role | Column | Confidence |
|---|---|---|
| Item ID | `item_id` (350 distinct, int) | Certain — named, and README confirms |
| Rater ID | `rater_id` (123 distinct, 15-digit) | Certain |
| Demographics | `rater_gender`, `rater_race`, `rater_age`, `rater_education`, `rater_raw_race` | Certain |
| Primary safety label | `Q_overall` — "Yes" / "No" / "Unsure" | Certain — README field 41, aggregate of Q2–Q6 |
| Secondary labels | `Q2_harmful_content_overall`, `Q3_bias_overall`, `Q6_policy_guidelines_overall`, plus 20 granular Q2–Q6 items | Certain |
| Expert gold | `safety_gold` ("Yes"/"No"), `safety_gold_reason` | Certain |
| Item metadata | `degree_of_harm` (4 levels), `harm_type` (71 distinct) | Certain |
| **Quality flag** | **Does not exist as a column** | See below |

### The quality flag — important

**There is no quality-flag column in DICES-350.** No column name contains qual/flag/removed/
excluded/valid/attention/trap/check. What the dataset provides instead is a **hardcoded list of
19 rater IDs in `350/README.md`**, under "Removed raters", described as the raters removed for
failing the authors' data quality checks.

Verified against the data:

- All 19 IDs are present in the CSV. None is missing or malformed.
- 123 − 19 = **104 raters kept**, which matches the DICES README's statement that published
  analyses "typically deal with 104 unique raters."
- Applying the list removes **6,650 of 43,050 rows (15.4%)**, and exactly 19 ratings from every
  one of the 350 conversations (123 → 104 per item).

So filtering is reproducible exactly as the DICES authors did it, but it is a **fixed list, not
a criterion**. Consequences for the paper, stated plainly:

1. The list is a constant copied from the README into `src/00_inspect_schema.py` as
   `REMOVED_RATERS_350`. It is not a derived quantity.
2. We cannot recover *why* each rater was removed, nor vary the filtering threshold. The
   published reason is only "failing data quality checks", with no per-rater detail and no
   underlying score. RQ1 therefore tests the effect of *this specific published filter*, not of
   quality filtering as an adjustable knob.
3. Because DICES-350 is fully crossed, this filter is unusually clean: it removes the same 19
   raters from every item, so per-item rating counts stay uniform (104 everywhere). No item is
   differentially thinned.

This is a limitation to state in the paper, not a blocker. If a graded, tunable filter is wanted
as a robustness check, plausible proxies exist in the data (`answer_time_ms`,
`Q1_whole_conversation_evaluation` ≠ "None of the above") — but per the brief I have **not**
constructed one and am not treating anything as a proxy without approval.

### Rater pool (n = 123)

| Gender | n | % |
|---|---|---|
| Woman | 62 | 50.4 |
| Man | 61 | 49.6 |

| Race/ethnicity | n | % |
|---|---|---|
| White | 30 | 24.4 |
| Black/African American | 29 | 23.6 |
| Asian/Asian subcontinent | 26 | 21.1 |
| LatinX, Latino, Hispanic or Spanish Origin | 22 | 17.9 |
| Multiracial | 16 | 13.0 |

| Age | n | % |
|---|---|---|
| gen z | 56 | 45.5 |
| millenial | 36 | 29.3 |
| gen x+ | 31 | 25.2 |

| Education | n | % |
|---|---|---|
| College degree or higher | 75 | 61.0 |
| High school or below | 41 | 33.3 |
| Other | 7 | 5.7 |

`phase` is "Phase3" for all 123 raters — the column carries no information in DICES-350.

Gender is near-perfectly balanced. Race is roughly balanced by design. Age is not — gen z is
nearly half the pool. Education was not a recruitment variable and the "Other" cell (n = 7) is
too small to support a per-group claim; Ns are reported everywhere.

### Item pool (n = 350)

- `safety_gold`: exactly 175 "Yes" / 175 "No" — balanced by construction.
- `degree_of_harm`: Extreme 199 (56.9%), Moderate 98 (28.0%), Benign 31 (8.9%), Debatable 22 (6.3%).
- `harm_type`: 71 distinct strings, but many are comma-joined multi-labels
  (e.g. "Racial,Political"). Most common single labels: Racial (80 items), Political (33),
  Gendered & Sexist (29). For the Phase 2 breakdown these need grouping into a smaller topic set
  — the 71 raw values are too thin per cell.
- `degree_of_harm`, `harm_type`, `safety_gold`, `context`, `response` are each constant within an
  `item_id` (verified: max 1 distinct value per item).

### Rating distribution, `Q_overall`, all 43,050 ratings

No 61.1% · Yes 32.7% · Unsure 6.3%

The "Unsure" option is real and non-trivial at 6.3%. It is a **third category, not missing
data**, and will be kept as its own bin in all soft-label distributions — collapsing it would
itself be a form of flattening, which is the thing this paper is about.

Raw "unsafe" (Yes) rate by group, all 123 raters, no filtering applied yet:

| Group | n raters | % Yes |
|---|---|---|
| Man | 61 | 31.2 |
| Woman | 62 | 34.2 |
| White | 30 | 27.4 |
| Black/African American | 29 | 29.4 |
| LatinX, Latino, Hispanic | 22 | 35.5 |
| Asian/Asian subcontinent | 26 | 36.6 |
| Multiracial | 16 | 38.2 |
| gen z | 56 | 31.8 |
| millenial | 36 | 32.2 |
| gen x+ | 31 | 34.9 |

Spread across race groups (27.4 → 38.2, ~11 points) is larger than across gender (~3 points) or
age (~3 points). Descriptive only — no test has been run, and these are marginal rates that
ignore item difficulty.

### Two small discrepancies between README and data

Both harmless; recorded so nobody trips over them later.

1. README field 37 is named `Q6_policy_guidelines_other_type`; the actual column is
   `Q6_policy_guidelines_other`.
2. README describes `rater_education` as two categories; the data has **three** — there is a
   third value, "Other" (7 raters).

### Conversation text

271 unique `context` strings and 344 unique `response` strings across 350 items, but all
**350 (context, response) pairs are unique** — no duplicate conversations. Contexts are reused
with different final chatbot responses. Useful for Phase 2: shared context prefixes are a
natural fit for prompt caching.

---

## Phase 1 — The filtering analysis (RQ1)

No API calls. Decisions carried in from Checkpoint 2: `Q_overall` stays 3-way; the filter is
the 19 published rater IDs, no proxy; Fisher's exact throughout; `phase` dropped;
`rater_raw_race` reserved for robustness.

**Scripts:** `src/02_filtering_removal.py`, `src/03_filtering_labels.py`,
`src/03b_direction_and_metric_checks.py`, `src/04_removed_rater_profile.py`
**Raw output:** `results/02_removal_report.txt`, `results/03_labels_report.txt`,
`results/03b_report.txt`, `results/04_profile_report.txt`
**Tables:** `results/02_removal_by_group.csv`, `results/02_removal_tests.csv`,
`results/03_item_jsd.csv`, `results/03_entropy_bins.csv`, `results/03b_direction_tests.csv`,
`results/04_rater_profile.csv`, `results/04_profile_tests.csv`
**Worked example:** `results/03_worked_example.txt`

### Headline

The published filter is **demographically biased but distributionally inert**. It removes men
at 3.4× the rate of women, and that is a real, statistically supported effect. But the labels
it produces are indistinguishable from removing 19 raters chosen at random. Both halves of
that sentence are load-bearing, and they point in opposite directions from what the RQ1
framing anticipated.

---

### 1.1 Verification against the DICES paper (Checkpoint 2, item 3)

All 19 cells match the required counts exactly. Every column sums to 19. Nothing was adjusted.

| Variable | Group | Pool | Removed | Target | Kept | Target kept | Status |
|---|---|---|---|---|---|---|---|
| Gender | Woman | 62 | 5 | 5 | 57 | 57 | MATCH |
| Gender | Man | 61 | 14 | 14 | 47 | 47 | MATCH |
| Age | gen z | 56 | 7 | 7 | 49 | 49 | MATCH |
| Age | millennial | 36 | 8 | 8 | 28 | 28 | MATCH |
| Age | gen x+ | 31 | 4 | 4 | 27 | 27 | MATCH |
| Race | Asian | 26 | 5 | 5 | 21 | 21 | MATCH |
| Race | Black | 29 | 6 | 6 | 23 | 23 | MATCH |
| Race | Latine/x | 22 | 0 | 0 | 22 | 22 | MATCH |
| Race | Multiracial | 16 | 3 | 3 | 13 | 13 | MATCH |
| Race | White | 30 | 5 | 5 | 25 | 25 | MATCH |

The filter removes exactly 19 of the 123 ratings on **every one of the 350 items**, so
per-item rating counts stay uniform at 104. 6,650 of 43,050 rows (15.4%) are dropped.

### 1.2 Who gets removed (RQ1a)

Removal rates with exact Clopper-Pearson 95% intervals. N is printed for every percentage.

| Variable | Group | n pool | n removed | rate | 95% CI |
|---|---|---|---|---|---|
| Gender | Woman | 62 | 5 | 8.1% | [2.7, 17.8] |
| Gender | Man | 61 | 14 | **23.0%** | [13.2, 35.5] |
| Race | Asian | 26 | 5 | 19.2% | [6.6, 39.4] |
| Race | Black | 29 | 6 | 20.7% | [8.0, 39.7] |
| Race | Latine/x | 22 | 0 | 0.0% | [0.0, 15.4] |
| Race | Multiracial | 16 | 3 | 18.8% | [4.0, 45.6] |
| Race | White | 30 | 5 | 16.7% | [5.6, 34.7] |
| Age | gen z | 56 | 7 | 12.5% | [5.2, 24.1] |
| Age | millennial | 36 | 8 | 22.2% | [10.1, 39.2] |
| Age | gen x+ | 31 | 4 | 12.9% | [3.6, 29.8] |
| Education | College+ | 75 | 11 | 14.7% | [7.6, 24.7] |
| Education | HS or below | 41 | 8 | 19.5% | [8.8, 34.9] |
| Education | Other | 7 | 0 | 0.0% | [0.0, 41.0] |
| **All** | | 123 | 19 | 15.4% | [9.6, 23.1] |

**Tests.** Omnibus is the Freeman-Halton exact test for the 2×C table, computed by full
enumeration over fixed margins (implemented in `src/02_filtering_removal.py`; scipy's
`fisher_exact` handles 2×2 only). Per-group tests are 2×2 Fisher's exact, group versus all
other raters, with conditional-MLE odds ratios.

| Variable | Omnibus exact p | Cramér's V | Verdict |
|---|---|---|---|
| **Gender** | **0.0262** | 0.206 | Removal is not independent of gender |
| Race/ethnicity | 0.1667 | 0.203 | No omnibus effect |
| Age group | 0.4523 | 0.121 | No effect |
| Education | 0.5907 | 0.122 | No effect |

| Variable | Group | removed/kept | OR | 95% CI | p |
|---|---|---|---|---|---|
| Gender | Woman | 5/57 | 0.30 | [0.08, 0.95] | 0.0262 |
| Gender | **Man** | 14/47 | **3.36** | [1.05, 12.83] | **0.0262** |
| Race | Asian | 5/21 | 1.41 | [0.36, 4.76] | 0.5487 |
| Race | Black | 6/23 | 1.62 | [0.45, 5.21] | 0.3859 |
| Race | Latine/x | 0/22 | 0 | [0.00, 0.89] | 0.0232 |
| Race | Multiracial | 3/13 | 1.31 | [0.22, 5.57] | 0.7127 |
| Race | White | 5/25 | 1.13 | [0.29, 3.75] | 0.7792 |
| Age | gen z | 7/49 | 0.66 | [0.20, 1.98] | 0.4609 |
| Age | millennial | 8/28 | 1.96 | [0.62, 6.01] | 0.2715 |
| Age | gen x+ | 4/27 | 0.76 | [0.17, 2.68] | 0.7792 |
| Education | College+ | 11/64 | 0.86 | [0.29, 2.69] | 0.8016 |
| Education | HS or below | 8/33 | 1.56 | [0.49, 4.73] | 0.4310 |
| Education | Other | 0/7 | 0 | [0.00, 3.87] | 0.5941 |

**Gender is the finding.** Men are removed at 23.0% against women's 8.1%, OR 3.36. The
interval [1.05, 12.83] only barely excludes 1, so the effect is real but its size is poorly
determined — anywhere from "slightly more" to "twelve times more" is consistent with the data.

Descriptively, no Latine/x rater was removed (0 of 22); the nominal p = 0.0232 for that cell
does not survive correction for the five per-group tests on that table (0.0232 × 5 = 0.116) and
the omnibus race test is null, so this is reported as a description and claimed as nothing.

**Power.** 123 raters, 19 removals, smallest race cell 16. Group intervals run roughly ±15
points wide. These tests are underpowered by construction. No null here is evidence of absence.

*Figure: `figures/fig1_removal_rate_by_group.{pdf,png}`*

### 1.3 How much do labels move (RQ1b)

Per item, the soft label distribution over (No, Yes, Unsure) from 123 raters versus 104.
**TVD is the primary metric** (Checkpoint 2 decision, reasons in §1.5); JSD is reported
alongside as secondary. TVD = ½Σ|p−q|, range [0, 1]. JSD is the **divergence**, base-2 logs,
range [0, 1]. Entropy is base 2 over 3 categories, max log2(3) = 1.585.

| Statistic | TVD (primary) | JSD (secondary) |
|---|---|---|
| mean | **0.01556** | 0.00055 |
| sd | 0.00918 | 0.00081 |
| median | 0.01403 | 0.00035 |
| 95th pct | 0.03308 | 0.00160 |
| max | 0.04339 (item 143) | 0.00871 (item 140) |

Spearman ρ between the two series across items is +0.837, so they broadly agree on ranking;
where they disagree is §1.5.

TVD reads directly as probability mass: **the filtered and unfiltered distributions differ by
1.56 percentage points of mass on average**, and by at most 4.3 points on any single item.
73% of items are below 0.02. As predicted at Checkpoint 2, the shift is small. The arithmetic
was checked by hand on the maximum-JSD item (`results/03_worked_example.txt`) and agrees to
six decimal places.

**Majority label changes: 9 of 350 (2.6%).** Eight move toward Yes, one away.

| item | majority all → kept | JSD | entropy | counts all (No/Yes/Unsure) | counts kept |
|---|---|---|---|---|---|
| 116 | No → Yes | 0.0015 | 1.4128 | 55/53/15 | 42/49/13 |
| 161 | No → Yes | 0.0012 | 1.3012 | 61/53/9 | 48/49/7 |
| 196 | No → tie | 0.0006 | 1.3433 | 59/53/11 | 47/47/10 |
| 204 | tie → Yes | 0.0004 | 1.3451 | 56/56/11 | 45/49/10 |
| 231 | No → Yes | 0.0012 | 1.3949 | 58/51/14 | 45/47/12 |
| 244 | No → Yes | 0.0005 | 1.2816 | 59/56/8 | 48/50/6 |
| 254 | No → tie | 0.0001 | 1.3970 | 56/53/14 | 46/46/12 |
| 267 | Yes → tie | 0.0004 | 1.2820 | 57/58/8 | 48/48/8 |
| 330 | No → Yes | 0.0006 | 1.3254 | 57/56/10 | 48/49/7 |

Ties are counted explicitly, not silently broken: 2 items are tied under the full pool, 4
under the filtered pool. **The flip count depends on this choice** — resolving ties by taking
the first label alphabetically gives 7 rather than 9. Both rules are reported; 9 is the
tie-aware number used throughout.

Every one of the nine sits in the **high-disagreement tertile**. Filtering changes decisions
only where the pool was already split near 50/50, which is exactly where a decision is
fragile to losing any 19 raters.

### 1.4 The null that changes the reading

Mean JSD of 0.00055 is small — but small compared to what? Removing 19 of 123 raters cannot
move a distribution much no matter who those 19 are. So the filter was compared against
removing 19 raters **at random**, 2,000 draws for magnitude and 10,000 for direction, seed
20260826.

| Statistic | Observed | Null mean | Null sd | Null 95% range | z | p |
|---|---|---|---|---|---|---|
| mean JSD | 0.000550 | 0.000575 | 0.000155 | [0.00040, 0.00099] | −0.16 | 0.4495 |
| mean signed ΔP(Yes) | +0.004519 | +0.000003 | 0.007660 | [−0.0156, +0.0144] | +0.59 | 0.5635 |
| mean \|ΔP(Yes)\| | 0.011873 | 0.012961 | 0.002143 | [0.0104, 0.0188] | −0.51 | 0.5577 |
| majority flips | 9 | 7.40 | 2.26 | [3, 12] | +0.71 | 0.4764 |

**On every measure, the published filter is inside the range produced by removing 19 arbitrary
raters.** Its distributional footprint is not distinguishable from arbitrary thinning of the
pool.

**Limitation, for the paper.** No item-level significance test is reported for this shift:
because the same 19 raters are removed from every conversation there is one independent draw
of a removal set rather than 350, so any test treating the items as independent observations
is pseudo-replicated, and all inference here comes from the rater-level permutation test above.

**Direction, descriptively.** P(Yes) rises: pool-level 0.32669 → 0.33121, mean per-item
+0.0045 (+0.45 points). The mechanism is visible: removed raters have P(Yes) = 0.302 against
kept raters' 0.331, and the filter drops 14/61 men versus 5/62 women while men rate Yes at
0.312 against women's 0.342. So filtering tilts the pool toward the group that sees more harm.
The direction is coherent and explainable. Its magnitude is not distinguishable from noise.

*Figures: `figures/fig2_jsd_histogram.{pdf,png}`, `figures/fig2b_jsd_by_entropy.{pdf,png}`*

### 1.5 Contested items — and a metric problem worth reporting

| Tertile | n | entropy range | mean JSD | mean TVD | mean \|ΔP(Yes)\| | majority flips |
|---|---|---|---|---|---|---|
| low | 118 | 0.275–0.960 | 0.00077 | 0.01281 | 0.00922 | 0 |
| medium | 115 | 0.961–1.207 | 0.00044 | 0.01580 | 0.01211 | 0 |
| high | 117 | 1.209–1.464 | 0.00044 | 0.01810 | 0.01431 | 9 |

**On the primary metric the answer is yes: filtering moves labels more on contested items.**
TVD rises monotonically across the tertiles (Kruskal-Wallis H = 18.78, **p = 8.4e-05**;
Spearman ρ with entropy = +0.266, p = 4.2e-07), as do mean |ΔP(Yes)| and the flip counts,
which are 0 / 0 / 9. **JSD alone dissents**, being flat to slightly decreasing
(H = 1.44, p = 0.487; Spearman ρ = −0.048, p = 0.376). That dissent is what led to the metric
change adopted at Checkpoint 2, and the reason for it is below.

The Spearman/Pearson split on JSD (ρ = −0.048 null, r = −0.256 p = 1.2e-06) was the tell that
something was wrong with the metric rather than the data. **Mechanism, verified not asserted:**
for a small perturbation Δ = q − p, a second-order expansion gives

> JSD ≈ (1 / 8 ln2) · Σ Δᵢ² / pᵢ

so each category's contribution is weighted by **1/pᵢ** — small cells are amplified. On this
data the approximation holds tightly (Pearson r between approximation and actual JSD = 0.931,
mean relative error 5.3%), so the weighting is real. Low-entropy items have small cells, so a
one- or two-vote change there produces a large JSD from a trivial movement in probability.
Item 140 is the clean illustration: its Unsure cell goes 2/123 → 0/104, which makes it the
**largest JSD of all 350 items** despite ranking only 45th under TVD and moving P(Yes) by 1.03
points.

Note the marginal correlation between smallest-cell probability and JSD is *not* significant
(Spearman −0.079, p = 0.142) — because the size of Δ varies too. The 1/pᵢ weighting shows up
in the verified approximation, not in a raw marginal correlation. Stated that way to avoid
overclaiming.

**Consequence, adopted at Checkpoint 2.** TVD is the primary metric for every distributional
comparison in this project, including Phase 2 and Phase 3; JSD is reported alongside as
secondary. JSD is the wrong instrument when categories can be near-empty, which they routinely
are in three-way soft labels with a small "Unsure" mass. This matters more, not less, in
Phase 2: persona-prompted LLM distributions built from three samples per cell will have many
zero cells, exactly the regime where JSD misbehaves. Stated plainly for the paper: this is not
metric-shopping. The 1/p mechanism was derived and empirically verified before the
disagreement between the two metrics was resolved in TVD's favour, and both are reported
everywhere so the reader can see the difference.

*Figure: `figures/fig2c_metric_sensitivity.{pdf,png}`*

### 1.6 Binary-collapse appendix (Checkpoint 2, item 1)

Unsure merged into No, i.e. Yes versus not-Yes. Primary analysis stays 3-way.

| | mean | median | max |
|---|---|---|---|
| binary JSD | 0.00024 | 0.00010 | 0.00495 |
| 3-way JSD | 0.00055 | 0.00035 | 0.00871 |

Ratio of means 0.433; Spearman between the two per-item series +0.616 (p = 5.8e-38). **The
conclusion is unchanged by the collapse.** Magnitudes fall because the binary version is blind
to movement between No and Unsure, which is a meaningful share of what filtering actually
does — 6.3% of all ratings are Unsure. That is itself an argument for keeping three categories.

### 1.7 Do the removed raters look like low-quality raters? (Checkpoint 2, item 8)

Descriptive only. No threshold fitted, nothing here feeds another analysis. Per-rater values
computed over each rater's 350 ratings.

| Metric | Removed median (n=19) | Kept median (n=104) | MWU p | CLES |
|---|---|---|---|---|
| median answer time (ms) | **41,118** | **123,360** | 0.0013 | 0.267 |
| modal-label share | 0.7143 | 0.6343 | 0.0194 | 0.669 |
| label entropy (bits) | 0.8915 | 1.0467 | 0.0189 | 0.330 |
| all 20 sub-answers identical | 0.6914 | 0.6143 | 0.2922 | 0.576 |
| agreement with pool majority | 0.7657 | 0.7271 | 0.2986 | 0.575 |
| Q1 non-comprehension rate | 0.0029 | 0.0029 | 0.5510 | 0.459 |

p-values are descriptive: seven metrics on the same 123 raters, no correction, and the removal
list was not defined by any of these quantities.

**The removed group is bimodal, and this is the clearest structural result in Phase 1.**

- **Eleven fast raters.** Median answer times 33.1k–44.3k ms, all *below the kept pool's
  minimum* of 45,162 ms. There is a clean gap: no kept rater is as fast as these eleven.
- **A slower cluster that straight-lines.** Rater ...806999 gave the identical answer to all
  350 conversations and all 20 sub-questions (modal share 1.000, label entropy 0.000, flat
  share 1.000). Raters ...131761, ...769980, ...971478, ...991638 are close behind at
  0.97–0.99 modal share.
- Two raters (...991638, ...769980) are constant on `Q_overall` while varying their
  sub-answers, and agree with the pool majority only 22.9% and 23.1% of the time, against a
  kept-pool minimum of 32.0%. Deviant in a third way again.

Counting against the kept pool's 5th/95th percentiles: 12/19 below p5 on speed, 7/19 above p95
on modal share, 7/19 below p5 on entropy, 5/19 above p95 on granular flatness, 3/19 below p5 on
majority agreement, 1/19 above p95 on Q1 non-comprehension. **18 of 19 are extreme on at least
one signal; 1 of 19 is extreme on none.**

This corroborates the published list as a genuine quality filter rather than an arbitrary set.
It cannot *validate* it — the criteria are unpublished and the list is fixed — but the removed
raters do look like speeders and straight-liners, which is what the DICES authors claim.

### 1.8 What Phase 1 means for the argument

RQ1 splits cleanly into two answers that do not agree with each other.

1. **Removal is demographically non-random.** Men are removed 3.4× as often as women
   (p = 0.026). The filter is not neutral with respect to who is in the pool.
2. **On DICES-350 that bias does not reach the labels.** Every distributional measure sits
   inside the random-removal null.

The reconciliation is a design fact, and it should be stated as a scope limit rather than
buried: **DICES-350 has 123 raters per item.** Losing 15% of a 123-rater pool cannot move a
soft label much, whoever is lost. The very replication depth that makes DICES suitable for
distribution comparison also makes it robust to filtering. In a dataset with 3–5 raters per
item — which is the normal case, and the case Fleisig et al. are concerned with — the same
demographic bias would operate on distributions estimated from a handful of votes, and the
label consequences would be far larger.

So the honest Phase 1 claim is narrower and more interesting than "filtering distorts labels":
**the filter's demographic bias is real and its label effect here is bounded by replication
depth.** Whether that bias compresses *between-group* spread — a different quantity from the
pooled distribution measured here, and the one that matters for the flattening thesis — is
Phase 3, and is not answered by anything above.

---

## Phase 1 amendments (Checkpoint 2 decisions, applied retroactively)

1. **TVD is now the primary metric everywhere**, JSD secondary, including Phase 2 and Phase 3.
   `src/03_filtering_labels.py` computes both; every table and figure leads with TVD.
   Figure renamed `fig2_label_shift_histogram` (was `fig2_jsd_histogram`).
2. **The Wilcoxon test is struck.** It no longer appears in any result. §1.4 carries the
   one-sentence limitation explaining why: the same 19 raters are removed from every item, so
   there is one independent draw of a removal set rather than 350, and any item-level test is
   pseudo-replicated. All inference is from the rater-level permutation test.
3. **The Latine/x cell is now one descriptive sentence** in §1.2, with the multiplicity
   correction stated inline.

---

## Phase 2 — Design (RQ2)

Nothing in this section has been executed. It records the design decisions and the artefacts
built for them, so the design is fixed before any response is seen.

**Scripts:** `src/05_choose_persona_cells.py`, `src/05b_build_personas.py`,
`src/06_build_prompts.py`, `src/07_run_pilot.py`, `src/08_pilot_report.py`
**Modules:** `src/personas.py` (the 19 condition texts), `src/prompts.py` (rubric, message
construction, strict parser)
**Raw output:** `results/05_cell_selection.txt`, `results/05b_persona_verification.txt`,
`results/06_pilot_prompts.txt`, `results/06_cost_estimate.txt`, `results/07_dryrun.txt`

### 2.1 The six cells, chosen by measured spread

The cells were not chosen by eye. If the human groups picked barely differ from one another,
RQ2 has nothing to detect no matter how the model behaves, so candidate schemes were scored by
the quantity Phase 3 will measure: mean pairwise TVD between group distributions, averaged over
the 350 items. Because small groups make noisy distributions and noise inflates distance, each
scheme is scored against a permutation null that reassigns raters to groups **preserving cell
sizes**. The reported effect is the excess over that null.

| Scheme | cells | min cell | observed TVD | size-matched null | excess | z |
|---|---|---|---|---|---|---|
| **race3 × gender2 (White/Black/Asian)** | 6 | 10 | 0.20315 | 0.16559 | **+0.03756** | **+3.61** |
| gender2 × age3 | 6 | 14 | 0.16156 | 0.13959 | +0.02197 | +2.78 |
| gender2 | 2 | 61 | 0.09170 | 0.07794 | +0.01377 | +1.47 |
| race3 only | 3 | 26 | 0.12887 | 0.11503 | +0.01384 | +1.23 |
| race5 | 5 | 16 | 0.13835 | 0.12749 | +0.01086 | +1.21 |
| age3 | 3 | 31 | 0.09603 | 0.09949 | −0.00346 | −0.37 |

**Chosen: race3 × gender2** — White / Black / Asian × Man / Woman.

Two things in that table are worth the paper's space. **Age alone scores *below* its null
(z = −0.37): there is no real between-group signal on age at all**, so an age-based scheme
would have asked the model to reproduce differences the humans do not show. And the
intersectional scheme beats both of its own marginals by a wide margin — the gender×race cells
differ far more from each other than gender groups or race groups do separately. That is a
direct empirical argument for intersectional analysis of the kind Homan et al. (2024) argue
for, and it falls out of the cell-selection step rather than being assumed.

| Cell | n raters | n removed by filter | % Yes (all) | % Yes (kept) |
|---|---|---|---|---|
| Asian-Man | 16 | 4 | 29.64 | 31.36 |
| Asian-Woman | 10 | 1 | **47.83** | 42.19 |
| Black-Man | 12 | 5 | 31.83 | 39.18 |
| Black-Woman | 17 | 1 | 27.63 | 28.89 |
| White-Man | 11 | 2 | **18.10** | 17.08 |
| White-Woman | 19 | 3 | 32.78 | 31.27 |

The most distant human pair is **Asian-Woman vs White-Man** (mean pairwise TVD 0.368; 47.8% vs
18.1% Yes, a ~30-point gap far larger than any marginal contrast).

**Limitations, stated up front.** The scheme covers 85 of 123 raters (69%) and **excludes
Latine/x (22) and Multiracial (16)**. Multiracial has the highest %Yes in the whole pool (38.2),
so the six chosen cells span a narrower range than the full pool does; this understates human
diversity in the RQ2 comparison. Both groups remain in the Phase 3 human conditions, which use
all 123 raters. The smallest cell holds 10 raters, so per-item human distributions in that cell
rest on 10 votes; Ns are printed everywhere and no cell-level claim is made without them.

### 2.2 Distinct personas per cell (Checkpoint 2, item 4)

Three **distinct** personas per cell, each sampled **once**, replacing three samples of one
persona. The human comparison group is a set of distinct individuals, so the simulated group is
built the same way.

What varies inside a cell is grounded in the data rather than invented: the cell fixes race and
gender, and the three personas differ on the DICES attributes the cell does *not* fix — age and
education — taking the **three most common (age, education) profiles actually present among the
real raters in that cell**. Every cell has at least three distinct profiles (the minimum is
Asian-Woman with exactly three). `src/05b_build_personas.py` re-derives these from the data and
asserts they match the values frozen in `src/personas.py`; it passes on all six cells.

Condition count is unchanged from the brief's budget: **6 cells × 3 personas = 18 persona
conditions, + 3 baseline samples = 21 per item × 350 items = 7,350 calls.**

**Persona wording states demographics only**, in the dataset's own category language. It never
says how a group is expected to rate and never mentions harm sensitivity — putting the expected
result in the prompt would make RQ2 unanswerable.

**One deliberate nudge, declared.** Every persona *and the baseline* carry the same line:
"You are one specific individual rater, not an average of any group. Other raters on this task
may reach different conclusions from yours, and that is expected." This pushes toward variation
and so **inflates** simulated spread. That biases Phase 3 *against* the flattening hypothesis,
which is the conservative direction: if personas still collapse while being told to be
individuals, the finding is stronger. Because the baseline carries the identical line, the
persona-vs-baseline contrast differs *only* in the demographic sentences.

### 2.3 Prompt structure

```
system  RUBRIC + blank line + PERSONA     (19 distinct persona/baseline texts)
user    CONVERSATION + fixed answer cue   (per item)
```

The rubric frames an annotation task, not a generation task, and is built from the DICES
codebook: the question is `Q_overall`, the options are exactly the dataset's **Yes / No /
Unsure**, and the reasons listed are the Q2–Q6 categories. It states explicitly that the user's
turns are not being rated and that a provoking user message is not itself grounds for an unsafe
rating — without this the model would rate the adversarial *user* rather than the chatbot
response. Output is forced to one word; `prompts.parse_rating` is strict and returns `None`
rather than coercing anything else into a rating.

**Placement is deliberate:** the persona sits in the system prompt, before the conversation.
This study's entire question is whether persona conditioning changes ratings, so the persona
must not be handicapped by placement — otherwise "personas did nothing" is confounded with
"we buried the persona".

**Provider-switch note.** The rubric and persona were originally two separate Anthropic system
blocks with a prompt-cache breakpoint between them. Caching is gone (§2.4), so they are now one
system string. **The text sent is byte-identical** to what the two-block form produced; only the
transport shape changed. Verified by assertion, not by eye. `prompts.parse_rating` is untouched.

### 2.4 Provider switch, and the constraint that actually binds

Phase 2 moved off the Anthropic API to free-tier providers. **Nothing is purchased.** Prompt
caching and all cost analysis were dropped with it: neither target provider exposes an
equivalent explicit-breakpoint cache, and there is no spend to model. The binding constraint is
no longer money. It is **rate limits and wall-clock**.

**Rate limits could only be established for one of the two providers.** Google no longer
publishes free-tier RPM/TPM/RPD figures. As of its 2026-08-18 revision,
`ai.google.dev/gemini-api/docs/rate-limits` says only that limits "depend on a variety of
factors (such as your usage tier) and can be viewed in Google AI Studio", pointing to
`aistudio.google.com/rate-limit` for per-account numbers, and warns that "specified rate limits
are not guaranteed". Groq publishes its limits per model; those are transcribed verbatim.

| Provider | Model | Published limits | Daily ceiling | Days for 7,350 calls |
|---|---|---|---|---|
| Gemini free | `gemini-2.5-flash` | **unverified placeholder** 10 RPM / 250 RPD | 250 (RPD-bound) | **29.4** |
| Groq free | `qwen/qwen3.8-27b` | 30 RPM / 1K RPD / 8K TPM / **2M TPD** | 1,000 (RPD-bound) | **7.3** |
| Groq free | `openai/gpt-oss-120b` | 30 RPM / 1K RPD / 8K TPM / **200K TPD** | 176 (**TPD**-bound) | **41.7** |

At ~1,134 measured-estimate input tokens per call, **the token-per-day cap, not the
request-per-day cap, decides whether a Groq model is usable at all.** `gpt-oss-120b` allows
1,000 requests/day on paper but only ~176 of *these* requests, because 200K TPD divided by 1,134
tokens runs out first. `qwen3.8-27b` is the only Groq model listed with enough TPD (2M) to
sustain the run. That is a selection criterion imposed by quota arithmetic, not by model
quality, and it should be stated as such in the paper.

**The full run does not fit on the Gemini free tier as documented.** 29.4 days against a
15 September deadline, from 26 August, is 20 days available. The Gemini figure rests on an
unverified placeholder, so it may be wrong in either direction — the AI Studio dashboard is the
only way to know. Options, in order of preference:

| Option | Calls | Days | Cost to the design |
|---|---|---|---|
| Groq `qwen3.8-27b`, full design | 7,350 | 7.3 | none |
| Drop to 2 personas per cell (14 conditions) | 4,900 | 4.9 | loses the 3rd within-cell draw |
| Halve the items to 175, stratified | 3,675 | 3.7 | halves per-item precision |
| Gemini, full design | 7,350 | 29.4 | misses the deadline |

Recommendation: **run on Groq `qwen/qwen3.8-27b` and keep the full design.** The user's stated
fallback rule was quality-triggered (refusal rate or format compliance); this is a second,
independent reason to prefer Groq, and it is a throughput reason. If the Gemini dashboard shows
materially higher limits than the placeholder, Gemini becomes viable again and the switch is one
line in `src/phase2_config.py`.

### 2.4b Model identity for the paper's data section

Recorded on **every** response record, so the paper can quote what actually answered rather than
what was requested:

- `provider`, `model_requested` — what was asked for (`gemini-2.5-flash` or `qwen/qwen3.8-27b`)
- `model_version` — the version string the API itself reports (`model_version` on Gemini,
  `model` on Groq)

`src/08_pilot_report.py` prints both and **flags it loudly if more than one version string
appears across a run**, since a provider silently rerouting mid-run would otherwise pool two
models into one condition. The exact version string cannot be reported until the pilot runs;
it is not guessed here.

### 2.5 Pilot composition — changed, and why

**60 calls held fixed. Composition changed from 20 items × 3 personas to 15 items × 4
conditions**, because item 4 created a question the original composition cannot answer.

| Condition | Tests |
|---|---|
| Asian-Woman#1 vs Asian-Woman#2 | **within-cell** — do two distinct personas in the *same* cell differ? Validates the item-4 design itself |
| Asian-Woman#1 vs White-Man#1 | **between-cell** — the most distant human pair (TVD 0.368) |
| Asian-Woman#1 vs baseline#1 | **persona effect** — does demographic text change anything vs none? |

If the within-cell contrast comes back identical on every item, three personas per cell buys
nothing and the cell collapses to one persona — which changes what is worth purchasing for the
full run. The old composition would not have surfaced that until after the money was spent.

Items are stratified: 5 from each unfiltered-entropy tertile (seed 20260826), so the pilot spans
uncontested and contested conversations rather than only easy ones.

**Dry run passed**: all 60 requests build and validate, the persona block is asserted present in
every persona request and absent from every baseline request, and the pilot contains exactly 4
distinct condition texts.

### 2.6 Runner architecture: sequential, rate-limited, resumable

**Provider abstraction.** `src/llm_client.py` exposes one interface:

```python
provider = make_provider("gemini")      # or "groq", "groq-gptoss"
reply = provider.generate(system_text, user_text)
```

The provider receives two flat strings and returns a `Reply`. It never builds prompts, never
knows what a persona is, and never parses a rating. **Swapping providers is a one-line change to
`src/phase2_config.py`**; `src/prompts.py` and `src/personas.py` are untouched by it.

**No Batch API.** Sequential calls, paced by a `RateLimiter` that respects RPM, TPM, RPD and TPD.
Exponential backoff on 429 and 5xx: base 2s doubling to a 120s cap, ±25% jitter so parallel
restarts do not resynchronise, `Retry-After` honoured when the provider sends it, 8 attempts.
Hitting a daily quota raises `DailyQuotaExhausted`, which stops the run **cleanly** so a resume
picks up the next day rather than burning retries against a wall.

**Resumability.** Every response is appended to a JSONL shard in `raw_responses/phase2/`,
keyed by `(item_id, condition)`, then `flush()`ed and `fsync()`ed **before the next call is
made**. On start the runner reads every shard, builds the completed-key set, and skips it.
Each session opens a *new* shard; existing shards are never rewritten, and the runner refuses to
overwrite one. A record counts as complete only if `error` is null, so failures are retried.

Tested rather than asserted, in `src/07b_test_resume.py` against synthetic shards in a temp
directory (never `raw_responses/`). All six cases pass:

| Case | Result |
|---|---|
| completed keys skipped | 30 done → 30 remaining |
| error records retried, not treated as complete | all 5 requeued |
| torn final line from a hard kill tolerated | reader does not raise |
| duplicate keys across shards counted once | 5 duplicates reported |
| records spread over several shards all honoured | 45 done → 15 remaining |
| complete shard set leaves nothing to do | 0 remaining |

**The guarantee, as a number:** worst-case loss on a crash, kill, quota wall, or power cut is
**one call** — the one in flight. A crash at call 5,000 of 7,350 costs call 5,000 only.

`--status` reports progress and the wall-clock projection without calling anything; `--dry-run`
validates every request and writes nothing; `--limit N` caps a session; one Ctrl-C stops cleanly
after the call in flight.

### 2.7 Provider and scope, decided at Checkpoint 3

**Provider: Groq, `qwen/qwen3.8-27b`.** Gemini was dropped. Google does not publish free-tier
RPM/TPM/RPD figures (its docs point to a per-account dashboard), and an undocumented limit is
not a basis for sizing a multi-day run. Groq publishes limits per model, and they are
transcribed into `src/phase2_config.py` with their fetch date.

**Scope: 175 items, all 21 conditions. 3,675 calls, 3.7 days.** Items were cut rather than
personas. Phase 3's metrics are means over items, so 175 still leaves ~58 per entropy tertile,
whereas a within-group spread estimated from 2 personas is not a usable variance estimate. The
cheap dimension was the one cut.

| Blocks | Items | Calls | Days at 1,000/day |
|---|---|---|---|
| 1 (running) | 175 | 3,675 | 3.7 |
| 1 + 2 (if extended) | 350 | 7,350 | 7.3 |

### 2.8 The frozen item sample

`src/09_select_items.py` draws **once** and writes `results/09_item_blocks.csv`. Re-running it
verifies that file and **refuses to redraw**. Every one of the 350 items carries a block number,
so the remaining 175 are a **queued continuation, not a discard**: the store is keyed by
`(item_id, condition)`, so `--blocks 1,2` adds keys and re-runs nothing.

Stratified on entropy tertile × `degree_of_harm`, proportional at 50% of each stratum with
largest-remainder rounding to land on exactly 175. Benign (31) and Debatable (22) are collapsed
into one stratum so no cell is 2–3 items; Extreme (199) and Moderate (98) stand alone.

| Block 1 | Extreme | Moderate | Benign/Debatable | total |
|---|---|---|---|---|
| low entropy | 34 | 14 | 11 | 59 |
| medium | 34 | 15 | 8 | 57 |
| high | 32 | 19 | 8 | 59 |
| **total** | **100** | **48** | **27** | **175** |

Proportions are preserved to within 0.6 points on every dimension (e.g. Extreme 57.1% vs 56.9%
in the full 350; entropy tertiles 33.7 / 32.6 / 33.7).

**Forced inclusion:** the 15 pilot items are forced into block 1. They were drawn earlier, with
a fixed seed, stratified by entropy, and before any response existed, so this introduces no
selection on outcome — and it makes all 60 pilot calls reusable as part of the main run rather
than discarded.

### 2.9 Groq quota facts, and what could not be established

**Asked, and answerable:** rate limits apply **at the organisation level**, not per user or per
key — Groq's docs state "Rate limits apply at the organization, not individual users." The
pilot uses the same organisation and the same model as the full run, so **yes, the pilot's 60
calls consume the same daily 1,000-request bucket.** Budget the run as 3,675 + 60, or reuse the
pilot records, which the frozen item list makes possible.

**Asked, and NOT answerable from documentation:** Groq does not document *when* the daily
buckets reset — no statement about midnight UTC, a rolling window, or an account anniversary.
This is not reported as a guess. Instead the API's own headers are captured on every call:

| Header | What it gives |
|---|---|
| `x-ratelimit-limit-requests` / `-remaining-requests` | RPD budget and what is left |
| `x-ratelimit-reset-requests` | **the observed reset time — the only authority on it** |
| `x-ratelimit-limit-tokens` / `-remaining-tokens` / `-reset-tokens` | TPM budget |
| `retry-after` | only present on a 429; honoured by the backoff |

The runner prints these on the first call of every session and stores them on every record;
`08_pilot_report.py` reports the last observed set. The reset time will therefore be a measured
fact in the pilot report, not an assumption.

### 2.10 Persona sensitivity: a capability check, kept separate from the finding

If the model returns identical ratings for two personas whose human counterparts differ
sharply, that is a fact about **the model**, not a finding about persona-prompting, and
conflating the two would let the paper claim something its design cannot support.
`src/persona_check.py` separates three questions and answers them in order.

**1. Provenance.** Every response record now stores `persona_sent` (the verbatim persona
paragraph), `system_sha256` and `user_sha256`. Without this, "the model ignored the persona"
cannot be distinguished from "the persona was never in the prompt". The report checks that each
condition sent exactly one distinct system text, that persona conditions contain "About you:",
and that the baseline does not.

**2. Sensitivity, against a same-items human reference.** The ~30-point gap quoted earlier is
pool-wide over 350 items; the pilot runs 15. So the reference is recomputed on the pilot's own
items. On those 15, the real humans give:

| Cell | n raters | n ratings | P(Yes) | P(No) | P(Unsure) |
|---|---|---|---|---|---|
| Asian-Woman | 10 | 150 | 0.433 | 0.460 | 0.107 |
| White-Man | 11 | 165 | 0.109 | 0.861 | 0.030 |

**Human gap on the pilot items: 32.4 points** — slightly wider than the pool-wide 29.7, so the
pilot is a fair test of the contrast rather than an easy one.

**3. The verdict.** Three branches, all pre-written so the reading is not decided after seeing
the number:

- **0 items differ** → `*** MODEL CAPABILITY PROBLEM - FLAGGED ***`. States explicitly that the
  paper **cannot** claim "personas fail to reproduce human diversity" from this, because the
  instrument did not respond to the manipulation and there is no measurable treatment. What it
  *can* say is narrower: this model does not condition safety ratings on demographic persona
  text under this prompt. Lists what to try, in order, and says **do not proceed to the
  3,675-call run** until one of them produces variation.
- **1 item differs** → near-degenerate; indistinguishable from sampling noise on 15 items at
  temperature 1.0; treat as a probable capability problem.
- **≥2 items differ** → a measurable treatment effect exists; the full run is worth executing.
  A directional preview against the human gap is printed but marked non-inferential.
- **provenance failed** → `INDETERMINATE`, and no capability claim is made at all.

Both branches were **smoke-tested against synthetic shards** in a temp directory before any real
data existed, so the verdict logic is known to fire correctly rather than assumed to. The
synthetic shards were deleted; `raw_responses/` has never been written to.

The report also flags **baseline degeneracy** separately: a baseline that returns one label on
every item has zero spread by construction and is a degenerate floor for Phase 3, which must be
reported rather than silently used.

### 2.12 Pilot results (executed 2026-08-27)

**Run:** `src/07_run.py --mode pilot`, 60 calls, 7.1 minutes, 8.4 calls/min achieved.
**Raw:** `raw_responses/phase2/pilot_groq_20260827T120802.jsonl` (60 records, append-only)
**Report:** `results/08_pilot_report.txt`, parsed table `results/08_pilot_parsed.csv`

#### Model identity, for the data section

| Field | Value |
|---|---|
| provider | Groq |
| `model_requested` | `qwen/qwen3.8-27b` |
| `model_version` (API-reported) | `qwen/qwen3.8-27b` |
| temperature | 1.0 |
| max output tokens | 8 |

The API returns no dated snapshot suffix, so requested and reported versions are identical.
**There is no pinnable version string**, which means a silent model upgrade would be invisible
in the data. The report checks for more than one version string across a run and flags it.

#### Mechanics: everything passed

| Check | Result |
|---|---|
| Responses returned | 60 / 60 |
| API errors | 0 |
| Retries needed | 0 (max attempts 1) |
| Provider-side blocks or refusals | **0 / 60 (0.0%)** |
| `finish_reason` | `stop` on all 60 |
| Parse success | **60 / 60 (100.0%)** |
| Provenance | **PASS** — one distinct system text per condition, "About you:" present in all 3 persona conditions and absent from the baseline |

Refusals were the main risk given the conversations are adversarial by design. There were none.
The rigid one-word output format held on every call; measured output was 2.05 tokens mean, max 3.

Measured input: **922.9 tokens mean** (min 826, max 1,195), against the 1,134 chars/token
estimate. The estimate was ~23% high, so TPM pacing is less binding than projected.

#### Quota mechanism, measured rather than assumed

Groq does not document a daily reset, and the headers show why: **there is no reset event.**
The first call returned `x-ratelimit-limit-requests: 1000`, `remaining: 999`,
`reset: 1m26.4s`. 86.4 s is exactly 86400/1000 — one request's worth of quota refilling
continuously. Token quota behaves the same way within the minute.

Consequences: there is no midnight cliff to schedule around; a run can burst against a full
bucket and then settles to a steady 1,000 requests/day; and a resumable runner can simply keep
going. **The pilot's 60 calls consume the same organisation-level bucket as the main run**
(Groq: limits "apply at the organization, not individual users"), so the run should be budgeted
as 3,675 + 60, or the pilot records reused — which the forced inclusion of the pilot items in
block 1 makes possible.

#### The persona question: NOT DEMONSTRATED, and not the same as a negative

All six pairwise disagreement rates, 15 items, one sample each:

| Pair | Differ | Rate | What it is |
|---|---|---|---|
| Asian-Woman#1 vs Asian-Woman#2 | **4/15** | 0.27 | **within-cell** (same race+gender) |
| Asian-Woman#1 vs baseline#1 | 3/15 | 0.20 | persona vs no persona |
| Asian-Woman#2 vs baseline#1 | 3/15 | 0.20 | persona vs no persona |
| Asian-Woman#1 vs White-Man#1 | **2/15** | 0.13 | **between-cell** |
| Asian-Woman#2 vs White-Man#1 | **2/15** | 0.13 | **between-cell** |
| White-Man#1 vs baseline#1 | 2/15 | 0.13 | persona vs no persona |

**Within-cell disagreement (4/15) exceeds between-cell (2/15).** Two personas that share race
and gender cannot differ *because of* race or gender, so that 4/15 is a noise floor made of
persona wording plus sampling at temperature 1.0. A demographic effect that does not exceed its
own noise floor has not been shown. Every one of the six rates sits in the same 2–4 band, which
is what a single undifferentiated noise process looks like.

The model is not frozen — it produced Yes, No and Unsure, and the baseline varies across items
(modal share 0.867, so it is an informative floor rather than a degenerate one). The problem is
that the variation present is **not attributable to the manipulation**.

**Correction to the reported verdict.** `src/persona_check.py` originally branched only on the
count of between-cell disagreements and printed "THE MODEL DOES RESPOND to persona
conditioning" on 2/15. That reading was wrong: with one sample per persona it confounds
conditioning with sampling noise. The function now compares between-cell against the within-cell
floor and returns **PERSONA CONDITIONING NOT DEMONSTRATED**. The saved report reflects the
corrected logic.

This is **unresolved, not negative**. 15 items with one sample each cannot separate the two, and
the paper cannot yet claim either that personas work or that they fail.

#### The calibration gap, which is a finding in its own right

On the same 15 items:

| Rater set | P(Yes) | P(No) | P(Unsure) |
|---|---|---|---|
| Humans, all 123 | 0.278 | 0.661 | 0.061 |
| Humans, Asian-Woman cell | 0.433 | — | — |
| Humans, White-Man cell | 0.109 | — | — |
| **Model, all 60 calls** | **0.050** | **0.900** | **0.050** |

**The model is 5.6× less likely to call an item unsafe than the human pool**, and its P(Yes)
of 0.050 sits below even the most permissive human cell (White-Man, 0.109). It answered "No"
on 90% of calls. This is independent of the persona question and does not depend on resolving
it: whatever the model is doing, it is not calibrated to this rater pool.

For RQ3 this matters directly. A condition pinned near-deterministically at "No" has almost no
between-group spread available to it *by construction*, so a low measured spread for the
persona-LLM condition would be partly an artefact of the base rate rather than evidence about
persona-prompting. This has to be stated in Phase 3, not glossed.

#### Where the six non-"No" answers fell

| item | condition | model | human P(Yes) | entropy bin |
|---|---|---|---|---|
| 10 | Asian-Woman#2 | Yes | 0.333 | high |
| 68 | Asian-Woman#1 | Yes | 0.561 | medium |
| 68 | baseline#1 | Unsure | 0.561 | medium |
| 274 | Asian-Woman#2 | Unsure | 0.179 | low |
| 274 | baseline#1 | Yes | 0.179 | low |
| 316 | Asian-Woman#1 | Unsure | 0.228 | medium |

Item 68 is the most human-contested of the fifteen (P(Yes) 0.561) and drew two of the six
deviations, which is mildly encouraging. But item 274 sits in the *low*-entropy tertile
(P(Yes) 0.179) and also drew two. With six deviations there is no pattern to test, only to note.

Across conditions, 11 of 15 items were unanimous, 2 had two distinct ratings, 2 had three.
Unanimity did not vary by human disagreement tertile (4/5, 3/5, 4/5 for low/medium/high), so
the model's variation is not tracking item difficulty as the humans experience it.

#### What has to happen before the 3,675-call run

The decisive test is cheap: **re-run one persona on the same 15 items several times** to measure
the pure sampling noise floor directly, then test whether the between-cell difference exceeds
it. At 5 repeats that is 75 calls, roughly nine minutes, and it converts an unresolved result
into either a usable treatment effect or a clean, defensible negative. Spending 3,675 calls
before knowing which would risk producing a dataset whose central comparison cannot be
interpreted.

### 2.13 Diagnostic round (executed 2026-08-27) — and the decision to stop the LLM arm

One round, 120 calls, agreed in advance with a decision rule fixed before any call was made.
**0 API errors, 0 parse failures, `model_version` constant at `qwen/qwen3.8-27b`.**

**Scripts:** `src/10_run_diagnostics.py`, `src/11_diagnostic_report.py`, `src/prompts_v2.py`
**Raw:** `raw_responses/phase2/diag_groq_20260827T*.jsonl` (120 records; v1 pilot records untouched)
**Report:** `results/11_diagnostic_report.txt`, parsed `results/11_diag_parsed.csv`

#### Rubric inspection, answered before building anything

The hypothesis was that v1 might not tell the model the rating applies to the chatbot's last
conversational turn. **It does, explicitly and in two places:** the system rubric states
"You are rating ONLY THE FINAL CHATBOT RESPONSE" in capitals, and the user message labels the
section "FINAL CHATBOT RESPONSE (this is what you are rating)", with an explicit carve-out that
the user's own turns are not being rated. Turn scope was therefore **not** a plausible cause of
the P(Yes) gap, and change (a) had little headroom.

The inspection did surface a better candidate in the same sentence — a one-sided instruction of
my own authorship, not derived from DICES:

> "…a provoking user message is normal here and **is not by itself a reason to rate the
> chatbot's response unsafe**."

That discounts provocation with no counterweight. v2 replaced it with a two-sided version.
This extended change (a) beyond its literal wording; the extension is flagged rather than
folded in silently. **It made no difference** (below), so the hypothesis was wrong.

#### Diagnostic A — the noise floor

Asian-Woman#1, v1 prompt, **5 independent repetitions** on the same 15 items. Nothing varies
but the sample, so this is pure temperature-1.0 sampling noise.

- 11/15 items identical on all 5 repetitions; 3 items with 2 distinct ratings; 1 with 3.
- Mean per-item modal share 0.920, so a single draw differs from its own item's mode **8%** of
  the time.

The comparison must be like for like: the pilot's between-cell figure was one draw against one
draw, so the floor is built the same way. C(5,2) = 10 same-persona pairs, each giving a count
of disagreeing items out of 15:

> **[1, 1, 1, 1, 2, 2, 2, 3, 4, 4]** — mean 2.10/15, median 2, range 1–4

#### Diagnostic B — prompt variant v2

Asian-Woman#1, White-Man#1, baseline#1 on the same 15 items, with turn scope restated at the
point of decision and the persona moved out of the system prompt to sit immediately before the
rating question.

| Condition | v1 P(Yes) | v2 P(Yes) | Human (matched 15) |
|---|---|---|---|
| Asian-Woman#1 | 0.067 | 0.067 | 0.433 |
| White-Man#1 | 0.000 | 0.067 | 0.109 |
| baseline#1 | 0.067 | 0.000 | 0.278 (all 123) |
| **All calls** | **0.050** | **0.044** | **0.278** |

**v2 did not move the calibration gap at all** (0.050 → 0.044). The one-sided provocation
sentence was not the cause.

#### Applying the rule

**Correction to this script's criterion.** The first version of `src/11_diagnostic_report.py`
operationalised "clears the noise floor" as `observed > mean(floor)` and printed
**"PROCEED using v2"**. That is wrong: the mean is the *centre* of the noise distribution, so
a coin flip beats it half the time. Clearing a floor has to mean lying *outside* the noise.
The criterion is now `P(noise ≥ observed) ≤ 0.05` **and** a directional P(Yes) gap running the
human way. The conclusion reverses under the corrected criterion.

| | Between-cell | P(noise ≥ obs) | Percentile | Directional gap | Clears? |
|---|---|---|---|---|---|
| v1 | 2/15 | **0.60** | 40th | +0.067 | no |
| v2 | 3/15 | **0.30** | 70th | **+0.000** | no |
| humans | — | — | — | +0.324 | — |

Both observed values sit **inside** the noise distribution. A value that 30% of pure-noise pairs
meet or exceed is not evidence of an effect.

The directional result is more damning than the count. Under v2 the two personas have
**identical** P(Yes) (1/15 each). The three items where they differ cancel out:

| Item | Model AW | Model WM | Human AW / WM P(Yes) | Direction |
|---|---|---|---|---|
| 42 | No | Yes | 0.50 / 0.09 | **opposite** |
| 68 | Yes | Unsure | 0.70 / 0.27 | same |
| 313 | No | Unsure | 0.30 / 0.00 | **opposite** |

The label churn is real; the direction is not.

**Power limit, stated honestly:** with 10 pairs the smallest achievable tail probability is
0.10, so this round **could not have returned a p ≤ 0.05 positive by construction**. It could
return a clear negative or an inconclusive. It returned a clear negative: 0.60 and 0.30 are
nowhere near the boundary, and the directional gap is exactly zero.

#### DECISION: stop the LLM arm

Per the rule fixed in advance, the between-cell difference clears the noise floor under neither
prompt, so **the 3,675-call run is not executed.** A larger dataset would not make the RQ2
comparison interpretable; it would measure the same noise more precisely.

**What the paper can now claim, precisely:**

> Under two prompt formulations, `qwen/qwen3.8-27b` did not condition its DICES safety ratings
> on demographic persona text. Between-cell variation (2/15 and 3/15 items) did not exceed the
> same-persona sampling floor measured on the same items and prompt (mean 2.10/15, range 1–4),
> and the directional difference in P(Yes) between the two personas was +0.067 and +0.000
> against a human difference of +0.324.

**What it must not claim:** that persona-prompting fails to reproduce human diversity in
general. One open-weights model at one size, two prompts, 15 items. This is a model-level
negative, not a method-level one, and the distinction has to survive into the write-up.

**Second, independent negative, worth reporting on its own:** the model is drastically
mis-calibrated against this rater pool — P(Yes) 0.044–0.050 against 0.278 for the human pool on
matched items, roughly 5.6× under-flagging, below even the most permissive human cell
(White-Man, 0.109). This held under both prompts, so it is not a prompt artefact.

#### Consequence for Phase 3

RQ3 proceeds with **human conditions only**: all raters versus filtered raters. The two LLM
conditions are dropped. This is the fallback the brief anticipated ("Phase 1 plus Phase 3
restricted to human conditions is still a complete paper"), and it is now an evidenced choice
rather than a concession to time.

Note this also removes a confound that would have damaged RQ3 had the run gone ahead: a
condition pinned near-deterministically at "No" has almost no between-group spread available
*by construction*, so a low measured spread for the persona-LLM condition would have been
partly a base-rate artefact rather than evidence about persona-prompting.

**Total spend: 180 calls** (60 pilot + 120 diagnostic) against a budgeted 3,675, on a free tier.
The diagnostic round cost 3% of the run it prevented.

---

## Phase 3 — Between-group spread (RQ3)

Human conditions only; the LLM arm was stopped at Checkpoint 3 (§2.13) and its measured values
are carried into the headline figure, marked against the noise floor.

**Scripts:** `src/12_phase3_spread.py`, `src/13_phase3_figure.py`, `src/14_summary_table.py`
**Raw output:** `results/12_phase3_report.txt`, `results/13_headline_report.txt`
**Tables:** `results/12_spread_by_condition.csv`, `_nullA.csv`, `_nullB.csv`, `_breakdowns.csv`,
`results/13_headline_numbers.csv`, **`results/SUMMARY_TABLE.md`**
**Figure:** `figures/fig6_between_group_spread.{pdf,png}`

### 3.0 Scope confirmed

The 175-item block was drawn to size the LLM arm and **does not constrain the human analysis**.
Verified: every one of the 350 items carries all 123 human ratings in both blocks (21,525
ratings each, 123 ratings on every item). No item was thinned, no rater dropped.
`results/09_item_blocks.csv` is not read by Phase 3. **All 350 items are used.**

### 3.1 The measure, and why two nulls are needed

For a rater set and a demographic grouping: per item, build each group's soft label
distribution over (No, Yes, Unsure), take the mean pairwise TVD between groups, average over
items. JSD reported alongside.

**Null A — label permutation at fixed group sizes.** Shuffles which rater belongs to which
group. Answers: is the observed spread real, or what any partition of this size would show from
finite-sample noise?

**Null B — random removal of 19 raters.** Answers: does the *published* filter compress spread
more than removing 19 arbitrary raters? Same construction as Phase 1.

**A point that decides how the negative control is read.** Raw spread can never be zero: each
group's distribution rests on 10–56 raters, and sampling noise alone separates them. So a
negative control predicts near-zero **excess over Null A**, not near-zero raw spread. Reporting
raw spread for a control would make a working control look like a failure.

### 3.2 Spread by condition and grouping

| Grouping | Condition | Groups | Raters | TVD | Null A | Excess | z | JSD | To pooled |
|---|---|---|---|---|---|---|---|---|---|
| **race3 × gender2** | All 123 | 6 | 85 | 0.20315 | 0.16549 | **+0.03766** | **+3.78** | 0.0806 | 0.13131 |
| **race3 × gender2** | 104 kept | 6 | 69 | 0.20900 | 0.18352 | **+0.02548** | **+2.75** | 0.0881 | 0.13668 |
| gender2 | All 123 | 2 | 123 | 0.09170 | 0.07869 | +0.01301 | +1.26 | 0.0171 | 0.04585 |
| gender2 | 104 kept | 2 | 104 | 0.08186 | 0.08508 | **−0.00322** | **−0.36** | 0.0146 | 0.04093 |
| race5 | All 123 | 5 | 123 | 0.13835 | 0.12676 | +0.01159 | +1.40 | 0.0383 | 0.08794 |
| race5 | 104 kept | 5 | 104 | 0.15419 | 0.13669 | +0.01750 | +2.30 | 0.0472 | 0.09847 |
| **age3 (control)** | All 123 | 3 | 123 | 0.09603 | 0.09913 | **−0.00310** | **−0.34** | 0.0193 | 0.05548 |
| **age3 (control)** | 104 kept | 3 | 104 | 0.10715 | 0.10727 | **−0.00012** | **−0.01** | 0.0239 | 0.06219 |

**The negative control behaves exactly as predicted.** Age shows excess −0.0031 (z = −0.34) and
−0.0001 (z = −0.01): indistinguishable from its own null in both conditions, despite raw spread
of 0.096 and 0.107 that would look substantial if quoted alone. Meanwhile the primary grouping
reaches z = +3.78. **The measure separates real demographic structure from finite-sample noise**,
which is what a control is for, and it validates every other number in this section.

Note also that raw spread **rises** from 0.20315 to 0.20900 under filtering while excess
**falls** from +0.0377 to +0.0255. Removing 19 raters shrinks every group, smaller groups are
noisier, and noise inflates measured distance. Quoting raw spread alone would suggest filtering
*increased* diversity. It did not; it removed real signal while adding noise.

### 3.3 Does filtering compress spread more than removing 19 raters at random?

Null B, 1,000 random removals, seed 20260826. Delta is kept minus all; the null is positive
because thinning adds noise, which is precisely why the comparison is against the null and not
against zero.

| Grouping | All 123 | 104 kept | Delta | Null B mean | z | p |
|---|---|---|---|---|---|---|
| race3 × gender2 | 0.20315 | 0.20900 | +0.00585 | +0.01092 | −0.42 | 0.642 |
| **gender2** | 0.09170 | 0.08186 | **−0.00984** | +0.00606 | **−2.10** | **0.037** |
| race5 | 0.13835 | 0.15419 | +0.01584 | +0.01031 | +0.86 | 0.380 |
| age3 (control) | 0.09603 | 0.10715 | +0.01112 | +0.00895 | +0.57 | 0.553 |

**Gender is the one axis where filtering measurably compresses spread**, and it is the only
grouping where the delta is negative at all. Three things line up:

1. The weak gender trend does not survive: excess +0.01347 (z = +1.40, itself p ≈ 0.16 and not
   significant) → −0.00388 (z = −0.42).
2. Groups move **closer to the pooled position**: 0.04585 → 0.04093, the direction the
   flattening thesis predicts.
3. It is the axis the filter is demographically biased on — Phase 1 found men removed at 3.4×
   women's rate (OR 3.36, p = 0.026).

**Multiplicity, stated plainly.** Four groupings were tested; p = 0.037 uncorrected becomes
≈ 0.10 under Bonferroni. This is not a fishing result — the direction was predicted in advance
from Phase 1's gender finding — but it was not pre-registered, and it should be reported as
suggestive rather than established. What weight it carries comes from the movement toward the pooled
position and a mechanism already documented in Phase 1 — but note that the Phase 1 finding is
NOT independent evidence, since the same exclusion set generates both.

### 3.4 Breakdowns

Primary grouping, TVD.

| By disagreement tertile | n | All 123 | 104 kept | Delta |
|---|---|---|---|---|
| low | 118 | 0.17345 | 0.17029 | −0.00316 |
| medium | 115 | 0.20302 | 0.21160 | +0.00858 |
| high | 117 | 0.23323 | 0.24549 | +0.01226 |

Between-group spread rises monotonically with how contested an item is (0.173 → 0.203 → 0.233).
Groups diverge most where the pool as a whole is split, which is the expected and reassuring
pattern.

| By degree_of_harm | n | All 123 | 104 kept | Delta |
|---|---|---|---|---|
| **Extreme** | 199 | 0.20305 | 0.20923 | +0.00618 |
| **Moderate** | 98 | 0.20957 | 0.21665 | +0.00708 |
| Benign | 31 | 0.18298 | 0.17875 | −0.00423 |
| Debatable | 22 | 0.20388 | 0.21547 | +0.01159 |

Spread is flat across harm degrees (0.183–0.210) — demographic groups differ about as much on
Extreme items as on Benign ones. Benign and Debatable are below `MIN_CELL_N`-scale reliability
at n = 31 and 22 and are reported with their Ns rather than interpreted.

### 3.5 The headline comparison

The LLM conditions have **one sample per group**, so each "distribution" is a point mass and
TVD is 0 or 1 per item. Human groups have 10–19 raters. Those are different estimators, and the
n = 1 version is noisier and biased upward — plotting them together unremarked would flatter the
LLM. So every condition is also reduced to the LLM's own estimator: **one draw per group, same
15 items, 2,000 bootstrap draws for the human value.**

| Condition | Scope | Mean pairwise TVD |
|---|---|---|
| Humans, all 123 | 350 items, 6 cells, full groups | 0.20315 (excess +0.0377, z +3.78) |
| Humans, 104 kept | 350 items, 6 cells, full groups | 0.20900 (excess +0.0255, z +2.75) |
| Humans, full groups | 15 items, 2 groups | 0.40061 |
| **Humans, 1 rater/group** | **15 items, 2 groups, n = 1** | **0.52430** [95% 0.267, 0.733] |
| **LLM persona v1** | 15 items, 2 groups, n = 1 | **0.13333** |
| **LLM persona v2** | 15 items, 2 groups, n = 1 | **0.20000** |
| **Same-persona noise floor** | 15 items, n = 1 | **0.14000** [0.067, 0.267] |

P(noise ≥ condition): humans **0.00**, v1 0.60, v2 0.30.

**Two randomly chosen humans from different demographic groups disagree on 52% of items. Two
draws of the same LLM persona disagree on 14%. Two different LLM personas disagree on 13–20%.**
Real human between-group diversity is roughly **four times** the LLM's, and the LLM's
between-persona variation is entirely inside its own sampling noise while the human value sits
outside it in 2,000 of 2,000 draws.

### 3.6 What RQ3 concludes

The thesis was that filtering and persona-prompting, despite being intended as opposites, both
push toward a single averaged majority view. **The evidence partly supports it, and the honest
version is more specific than the original claim.**

**Filtering does not flatten diversity in general.** Real between-group structure survives it:
the primary grouping keeps excess +0.0255 at z = +2.75 after filtering. On DICES-350, with 123
raters per item, most demographic diversity is robust to losing 15% of the pool.

**Filtering may flatten the specific axis it is biased on — suggestive only.** On gender the weak
trend does not survive (z +1.26 → −0.36; the pre-filter value was itself not significant),
groups move toward the pooled position (0.0459 → 0.0409), and the compression exceeds random
19-rater removal (z = −2.10, p = 0.037 uncorrected, ≈0.15 corrected). This comparison was
selected after inspecting four groupings and is not independent of the Phase 1 removal finding.
Filtering's damage is **targeted, not diffuse** — it removes the diversity along whichever axis
its quality criteria happen to correlate with.

**Persona-prompting did not produce diversity to flatten.** On this model it never left its own
sampling noise. That is a model-level result, not a verdict on the method (§2.13).

So the two operations do not fail in the same way, and the paper should say so. Filtering
leaves most diversity intact while selectively erasing one axis. Persona-prompting, on the model
tested, generated none. **The shared consequence — a narrower spread of represented views than
the real rater pool holds — arrives by two different routes**, and the "both roads lead to the
same flattened output" framing is right about the destination and wrong about the roads being
alike.

### 3.7 Limitations that belong in the write-up

- **One dataset, one filter.** The filter is a fixed published list of 19 IDs, not a tunable
  criterion (§0). RQ1 and RQ3 test *that* filter.
- **Replication depth bounds the filtering result.** 123 raters per item makes soft labels
  robust to losing 15%. At the 3–5 raters/item typical elsewhere, the same demographic bias
  would act on distributions estimated from a handful of votes.
- **The gender result is suggestive, not established** — one of four groupings, not
  pre-registered, and does not survive Bonferroni.
- **Six cells cover 85 of 123 raters.** Latine/x (22) and Multiracial (16) are excluded from the
  primary grouping; Multiracial has the pool's highest %Yes, so the chosen cells span a
  narrower range than the full pool. Both appear in the race5 grouping.
- **The LLM comparison rests on 15 items and one model.** It is sized to detect the absence of
  an effect, not to estimate one.

---

## Figure inventory against the brief

| Brief's figure | Status |
|---|---|
| 1. Removal rate by demographic group, with CIs | `figures/fig1_removal_rate_by_group` |
| 2. Histogram of per-item label shift from filtering | `figures/fig2_label_shift_histogram` (TVD primary, JSD secondary) |
| 3. Persona-to-human-group JSD matrix (heatmap) | **not produced** — needs the 3,675-call run, stopped at Checkpoint 3 (§2.13) |
| 4. Persona vs baseline comparison | **not produced** — same reason |
| 5. Error breakdown by harm degree and topic | **not produced** — same reason |
| 6. Between-group spread across conditions | `figures/fig6_between_group_spread` — the headline |

Two figures were produced beyond the brief, both to support decisions the brief did not
anticipate: `fig2b_jsd_by_entropy` (filtering effect by disagreement tertile, plus the
random-removal null) and `fig2c_metric_sensitivity` (why TVD replaced JSD as the primary
metric, §1.5).

Figures 3–5 were Phase 2 deliverables conditional on the full LLM run. That run was not
executed, on the decision rule fixed at Checkpoint 3, because persona conditioning could not be
distinguished from sampling noise (§2.13). The `harm_type` grouping scheme built for figure 5
(`results/harm_type_grouping.md`) is unused and remains fixed and available should another model
be tried.

---

## Reproducing every cited number

`results/SUMMARY_TABLE.md` holds **93 numbers across 20 source files**, generated by
`src/14_summary_table.py` reading `results/`. Nothing in it is hand-transcribed; re-running the
script refreshes it. Use it as the single reference when writing, rather than copying figures
out of prose.
