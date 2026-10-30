# PLAN.md — Pre-Registration & Analysis Protocol: Tendency Tax

## 1. Research Question

Does an NFL offense experience an expected points added (EPA) penalty or benefit when its play call (pass vs. run) is highly predictable from the pre-snap situation?

Specifically, we test for an empirical **tendency tax**: whether predictable play calls systematically underperform compared to counter-tendency calls when conditioning on down, distance, and game context.

---

## 2. Data Source

- **Dataset**: `nflverse` regular season play-by-play data, seasons 2016 through 2025.
- **Season Scope**: Regular season games only (`season_type == 'REG'`).
- **Holdout Protocol**: Seasons 2024–2025 are strictly locked as confirmatory holdout data and must not be accessed, inspected, or processed until explicitly unlocked.

---

## 3. Sample Filtering

Plays are included if and only if they meet all the following criteria:
1. `season_type == 'REG'` (regular season only).
2. `play_type` in `{'pass', 'run'}`.
3. `down` in `{1, 2, 3}` (excludes 4th downs to avoid special teams and turnover-on-downs distortion).
4. Win probability `wp` between `0.05` and `0.95` (excludes non-competitive garbage time).
5. Exclude quarterback kneels (`qb_kneel == 1`) and quarterback spikes (`qb_spike == 1`).
6. Exclude plays in the final 2 minutes of the 2nd and 4th quarters (`half_seconds_remaining <= 120` when `qtr` in `{2, 4}`) to remove two-minute drill structural pass bias.

---

## 4. Call Label

- **Label**: `qb_dropback` (1 = pass / dropback, 0 = designed run).
- **Rationale**: `qb_dropback` captures scrambles and sacks as intended pass calls, accurately reflecting offensive pre-snap play-call intention rather than post-play execution (`play_type`).

---

## 5. Model Features (Pre-Snap Situation Only)

The tendency model estimates pass probability $\hat{p} = P(\text{qb\_dropback} = 1 \mid X)$ using strictly pre-snap situational features.

To avoid post-snap leakage and formation-based confounding, **`shotgun` and `no_huddle` are excluded from the primary model**. `epa` is an outcome only, and `xpass` is reference only.

Exactly 10 situational features:
1. `down`: Current down (1, 2, 3).
2. `ydstogo`: Yards to go for a first down or touchdown.
3. `yardline_100`: Distance to opponent's end zone (1–99).
4. `qtr`: Game quarter (1, 2, 3, 4, or 5 for overtime).
5. `game_seconds_remaining`: Seconds remaining in the regulation/overtime game.
6. `half_seconds_remaining`: Seconds remaining in the current half.
7. `score_differential`: Possession team score minus defending team score.
8. `posteam_timeouts_remaining`: Possession team timeouts remaining (0–3).
9. `defteam_timeouts_remaining`: Defensive team timeouts remaining (0–3).
10. `posteam_spread`: Spread line from possession team perspective (`spread_line > 0` indicates home team favored).

---

## 6. Data Splits

- **Early Split (2016–2021)**: Baseline model estimation, exploratory analysis, and initial specification checks.
- **Tune Split (2022–2023)**: Validation, model calibration, hyperparameter tuning, and decision threshold confirmation.
- **Holdout Split (2024–2025)**: Confirmatory evaluation dataset (strictly holdout; no access until confirmatory phase).

---

## 7. Estimand and Specification B (Spec B)

The primary econometric specification (Spec B) models play-level EPA using an interaction between the call and predicted call probability $\hat{p}$, with down $\times$ distance-bucket fixed effects:

$$\text{EPA}_i = \beta_0 + \beta_1 \cdot \text{call}_i + \beta_2 \cdot \hat{p}_i + \beta_3 \cdot (\text{call}_i \cdot \hat{p}_i) + \alpha_{\text{down} \times \text{dist\_bucket}} + \epsilon_i$$

where:
- $\text{call}_i = \text{qb\_dropback}_i \in \{0, 1\}$
- $\hat{p}_i$: Pre-snap estimated pass probability
- $\alpha_{\text{down} \times \text{dist\_bucket}}$: Fixed effects for Down $\times$ Distance bucket (Short: 1–3, Medium: 4–7, Long: 8+)
- **Estimand**:
  $$\delta = -\frac{\beta_3}{2}$$

### Interpretation of $\delta$:
$\delta$ quantifies the differential EPA impact associated with situational predictability (the tendency tax). A negative $\delta$ indicates that offenses lose EPA when calling the situationally predictable play relative to counter-tendency calls.

---

## 8. Inference Protocol

1. **Game-Clustered Bootstrap**:
   - Re-sample games with replacement (1,000 bootstrap replications).
   - Compute the empirical 95% confidence interval for $\delta$.
2. **Placebo Distribution**:
   - Permute / shuffle $\hat{p}$ within situation cells across 500 replications to generate the empirical null distribution.
   - Calculate the 2.5th and 97.5th percentiles of the placebo distribution.

---

## 9. Pre-Registered Confirmatory Rule

The hypothesis of an empirical tendency tax is designated as **"Supported"** if and only if all three conditions are satisfied on the holdout split (2024–2025):
1. **Zero Exclusion**: The holdout 95% bootstrap confidence interval excludes 0.
2. **Directional Consistency**: The sign of $\delta$ in the holdout matches the sign estimated in the early split.
3. **Placebo Separation**: The empirical holdout $\delta$ falls outside the placebo 2.5%–97.5% range ($p < 0.05$ under the null).

If any of these three conditions is violated, the outcome is officially reported as **"Not Supported / Inconclusive"**.

---

## 10. Human Decision Blanks

The following three parameters require explicit human decision. Do not implement or hardcode these values until filled:

### Decision 1: Field-Zone Cut Points in `yardline_100`
- **Value**: `[20, 80]` (Confirmed: Red Zone: 1–20, Midfield: 21–79, Backed Up: 80–99)
- **Tradeoff**: Finer cut points (e.g. 4–5 zones) isolate distinct field-position strategies but reduce play counts per zone, whereas coarser cut points preserve statistical power but pool heterogeneous tactical zones.

### Decision 2: Minimum Plays Per Cell
- **Value**: `[ HUMAN DECISION: ____________________ ]`
- **Suggested Value**: `50`
- **Tradeoff**: A higher threshold (e.g., 100 plays) ensures reliable cell empirical pass rates but discards rarer situational combinations, whereas a lower threshold (e.g., 25 plays) retains broader coverage at the cost of higher noise in cell-level tendencies.

### Decision 3: Default Splits for Cell Grid and Team Card
- **Value**: `[ HUMAN DECISION: ____________________ ]`
- **Suggested Value**: Down (1, 2, 3) $\times$ Distance (Short 1–3, Medium 4–7, Long 8+), aggregated over the `tune` split (2022–2023) for team cards.
- **Tradeoff**: Adding field zone or score margin produces a granular situational grid but dilutes sample size per team card, whereas a 3x3 Down $\times$ Distance grid guarantees high statistical power per team while averaging over field-position dynamics.
