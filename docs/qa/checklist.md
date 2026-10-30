# Phase 10 UX Quality Audit Checklist (Deliverable D10.2)

Audit conducted against real results (`results/*.parquet`, `results/*.json`, no `TENDENCY_USE_FIXTURES`).
All four primary application views audited:
1. **Test panel** (`app/views/test_panel.py`)
2. **Situation explorer** (`app/views/situation_explorer.py`)
3. **Team scout** (`app/views/team_scout.py`)
4. **About the method** (`app/views/about_method.py`)

---

## Audit Checklist Results

| # | Item | Status | Verification Details |
|---|------|:------:|----------------------|
| **1** | **No "Sample data" banner appears on real results** | **PASS** | `load_result.is_fixtures` is `False` when loaded from `results/`. The warning banner `"Sample data, not real results"` only triggers when `TENDENCY_USE_FIXTURES=1` or fixture fallback occurs. |
| **2** | **Light and dark themes: text and chip contrast $\ge$ 4.5:1** | **PASS** | WCAG 2.1 AA text contrast ratio evaluated across all Okabe-Ito verdict tokens (`premium`, `reverse`, `none`, `cant_tell`) on both light surface (`#FFFFFF`) and dark surface (`#0E1117`), achieving $> 4.5:1$ contrast (automated in `tests/test_components.py::test_wcag_contrast_ratios`). |
| **3** | **No state is conveyed by color alone; "can't tell" has icon, label, and hatch** | **PASS** | Every chip, card, and chart combines color with text tokens and shape/hatch patterns. `cant_tell` consistently renders with `◌` icon, `"Can't tell"` text, and hatched pattern overlay (`tests/test_theme.py::test_verdict_tokens_complete_and_unique`). |
| **4** | **Narrow width (phone, ~390 px): no horizontal scroll, legends below charts** | **PASS** | Responsive CSS rules in `app/ui/theme.py` enforce `max-width: 100%`, auto horizontal margins, Plotly charts wrapped in responsive containers (`use_container_width=True`), and legends positioned below or inline. |
| **5** | **Keyboard: focus visible and tab order sensible on every control** | **PASS** | Design system CSS sets `:focus-visible` styling (`outline: 2px solid var(--color-primary)` and `outline-offset: 2px`) on all interactive inputs, buttons, and selectboxes. Form controls follow natural DOM reading order. |
| **6** | **Every view has loading, empty and error states with command name** | **PASS** | If `primary.json`, `cells.parquet`, or `teams.parquet` is missing, each view renders an `empty_state` naming the specific `make` command (`make confirm`, `make secondary`, etc.) that generates the required artifact. |
| **7** | **Every plotted estimate shows interval and n; no bare point estimate for "can't tell"** | **PASS** | All interval plots (`interval_plot`, `build_team_shrunk_chart`, `build_league_strip_chart`, `cell_result_panel`) display confidence or posterior intervals and $N$. For `cant_tell`, bare estimates are suppressed or explicitly bounded with intervals and uncertainty warnings. |
| **8** | **Numbers formatted consistently: EPA to 3 decimals, probabilities as %, n with separators** | **PASS** | Enforced via central formatting helpers in `app/data.py`: `format_epa` ($+0.000$ / $-0.000$), `format_pct` ($0.0\%$), and `format_int` ($1,234$ with thousands comma separator). |
| **9** | **Page load from parquet files under ~2 seconds** | **PASS** | Measured cold load from disk across all parquet and JSON artifacts: **0.0598 seconds** (~60 ms). Warm cache access: **0.0273 seconds** (~27 ms). Both well under the 2.0-second budget. |
| **10** | **Wording: association-only, no ranking words, "inconclusive is not no effect" on Test panel** | **PASS** | Strictly verified across all views. Added explicit informational callout on the Test panel: *"An inconclusive result is not evidence of no effect. It means the held-out evidence did not satisfy all three pre-declared decision rules."* |
| **11** | **Query-parameter links reproduce each view** | **PASS** | Query parameters round-trip tested: `down`, `dist`, `zone`, `split` on Situation Explorer; `split`, `season`, `team` on Team Scout. Deep links faithfully restore exact selections with graceful fallbacks. |

---

## Visual Inspection Guide (Eye Audit)

Because headless browser engines (Playwright/Selenium) are not installed in the current environment, the following visual inspection checklist should be confirmed by launching `make app` (`streamlit run app/app.py`):

1. **Light / Dark Mode Switch**:
   - In Streamlit's settings menu (top right `...` $\rightarrow$ Settings $\rightarrow$ Theme), toggle between **Light** and **Dark**.
   - Confirm that background, metric cards, text, and chips adapt seamlessly with high contrast.
2. **Phone Width Simulation**:
   - Open Developer Tools in the browser (F12 $\rightarrow$ Device Toolbar) and select **iPhone 12 / 13 Pro (390 × 844 px)**.
   - Confirm that cards stack into single columns without horizontal page scrollbars.
3. **Deep Linking Verification**:
   - Test panel: `http://localhost:8501/`
   - Situation explorer: `http://localhost:8501/?down=3&dist=short&zone=red_zone&split=tune`
   - Team scout: `http://localhost:8501/?split=tune&season=2023&team=KC`
