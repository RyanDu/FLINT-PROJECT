# Flint Financial Scenario Builder

Forecast monthly **Revenue**, **COGS**, and **Gross Income** under multiple planning scenarios, powered by a reusable SKU schedule engine. The app reads your Excel workbook (Rev/COGS SKU sheets), turns each SKU into a per-person monthly cash-flow vector, and then “shifts and sums” cohorts of hires to produce forecasts. Includes **scenario toggling** and **36-month comparisons**.

---

## ✨ Features

- **Two parsers, one engine**
  - Parse *Rev (SKU)* → per-person Revenue schedule  
  - Parse *COGS (SKU)* → per-person COGS schedule  
  - Both become `{ "SKU": [month1, month2, ...] }` dictionaries
- **Forecasting via cohorts**
  - Define monthly hires + SKU mix + start lag → create cohorts  
  - Aggregate by “shifted & scaled” addition (discrete convolution)
- **Scenario presets**
  - Conservative / Realistic / Aggressive (editable multipliers)
- **Visualization**
  - **Monthly:** Revenue / COGS / Gross / Cash Flow + **Margin %**  
  - **36-month scenario comparison:** Gross & Margin% across presets
- **Robust to messy sheets**
  - Finds the `Months` row and the `Revenue` / `Total` / `Cost` row inside each SKU block  
  - Fills missing months with `0` and preserves negatives (e.g., deposit refunds)

---

## 📦 Project Structure

```bash
flint-project/  
├─ app.py # Single-file Streamlit app (engine + UI)  
├─ sku_engine.py # Shared parsing + forecast engine  
└─ requirements.txt # Streamlit config
```

---

## 📊 Input Workbook (expected sheets)

- **Rev (SKU)**  
  Blocks like:
  - Row with **SKU name** (e.g., `Rev-SKU#R10`)
  - A **Months** row: 1, 2, 3, …  
  - A **Revenue** row: amounts aligned to those months (others = 0)

- **COGS (SKU)**  
  Same pattern but values row is **Total** 

> The parser auto-detects blocks by the first column (SKU name), the `Months` row, and then a `Revenue` / `Total` / `Cost` row nearby.

---

## 🧠 How It Works (in short)

1. **Parse** each SKU into a one-person schedule vector `s = [m1, m2, ...]` where index 0 = Month 1.  
2. **Build cohorts** from your planning inputs:
   - Monthly nurses (baseline × scenario multiplier)
   - SKU mix (weights sum to ~1)
   - Start lag (e.g., selection → working start in months)  
   - → For each month and SKU, create a cohort `(start_month, headcount)`.
3. **Forecast** by **shift-and-sum**:
   - For each cohort, add `s` into the output starting at `start_month` (scaled by headcount).  
   - Do this for revenue schedules and again for the mapped COGS schedules.
4. **Metrics**:
   - `Gross = Revenue - COGS`
   - `Margin % = Gross / Revenue`
   - `Cash Flow = Gross` *(in this version; extend with OpEx/CapEx later)*

---

## 🚀 Getting Started

### Prerequisites
- Python **3.9+** (3.12 works great)
- Local copy of your Excel workbook (e.g., `Flint SWE Project - Finance Inputs.xlsx`)

### Install
```bash
pip install streamlit pandas numpy openpyxl
```

### Run (split engine + UI)
```bash
streamlit run app.py
```

When the app opens:
1.  Adjust **Scenario** and **Multipliers** in the sidebar.
2.  Edit **Hiring plan**, **SKU mix**, and **Rev→COGS mapping** in-page.
3.  View **Monthly** metrics and **36-month** scenario comparisons.
---
## ⚙️ Configuration (what you can tweak)

-   **Scenario presets**:  
    `hire_multiplier`, `start_lag`, `price_multiplier`, `cogs_multiplier`
    
-   **Hiring plan**:  
    A table of baseline hires per month (you can extend beyond 12 months)
    
-   **SKU mix**:  
    Weights per Revenue SKU (auto-normalized to sum to 1)
    
-   **Rev→COGS mapping**:  
    Which COGS SKU to use for each Revenue SKU

## 🧾 Formulas

-   **Gross Income (Monthly)** = `Revenue - COGS`
    
-   **Margin % (Monthly)** = `Gross / Revenue` (0% when Revenue = 0)
    
-   **Cash Flow (Monthly)** = `Gross`  
    _(To refine Cash Flow, add OpEx/CapEx/working-capital deltas later.)_

## 🧩 Engine API (if you import `sku_engine.py`)

```python
from sku_engine import build_structure_dict_from_sheet, forecast_from_cohorts

rev_dict  = build_structure_dict_from_sheet(xlsx, "Rev (SKU)",  ["Revenue"])
cogs_dict = build_structure_dict_from_sheet(xlsx, "COGS (SKU)", ["Total", "Cost"]) # schedule: per-person vector, cohorts: [(start_month, headcount)] y = forecast_from_cohorts(schedule, cohorts, horizon=60) # strict to horizon
```

-   **`build_structure_dict_from_sheet`** → `{sku: np.array([m1, m2, ...])}`
    
-   **`forecast_from_cohorts`** → Shift-sum aggregation, **strictly** returns length = `horizon` (truncates/pads as needed).

## 🧪 Testing Ideas
-   Use a synthetic SKU with an obvious pattern, e.g. `s = [100, 0, 0, 50, 50]`, cohorts `[(1, 1), (4, 2)]` ⇒ manually verify sums.
-   Include a negative month in Rev to simulate deposit refunds; make sure it nets correctly.
-   Set `start_lag` to push cohorts near the horizon edge to verify truncation.

## 🧯 Troubleshooting

**ValueError: operands could not be broadcast together…**  
Cause: a SKU forecast extended beyond your fixed `horizon` (e.g., 63 vs 60).  
Fix (already built-in): `forecast_from_cohorts` returns a fixed length when `horizon` is provided; aggregation also truncates/pads.

**Workbook path not found**  
Use an **absolute path**, e.g. `/Users/you/Desktop/flint/Flint SWE Project - Finance Inputs.xlsx`.

**Margin % shows 0%**  
When Revenue is 0 for a month, Margin% is defined as 0 (avoid divide-by-zero).

## 📝 License

MIT (or your company standard) — update as needed.

## 🙌 Acknowledgements

Thanks to the Flint team for the inputs and SKU definitions. This tool is a first step toward a CFO agent: transparent, auditable, and scenario-driven.