import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, Iterable, Tuple, Optional, List
from pathlib import Path

st.set_page_config(page_title="Flint Scenario Builder", layout="wide")
st.title("Flint — Revenue • COGS • Gross (Scenarios)")


DEFAULT_WB = "https://docs.google.com/spreadsheets/d/11E79iziBz0g4xvppUXDZ_Jd6AoLPhboo0X2L-GG3D_U/export?format=xlsx"

def _find_numeric_positions(row: pd.Series):
    return [i for i, v in enumerate(row) if isinstance(v, (int, float)) and not pd.isna(v)]

def _row_has_token(row: pd.Series, token: str) -> bool:
    token_lower = str(token).strip().lower()
    for v in row:
        if isinstance(v, str) and v.strip().lower() == token_lower:
            return True
    return False

def _extract_blocks_generic(df: pd.DataFrame, value_labels: Iterable[str]) -> Dict[str, np.ndarray]:
    value_labels = [s.strip().lower() for s in value_labels]
    result: Dict[str, np.ndarray] = {}
    first_col = df.columns[0]
    idxs = df[df[first_col].notna()].index.tolist()

    for start_idx in idxs:
        name = str(df.iloc[start_idx, 0]).strip()
        if not name or name.lower().startswith("nan"):
            continue

        months_row_idx = None
        for r in range(start_idx + 1, min(start_idx + 6, len(df))):
            if _row_has_token(df.iloc[r], "Months"):
                months_row_idx = r
                break
        if months_row_idx is None:
            continue

        values_row_idx = None
        for r in range(months_row_idx + 1, min(months_row_idx + 6, len(df))):
            row = df.iloc[r]
            if any(isinstance(val, str) and val.strip().lower() in value_labels for val in row):
                values_row_idx = r
                break
            if r == months_row_idx + 1:
                values_row_idx = r
        if values_row_idx is None:
            continue

        months_row = df.iloc[months_row_idx]
        values_row = df.iloc[values_row_idx]
        num_pos = _find_numeric_positions(months_row)
        if not num_pos:
            continue

        months = months_row.iloc[num_pos].astype(int).tolist()
        vals = pd.to_numeric(values_row.iloc[num_pos], errors="coerce").fillna(0.0).astype(float).tolist()

        horizon = int(max(months))
        vec = np.zeros(horizon, dtype=float)
        for m, v in zip(months, vals):
            if m is not None and 1 <= int(m) <= horizon:
                vec[int(m) - 1] = float(v)
        result[name] = vec

    return result

def build_structure_dict_from_sheet(excel_path: str, sheet_name: str, value_labels: Iterable[str]) -> Dict[str, np.ndarray]:
    df = pd.read_excel(excel_path, sheet_name=sheet_name)
    return _extract_blocks_generic(df, value_labels)

def forecast_from_cohorts(schedule: np.ndarray, cohorts: Iterable[Tuple[int, int]], horizon: Optional[int] = None) -> np.ndarray:
    s = np.asarray(schedule, dtype=float)
    if horizon is None:
        max_end = 0
        for start, _ in cohorts:
            end = int(start) - 1 + len(s)
            if end > max_end:
                max_end = end
        horizon = max(max_end, len(s))
    y = np.zeros(horizon, dtype=float)
    for start, cnt in cohorts:
        offset = int(start) - 1
        if offset >= horizon:
            continue
        take = min(len(s), horizon - offset)
        if take > 0:
            y[offset:offset + take] += s[:take] * float(cnt)
    return y

def _fix_len(x: np.ndarray, L: int) -> np.ndarray:
    if len(x) > L:
        return x[:L]
    if len(x) < L:
        return np.pad(x, (0, L - len(x)))
    return x

@st.cache_data(show_spinner=False)
def load_structures(workbook_path: str):
    rev_dict = build_structure_dict_from_sheet(workbook_path, "Rev (SKU)", value_labels=["Revenue"])
    cogs_dict = build_structure_dict_from_sheet(workbook_path, "COGS (SKU)", value_labels=["Total", "Cost"])
    return rev_dict, cogs_dict

# ---------------------------
# data
# ---------------------------
workbook_file = st.text_input("Workbook path (local .xlsx)", value=DEFAULT_WB)
if not Path(workbook_file).exists():
    st.info("Paste an absolute path if the workbook isn't in the same folder.")
try:
    rev_dict, cogs_dict = load_structures(workbook_file)
except Exception as e:
    st.error(f"Failed to load workbook: {e}")
    st.stop()

rev_skus = sorted(rev_dict.keys())
cogs_skus = sorted(cogs_dict.keys())

# ---------------------------
# scenario 
# ---------------------------
SCENARIOS = {
    "Conservative": {"hire_multiplier": 0.8, "start_lag": 2, "price_multiplier": 0.98, "cogs_multiplier": 1.05},
    "Realistic":    {"hire_multiplier": 1.0, "start_lag": 1, "price_multiplier": 1.00, "cogs_multiplier": 1.00},
    "Aggressive":   {"hire_multiplier": 1.2, "start_lag": 0, "price_multiplier": 1.03, "cogs_multiplier": 0.97},
}

# ---------------------------
# Control
# ---------------------------
st.sidebar.header("Single Scenario Controls")
scenario_name = st.sidebar.radio("Scenario preset", list(SCENARIOS.keys()), index=1)
scn = SCENARIOS[scenario_name]

horizon = st.sidebar.number_input("Horizon (months)", min_value=12, max_value=120, value=60, step=12)
hire_multiplier = st.sidebar.number_input("Hires Multiplier", value=float(scn["hire_multiplier"]), step=0.05)
start_lag = st.sidebar.number_input("Start Lag (months)", min_value=0, max_value=12, value=int(scn["start_lag"]))
price_mult = st.sidebar.number_input("Revenue Multiplier", value=float(scn["price_multiplier"]), step=0.01, format="%.2f")
cogs_mult = st.sidebar.number_input("COGS Multiplier", value=float(scn["cogs_multiplier"]), step=0.01, format="%.2f")

st.sidebar.caption("These controls drive the single-scenario charts below. The comparison section uses presets over 36 months.")

# ---------------------------
# plan, calculate
# ---------------------------
st.subheader("Hiring plan & SKU mix")
st.caption("Define baseline hires per month and how they split across Revenue SKUs. Scenario multipliers apply on top.")

months = list(range(1, 13))
default_hires = [8,8,10,10,12,12, 14,14,16,16,18,18]
hires_df = pd.DataFrame({"Month": months, "Baseline Hires": default_hires})
hires_df = st.data_editor(hires_df, num_rows="dynamic", use_container_width=True)

mix_df = pd.DataFrame({"Rev SKU": rev_skus, "Weight": [round(1/len(rev_skus), 3)]*len(rev_skus)})
mix_df = st.data_editor(mix_df, use_container_width=True)

map_df = pd.DataFrame({
    "Rev SKU": rev_skus,
    "COGS SKU": [cogs_skus[min(i, len(cogs_skus)-1)] for i in range(len(rev_skus))]
})
map_df = st.data_editor(map_df, use_container_width=True)

def normalize_weights(df: pd.DataFrame) -> Dict[str, float]:
    w = {r["Rev SKU"]: float(r["Weight"]) for _, r in df.iterrows()}
    s = sum(w.values()) or 1.0
    return {k: (v / s) for k, v in w.items()}

def build_cohorts(hire_rows: pd.DataFrame, mix: Dict[str, float], lag: int, multiplier: float) -> Dict[str, List[Tuple[int,int]]]:
    cohorts: Dict[str, List[Tuple[int,int]]] = {k: [] for k in mix.keys()}
    for _, row in hire_rows.iterrows():
        m = int(row["Month"])
        hires = int(round(float(row["Baseline Hires"]) * multiplier))
        if hires <= 0:
            continue
        for sku, w in mix.items():
            cnt = int(round(hires * w))
            if cnt > 0:
                cohorts[sku].append((m + lag, cnt))
    return cohorts

mix = normalize_weights(mix_df)
cohorts = build_cohorts(hires_df, mix, lag=int(start_lag), multiplier=float(hire_multiplier))

# ---------------------------
# forecast
# ---------------------------
def run_forecast(cohorts: Dict[str, List[Tuple[int,int]]], horizon: int,
                 price_mult: float, cogs_mult: float):
    revenue = np.zeros(horizon, dtype=float)
    cogs = np.zeros(horizon, dtype=float)
    per_sku_rev: Dict[str, np.ndarray] = {}
    per_sku_cogs: Dict[str, np.ndarray] = {}

    cogs_map = {r["Rev SKU"]: r["COGS SKU"] for _, r in map_df.iterrows()}

    for rev_sku, cohort_list in cohorts.items():
        if rev_sku not in rev_dict:
            continue

        rev_sched = rev_dict[rev_sku] * float(price_mult)
        r = forecast_from_cohorts(rev_sched, cohort_list, horizon)
        r = _fix_len(r, horizon)
        revenue += r
        per_sku_rev[rev_sku] = r

        cogs_sku = cogs_map.get(rev_sku)
        if cogs_sku and cogs_sku in cogs_dict:
            c_sched = cogs_dict[cogs_sku] * float(cogs_mult)
            c = forecast_from_cohorts(c_sched, cohort_list, horizon)
            c = _fix_len(c, horizon)
            cogs += c
            per_sku_cogs[rev_sku] = c
        else:
            per_sku_cogs[rev_sku] = np.zeros(horizon, dtype=float)

    gross = revenue - cogs
    cash_flow = gross.copy() 
    margin_pct = np.divide(gross, revenue, out=np.zeros_like(gross), where=(revenue!=0)) * 100.0
    return revenue, cogs, gross, cash_flow, margin_pct, per_sku_rev, per_sku_cogs

revenue, cogs, gross, cash_flow, margin_pct, per_sku_rev, per_sku_cogs = run_forecast(
    cohorts, horizon=horizon, price_mult=float(price_mult), cogs_mult=float(cogs_mult)
)

# ---------------------------
# KPI
# ---------------------------
c1, c2, c3 = st.columns(3)
c1.metric("Total Revenue", f"${revenue.sum():,.0f}")
c2.metric("Total COGS",    f"${cogs.sum():,.0f}")
c3.metric("Gross Income",  f"${gross.sum():,.0f}")

# ---------------------------
# monthly scheme
# ---------------------------
st.subheader("Monthly Metrics — Single Scenario")
monthly_df = pd.DataFrame({
    "Month": np.arange(1, len(revenue)+1),
    "Revenue": revenue,
    "COGS": cogs,
    "Gross": gross,
    "Cash Flow": cash_flow,
    "Margin %": margin_pct,
})
st.line_chart(monthly_df.set_index("Month")[["Revenue", "COGS", "Gross", "Cash Flow"]])
st.line_chart(monthly_df.set_index("Month")[["Margin %"]])

with st.expander("Monthly table"):
    st.dataframe(monthly_df, use_container_width=True)

with st.expander("Per-SKU breakdown (monthly)"):
    br = pd.DataFrame({"Month": np.arange(1, len(revenue)+1)})
    for k, v in per_sku_rev.items():
        br[f"Rev | {k}"] = v
    for k, v in per_sku_cogs.items():
        br[f"COGS | {k}"] = v
    st.dataframe(br, use_container_width=True)

with st.expander("Cohorts used"):
    st.write({k: v for k, v in cohorts.items() if v})

st.divider()

# ---------------------------
# 36 months
# ---------------------------
st.header("Scenario Comparison — 36-month Forecast")

def run_with_preset(preset: Dict, months: int = 36):
    mix = normalize_weights(mix_df)
    cohorts = build_cohorts(hires_df, mix, lag=int(preset["start_lag"]), multiplier=float(preset["hire_multiplier"]))
    r, c, g, cf, mp, _, _ = run_forecast(
        cohorts, horizon=months, price_mult=float(preset["price_multiplier"]), cogs_mult=float(preset["cogs_multiplier"])
    )
    return r, c, g, cf, mp

comp_gross = pd.DataFrame({"Month": np.arange(1, 36+1)})
comp_margin = pd.DataFrame({"Month": np.arange(1, 36+1)})
totals_rows = []

for name, preset in SCENARIOS.items():
    r, c, g, cf, mp = run_with_preset(preset, months=36)
    comp_gross[name] = g
    comp_margin[name] = mp
    totals_rows.append({
        "Scenario": name,
        "Total Revenue (36m)": r.sum(),
        "Total COGS (36m)": c.sum(),
        "Total Gross (36m)": g.sum(),
        "Avg Margin % (36m)": float(np.nanmean(mp)),
    })

st.subheader("Gross Income — by Scenario (36 months)")
st.line_chart(comp_gross.set_index("Month"))

st.subheader("Margin % — by Scenario (36 months)")
st.line_chart(comp_margin.set_index("Month"))

with st.expander("Scenario totals (36 months)"):
    totals_df = pd.DataFrame(totals_rows)
    fmt = totals_df.copy()
    for col in ["Total Revenue (36m)", "Total COGS (36m)", "Total Gross (36m)"]:
        fmt[col] = fmt[col].map(lambda x: f"${x:,.0f}")
    fmt["Avg Margin % (36m)"] = fmt["Avg Margin % (36m)"].map(lambda x: f"{x:.1f}%")
    st.dataframe(fmt, use_container_width=True)

st.caption("Note: Cash Flow here equals Gross Income because only Revenue and COGS are modeled. Add OpEx/CapEx/working-capital items to refine cash flow later.")
