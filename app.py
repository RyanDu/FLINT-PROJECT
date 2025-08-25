import pandas as pd
import math
import streamlit as st
import numpy as np
from typing import Dict, Any, List, Tuple

# --- streamlit init

st.set_page_config(page_title="Flint project", layout="wide")
st.title("Flint - data")

def google_sheet_xls_url() -> str:
    return "https://docs.google.com/spreadsheets/d/11E79iziBz0g4xvppUXDZ_Jd6AoLPhboo0X2L-GG3D_U/export?format=xlsx"

url = google_sheet_xls_url()

xls = pd.ExcelFile(google_sheet_xls_url())
sheet_names = xls.sheet_names

rev_sheet_name = "Rev (SKU)"
cog_sheet_name = "COGS (SKU)"
df = pd.read_excel(xls, sheet_name=rev_sheet_name)

def get_sheet(xls: pd.ExcelFile, key: str):
    return pd.read_excel(xls, sheet_name=key, header=None)

df_rev_sheet = get_sheet(xls, rev_sheet_name)
df_cog_sheet = get_sheet(xls, cog_sheet_name)


def extract_revenue_lists(df: pd.DataFrame)-> Dict[str, List[int]]:
    df.iloc[:, 0] = df.iloc[:,0].ffill()

    structure_series = df.iloc[:, 0].astype(str).str.strip()
    df = df[structure_series.ne("") & ~structure_series.str.lower().str.startswith("total")]

    label_series = df.iloc[:, 1].astype(str)
    revenue_rows = df[label_series.str.contains("revenue", case=False, na=False)].copy()

    if revenue_rows.empty:
        return {}
    
    value_df = revenue_rows.iloc[:, max(0, 1) + 1 :].copy()

    def _to_number(x):
        if pd.isna(x):
            return np.nan
        s = str(x).replace("$", "").replace(",","").strip()
        return pd.to_numeric(s, errors="coerce")
    
    value_df = value_df.applymap(_to_number).fillna(0)

    value_df = value_df.round(0).astype(int)

    names = revenue_rows.iloc[:, 0].astype(str).str.strip().tolist()
    data_dict: Dict[str, List[int]] = {}

    for idx, name in zip(revenue_rows.index, names):
        seq = value_df.loc[idx].tolist()
        data_dict[name] = seq
    
    return data_dict

rev_dict = extract_revenue_lists(df_rev_sheet)
print(rev_dict)

def _parse_month_like(s: str) -> pd.Period:
    s = str(s).strip()
    try:
        dt = pd.to_datetime(s, format="%b %y", errors="raise")
    except Exception:
        dt = pd.to_datetime(s, errors="raise")
    return dt.to_period("M")

def compute_revenue_waterfall(
    rev_dict: Dict[str, List[float]],
    starts: pd.DataFrame,
    model_col: str = "Revenue SKU",
    start_col: str = "Month Nurse Selected", 
    count_col: str = "# of Nurses",
    horizon: int | None = None,               
    month_format: str = "%b %y",              
) -> Tuple[pd.DataFrame, Dict[str, np.ndarray], np.ndarray]:
    df = starts.copy()

    df[model_col] = df[model_col].astype(str).str.strip()
    df[count_col] = df[count_col].astype(int)
    df["_start_period"] = df[start_col].apply(_parse_month_like)

    # calc timeline
    min_p = df["_start_period"].min()
    max_end = min_p
    for _, row in df.iterrows():
        m = row[model_col]
        if m not in rev_dict:
            raise KeyError(f"Model '{m}' not found in rev_dict.")
        L = len(rev_dict[m])
        end_p = row["_start_period"] + (L - 1)
        if end_p > max_end:
            max_end = end_p

    if horizon is not None and horizon > 0:
        desired_end = min_p + (horizon - 1)
        if desired_end > max_end:
            max_end = desired_end

    periods = pd.period_range(min_p, max_end, freq="M")
    N = len(periods)
    idx_map = {p: i for i, p in enumerate(periods)}

    models = sorted(df[model_col].unique())
    breakdown: Dict[str, np.ndarray] = {m: np.zeros(N, dtype=float) for m in models}
    total = np.zeros(N, dtype=float)

    for _, row in df.iterrows():
        m = row[model_col]
        start_idx = idx_map[row["_start_period"]]
        cnt = int(row[count_col])
        sched = np.asarray(rev_dict[m], dtype=float)
        L = len(sched)

        end_idx = min(start_idx + L, N)
        window = end_idx - start_idx
        if window <= 0:
            continue

        contrib = sched[:window] * cnt
        breakdown[m][start_idx:end_idx] += contrib
        total[start_idx:end_idx] += contrib

    month_labels = periods.to_timestamp().strftime(month_format)
    out_df = pd.DataFrame({"month": month_labels, "total": total})
    for m in models:
        out_df[m] = breakdown[m]

    return out_df, breakdown, total


starts = pd.DataFrame({
  "Revenue SKU": ["Rev-SKU#R10", "Rev-SKU#R10", "Rev-SKU#R10"],
  "Month Nurse Selected": ["Sep 25", "Nov 25", "Oct 25"],  # 1-based
  "# of Nurses": [10, 20, 30],
})

out_df, breakdown, total = compute_revenue_waterfall(rev_dict, starts)
print(out_df)