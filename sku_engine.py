
from typing import Dict, List, Tuple, Iterable, Optional
import pandas as pd
import numpy as np

def _find_numeric_positions(row: pd.Series):
    return [i for i, v in enumerate(row) if isinstance(v, (int, float)) and not pd.isna(v)]

def _row_has_token(row: pd.Series, token: str) -> bool:
    token_lower = str(token).strip().lower()
    for v in row:
        if isinstance(v, str) and v.strip().lower() == token_lower:
            return True
    return False

def _extract_blocks_generic(df: pd.DataFrame, value_labels: Iterable[str]):
    value_labels = [s.strip().lower() for s in value_labels]
    result = {}
    first_col = df.columns[0]
    idxs = df[df[first_col].notna()].index.tolist()
    for start_idx in idxs:
        name = str(df.iloc[start_idx, 0]).strip()
        if not name or name.lower().startswith("nan"):
            continue
        # locate months row
        months_row_idx = None
        for r in range(start_idx + 1, min(start_idx + 6, len(df))):
            if _row_has_token(df.iloc[r], "Months"):
                months_row_idx = r
                break
        if months_row_idx is None:
            continue
        # locate values row (Revenue/Total/Cost or next immediate row if numeric)
        values_row_idx = None
        for r in range(months_row_idx + 1, min(months_row_idx + 6, len(df))):
            row = df.iloc[r]
            if any(isinstance(val, str) and val.strip().lower() in value_labels for val in row):
                values_row_idx = r
                break
            # Fall back: next row likely the numeric values
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

def build_structure_dict_from_sheet(excel_path: str, sheet_name: str, value_labels: Iterable[str]):
    df = pd.read_excel(excel_path, sheet_name=sheet_name)
    return _extract_blocks_generic(df, value_labels)

def forecast_from_cohorts(schedule: np.ndarray, cohorts: Iterable[Tuple[int, int]], horizon: Optional[int] = None):
    if not isinstance(schedule, np.ndarray):
        schedule = np.array(schedule, dtype=float)
    max_shifted_end = 0
    if horizon is None:
        for start, cnt in cohorts:
            end = int(start) - 1 + len(schedule)
            if end > max_shifted_end:
                max_shifted_end = end
        horizon = max(max_shifted_end, len(schedule))
    out = np.zeros(horizon, dtype=float)
    for start, cnt in cohorts:
        offset = int(start) - 1
        end = offset + len(schedule)
        if end > len(out):
            new_out = np.zeros(end, dtype=float)
            new_out[:len(out)] = out
            out = new_out
        out[offset:end] += schedule * float(cnt)
    return out
