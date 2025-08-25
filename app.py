import pandas as pd
import streamlit as st
from sku_engine import build_structure_dict_from_sheet, forecast_from_cohorts

# --- streamlit init

st.set_page_config(page_title="Flint project", layout="wide")
st.title("Flint - data")

def google_sheet_xls_url() -> str:
    return "https://docs.google.com/spreadsheets/d/11E79iziBz0g4xvppUXDZ_Jd6AoLPhboo0X2L-GG3D_U/export?format=xlsx"


WB = google_sheet_xls_url()

rev_dict  = build_structure_dict_from_sheet(WB, "Rev (SKU)",  value_labels=["Revenue"])
cogs_dict = build_structure_dict_from_sheet(WB, "COGS (SKU)", value_labels=["Total", "Cost"])

rev_schedule = rev_dict["Rev-SKU#R10"]
rev_series = forecast_from_cohorts(rev_schedule, cohorts=[(1, 10), (4, 5)], horizon=60)  # np.ndarray

cogs_schedule = cogs_dict["COGS-SKU#C1"]
cogs_series = forecast_from_cohorts(cogs_schedule, cohorts=[(2, 8)], horizon=60)

print(rev_dict)
print(rev_series)
print(cogs_dict)
print(cogs_schedule)
print(cogs_series)