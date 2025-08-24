import pandas as pd

def google_sheet_xls_url() -> str:
    return "https://docs.google.com/spreadsheets/d/11E79iziBz0g4xvppUXDZ_Jd6AoLPhboo0X2L-GG3D_U/export?format=xlsx"

url = google_sheet_xls_url()
df = pd.read_excel(url, sheet_name=0)
print("url succeed")

