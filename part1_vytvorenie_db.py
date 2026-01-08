import os
import numpy as np
import pandas as pd
import sqlite3


INPUT_DIR = r"."                 
OUT_DB = "final.db"              
OUT_XLSX = "final.xlsx"          
TABLE = "data"                   
CODE_COL = "Kód"                 

def read_any(path: str) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    if ext in [".xlsx", ".xls"]:
        return pd.read_excel(path)
    if ext == ".csv":
        return pd.read_csv(path)
    return None

frames = []
for name in sorted(os.listdir(INPUT_DIR)):
    path = os.path.join(INPUT_DIR, name)
    if not os.path.isfile(path):
        continue

    df = read_any(path)
    if df is None:
        continue

    df["__source_file"] = name
    frames.append(df)

if not frames:
    raise SystemExit("No supported files found in folder")

merged = pd.concat(frames, ignore_index=True)

drop_cols = [c for c in merged.columns if str(c).startswith("Unnamed")]
merged = merged.drop(columns=drop_cols, errors="ignore")

merged = merged.dropna(axis=1, how="all")

if CODE_COL not in merged.columns:
    raise SystemExit(f"Column '{CODE_COL}' not found. Available: {list(merged.columns)[:30]} ...")

def merge_group(g: pd.DataFrame) -> pd.Series:
    row = {}
    row[CODE_COL] = g[CODE_COL].iloc[0]

    if "__source_file" in g.columns:
        row["__source_files"] = ";".join(sorted({str(v) for v in g["__source_file"].dropna().unique()}))
-
    for c in g.columns:
        if c in [CODE_COL, "__source_file"]:
            continue
        s = g[c].dropna()
        row[c] = s.iloc[0] if len(s) else np.nan

    return pd.Series(row)

final_df = merged.groupby(CODE_COL, dropna=False).apply(merge_group).reset_index(drop=True)

with sqlite3.connect(OUT_DB) as conn:
    final_df.to_sql(TABLE, conn, if_exists="replace", index=False)

final_df.to_excel(OUT_XLSX, sheet_name=TABLE, index=False)

print(f"OK: {len(final_df):,} rows (unique {CODE_COL}) → {OUT_DB} + {OUT_XLSX}")
