import os
import re
import pandas as pd
import sqlite3

INPUT_DIR = r"."                 
OUT_DB = "converted_data.db"     

SUPPORTED_EXT = (".xlsx", ".xls", ".csv")

def normalize_table_name(filename: str) -> str:
    name = os.path.splitext(os.path.basename(filename))[0].lower()
    name = re.sub(r"[^a-z0-9]+", "_", name).strip("_")
    return name if name else "table"

def read_any(path: str) -> pd.DataFrame | None:
    ext = os.path.splitext(path)[1].lower()
    if ext in (".xlsx", ".xls"):
        return pd.read_excel(path)
    if ext == ".csv":
        return pd.read_csv(path)
    return None

tables_written = 0

with sqlite3.connect(OUT_DB) as conn:
    for fname in sorted(os.listdir(INPUT_DIR)):
        fpath = os.path.join(INPUT_DIR, fname)
        if not os.path.isfile(fpath):
            continue
        if not fname.lower().endswith(SUPPORTED_EXT):
            continue

        df = read_any(fpath)
        if df is None:
            continue

        df = df.drop(columns=[c for c in df.columns if str(c).startswith("Unnamed")], errors="ignore")
        df = df.dropna(axis=1, how="all")

        table = normalize_table_name(fname)
        df.to_sql(table, conn, if_exists="replace", index=False)
        tables_written += 1

print(f"OK -> {OUT_DB} | tables: {tables_written}")
