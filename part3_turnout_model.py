import sqlite3
import pandas as pd
import numpy as np

import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

import matplotlib.pyplot as plt


DB_PATH = "converted_data.db"


FEATURES = [
    "age_65_plus",
    "age_0_14",
    "female_share",
    "working_pensioners",
    "working_share",
    "urban",
    "log_population",
    "edu_uni",
    "edu_none",
    "unemployment",
    "unemployment_sq",
    "avg_wage",
    "foreigners"
]

FINAL_FEATURES = [
    "urban",
    "unemployment",
    "unemployment_sq",
    "edu_uni",
    "edu_none",
    "avg_wage",
    "foreigners"
]


def clean_col(col: str) -> str:
    return (
        col.strip()
           .replace("\n", " ")
           .replace("\r", " ")
           .replace("%", "pct")
           .replace("(", "")
           .replace(")", "")
           .replace("–", "-")
           .replace(".", "")
           .lower()
    )


def clean_name(s):
    # as not all names are the same we need to normalize it
    if pd.isna(s):
        return None
    return str(s).lower().replace("okres ", "").strip()


def load_data(db_path: str) -> pd.DataFrame:
    with sqlite3.connect(db_path) as conn:
        edata = pd.read_sql("SELECT * FROM edata_okresy", conn)
        volby = pd.read_sql("SELECT * FROM volenia_a_participacia_okres", conn)

    edata.columns = [clean_col(c) for c in edata.columns]
    volby.columns = [clean_col(c) for c in volby.columns]

    edata["okres_name"] = edata["územná jednotka"].apply(clean_name)
    volby["okres_name"] = volby["názov okresu"].apply(clean_name)

    volby = volby[volby["okres_name"] != "zahraničie"]

    df = edata.merge(volby, on="okres_name", how="inner")
    df = df.groupby("okres_name", as_index=False).first()

    return df


def rename_and_engineer(df: pd.DataFrame) -> pd.DataFrame:
    #take only columns that we need and create easier names for them
    df = df.rename(columns={
        "účasť voličov_x000d_ v pct": "turnout",
        "poproduktívny vek 65 a viac rokov pct": "age_65_plus",
        "predproduktívny vek 0-14 rokov pct": "age_0_14",
        "ženy pct": "female_share",
        "pracujúci dôchodca pct": "working_pensioners",
        "zamestnanec pct": "working_share",
        "spolu": "population",
        "mesto pct": "urban",
        "vysokoškolské vzdelanie pct": "edu_uni",
        "bez školského vzdelania - osoby vo veku 15 rokov a viac pct": "edu_none",
        "miera evidovanej nezamestnanosti v pct": "unemployment",
        "np3110rr_value": "avg_wage",
        "cudzinci pct": "foreigners"
    })

    df["log_population"] = np.log(df["population"])
    df["unemployment_sq"] = df["unemployment"] ** 2

    return df

#main part

def run_ols(X: pd.DataFrame, y: pd.Series):
    #first ols
    X_const = sm.add_constant(X)
    model = sm.OLS(y, X_const).fit()
    print(model.summary())
    return model


def compute_vif(X: pd.DataFrame) -> pd.DataFrame:
    # check correlation between features
    vif = pd.DataFrame({
        "feature": X.columns,
        "VIF": [
            variance_inflation_factor(X.values, i)
            for i in range(X.shape[1])
        ]
    }).sort_values("VIF", ascending=False)

    print(vif)
    return vif


def run_lasso(X: pd.DataFrame, y: pd.Series):
    #lasso for selection of most important features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    lasso = LassoCV(cv=5).fit(X_scaled, y)
    coef = pd.Series(lasso.coef_, index=X.columns)

    print(coef[coef != 0].sort_values())
    return lasso, coef


def run_standardized_ols(X: pd.DataFrame, y: pd.Series):
    # ols on standarted data
    scaler = StandardScaler()
    X_std = pd.DataFrame(
        scaler.fit_transform(X),
        columns=X.columns,
        index=X.index
    )

    X_std_const = sm.add_constant(X_std)
    model = sm.OLS(y, X_std_const).fit()
    print(model.summary())
    return model


def run_restricted_ols(df: pd.DataFrame, y: pd.Series, features: list):
    #ols only on selected features
    X = sm.add_constant(df[features])
    model = sm.OLS(y, X).fit()
    print(model.summary())
    return model


def run_random_forest(X: pd.DataFrame, y: pd.Series):
    # check for a result stability
    rf = RandomForestRegressor(
        n_estimators=500,
        random_state=42
    )

    rf.fit(X, y)

    importances = pd.Series(
        rf.feature_importances_,
        index=X.columns
    ).sort_values(ascending=False)

    print("RF R2 (in-sample):", r2_score(y, rf.predict(X)))
    print(importances)

    importances.head(8).plot(kind="barh")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()

    return rf, importances



def main():
    df = load_data(DB_PATH)
    df = rename_and_engineer(df)

    X = df[FEATURES]
    y = df["turnout"]

    run_ols(X, y)

    compute_vif(X)

    run_lasso(X, y)

    run_standardized_ols(X, y)

    run_restricted_ols(df, y, FINAL_FEATURES)

    run_random_forest(X, y)


if __name__ == "__main__":
    main()

