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


def clean_col(col):
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
    if pd.isna(s):
        return None
    return str(s).lower().replace("okres ", "").strip()


with sqlite3.connect(DB_PATH) as conn:
    edata = pd.read_sql("SELECT * FROM edata_okresy", conn)
    volby = pd.read_sql("SELECT * FROM volenia_a_participacia_okres", conn)


edata.columns = [clean_col(c) for c in edata.columns]
volby.columns = [clean_col(c) for c in volby.columns]

edata["okres_name"] = edata["územná jednotka"].apply(clean_name)
volby["okres_name"] = volby["názov okresu"].apply(clean_name)

volby = volby[volby["okres_name"] != "zahraničie"]

df = edata.merge(volby, on="okres_name", how="inner")
df = df.groupby("okres_name", as_index=False).first()


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


X = df[FEATURES]
y = df["turnout"]


X_ols = sm.add_constant(X)
ols = sm.OLS(y, X_ols).fit()
print(ols.summary())



vif = pd.DataFrame({
    "feature": X.columns,
    "VIF": [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
}).sort_values("VIF", ascending=False)

print(vif)


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

lasso = LassoCV(cv=5).fit(X_scaled, y)
lasso_coef = pd.Series(lasso.coef_, index=X.columns)

print(lasso_coef[lasso_coef != 0].sort_values())

X_std = pd.DataFrame(
    scaler.fit_transform(X),
    columns=X.columns,
    index=X.index
)

X_std_ols = sm.add_constant(X_std)
ols_std = sm.OLS(y, X_std_ols).fit()
print(ols_std.summary())


final_features = [
    "urban",
    "unemployment",
    "unemployment_sq",
    "edu_uni",
    "edu_none",
    "avg_wage",
    "foreigners"
]

Xf = sm.add_constant(df[final_features])
ols_restricted = sm.OLS(y, Xf).fit()
print(ols_restricted.summary())

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
