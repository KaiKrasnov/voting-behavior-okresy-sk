import sqlite3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA


DB_PATH = "converted_data.db"
K_RANGE = range(2, 9)
RANDOM_STATE = 83

#casti od Maksa:
#K-means robime na socio-ekonomických alebo demografickych charakteristikach; nie na ucasti, nie na hlasoch vo volbach
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
    "foreigners",
]

#vo vizualizáciách hlasov nechceme mať vela stran, berieme len tie vyznamnejsie
PRAH_VYRAZNEJ_STRANY = 5.0 #priemer >=5%
MIN_POCET_STRAN = 7 #keby bolo malo berieme aspoň 7

MAP_STRANY = {
    "Slovenská ľudová strana Andreja Hlinku": "sls",
    "DOBRÁ VOĽBA": "dobra_volba",
    "Sloboda a Solidarita": "sas",
    "SME RODINA": "sme_rodina",
    "Slovenské Hnutie Obrody": "sho",
    "ZA ĽUDÍ": "za_ludi",
    "MÁME TOHO DOSŤ !": "dost",
    "Slovenská národná strana": "sns",
    "Demokratická strana": "demokraticka",
    "OBYČAJNÍ ĽUDIA a nezávislé osobnosti (OĽANO), NOVA, Kresťanská únia (KÚ), ZMENA ZDOLA": "olano",
    "Koalícia Progresívne Slovensko a SPOLU - občianska demokracia": "ps_spolu",
    "STAROSTOVIA A NEZÁVISLÍ KANDIDÁTI": "starostovia",
    "99 % - občiansky hlas": "devdev_perc",
    "Kresťanskodemokratické hnutie": "kdh",
    "Slovenská liga": "slovenska_liga",
    "VLASŤ": "vlast",
    "MOST - HÍD": "most_hid",
    "SMER - sociálna demokracia": "smer",
    "SOLIDARITA - Hnutie pracujúcej chudoby": "solidarita",
    "HLAS ĽUDU": "hlas",
    "Magyar Közösségi Összefogás - Maďarská komunitná spolupatričnosť": "madari",
    "Práca slovenského národa": "praca_slov",
    "Kotlebovci - Ľudová strana Naše Slovensko": "lsns",
    "Socialisti.sk": "socialisti",
}


#moje:
#konverzia cisel
def to_num(x):
    if pd.isna(x):
        return np.nan
    return pd.to_numeric(str(x).replace(",", "."), errors="coerce")


#cistenie dat + normalitacia nazvov - podobne ako Maks
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

def clean_okres_name(x) -> str | None:
    if pd.isna(x):
        return None
    s = str(x).strip().lower()
    s = s.replace("okres ", "")
    return s


def plot_xy(x, y, best_x, xlabel, ylabel, title, path):
    #krivka + zvislá čiara na najlepšie k
    plt.figure(figsize=(7, 4))
    plt.plot(list(x), y, marker="o")
    plt.axvline(best_x, linestyle="--")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(alpha=0.25)
    plt.savefig(path)
    plt.close()

#podobne ako od Maksa:
def load_edata_okresy(db_path: str) -> pd.DataFrame:
    with sqlite3.connect(db_path) as conn:
        edata = pd.read_sql("SELECT * FROM edata_okresy", conn)

    edata.columns = [clean_col(c) for c in edata.columns]
    edata["okres_name"] = edata["územná jednotka"].apply(clean_okres_name)
    return edata


def load_volby_a_okresy(db_path: str) -> pd.DataFrame:
    with sqlite3.connect(db_path) as conn:
        volby = pd.read_sql("SELECT * FROM volenia_a_participacia_okres", conn)

    volby.columns = [clean_col(c) for c in volby.columns]
    volby["okres_name"] = volby["názov okresu"].apply(clean_okres_name)
    volby = volby[volby["okres_name"] != "zahraničie"].copy()
    return volby

#podobné ako od Maxa:
def rename_and_engineer(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns={
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
        "cudzinci pct": "foreigners",
        }
    )
    #ciselna konverzia
    for c in ["population", "age_65_plus", "age_0_14", "female_share", "working_pensioners",
              "working_share", "urban", "edu_uni", "edu_none", "unemployment", "avg_wage", "foreigners"]:
        if c in df.columns:
            df[c] = df[c].apply(to_num)

    df["log_population"] = np.log(df["population"].replace(0, np.nan)) #davame si pozor na 0, aby nevzniklo -inf
    df["unemployment_sq"] = df["unemployment"] ** 2 #skusam aj nelinearny vztah nezamestnanosti
    return df

def build_votes_wide(volby_long: pd.DataFrame, mapa_stran: dict) -> pd.DataFrame:
    col_strana = "názov politického subjektu"
    col_podiel = "podiel platných hlasov v pct"
    col_turnout = "účasť voličov_x000d_ v pct"

    #účasť pre okres (opakuje sa rovnaka hodnota pre vsetky riadky okresu, tak ber prvu nenulovú)
    turnout_df = volby_long[["okres_name", col_turnout]].copy()
    turnout_df = turnout_df.dropna(subset=[col_turnout])
    turnout_df = turnout_df.groupby("okres_name", as_index=False).first()
    turnout_df = turnout_df.rename(columns={col_turnout: "turnout"})
    turnout_df["turnout"] = turnout_df["turnout"].apply(to_num)

    #hlasy: strana - skratka; pivot do wide formatu
    v = volby_long.dropna(subset=[col_strana, col_podiel]).copy()
    v["kod_strany"] = v[col_strana].map(mapa_stran)
    v[col_podiel] = v[col_podiel].apply(to_num)

    wide = v.pivot_table(index="okres_name", columns="kod_strany", values=col_podiel, aggfunc="first",).reset_index()

    #premenuj stlpec podiel hlasov
    for c in wide.columns:
        if c != "okres_name":
            wide = wide.rename(columns={c: f"podiel_hlasov_{c}"})

    #turnout pridaj
    return wide.merge(turnout_df, on="okres_name", how="left")

#K-means + PCA
def naj_k_pre_siluetu(X_scaled: np.ndarray, k_range) -> tuple[int, list[float], list[float]]:
    inertie = [] #vnutrozhlukova suma stvorcov vzdialenosti
    siluety = []

    for k in k_range: #viac startov - stabilnejsie
        k_mean = KMeans(n_clusters=k, init="k-means++", n_init=20, random_state=RANDOM_STATE)
        labels = k_mean.fit_predict(X_scaled)
        inertie.append(k_mean.inertia_)
        siluety.append(silhouette_score(X_scaled, labels))

    naj_index = int(np.argmax(siluety))
    naj_k = list(k_range)[naj_index]
    return naj_k, inertie, siluety


def pick_significant_parties(df: pd.DataFrame) -> list[str]:
    cols = [c for c in df.columns if c.startswith("podiel_hlasov_")]

    priemery = df[cols].mean(numeric_only=True).sort_values(ascending=False)
    vyber = priemery[priemery >= PRAH_VYRAZNEJ_STRANY].index.tolist()

    if len(vyber) < MIN_POCET_STRAN:
        vyber = priemery.head(MIN_POCET_STRAN).index.tolist()
    return vyber

def plot_pca(df: pd.DataFrame, path: str, title: str):
    plt.figure(figsize=(7, 5))
    for cl in sorted(df["cluster"].unique()):
        sub = df[df["cluster"] == cl]
        plt.scatter(sub["PC1"], sub["PC2"], s=25, alpha=0.85, label=f"cluster {cl}")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title(title)
    plt.grid(alpha=0.25)
    plt.legend()
    plt.savefig(path)
    plt.close()

def plot_parties_by_cluster(cluster_mean: pd.DataFrame, vote_cols: list[str], path: str):
    vote_cols = [c for c in vote_cols if c in cluster_mean.columns]
    x = np.arange(len(cluster_mean.index))
    m = len(vote_cols)
    w = 0.8 / max(m, 1)

    plt.figure(figsize=(10, 5))
    for i, c in enumerate(vote_cols):
        plt.bar(x + (i - (m - 1) / 2) * w, cluster_mean[c].values, width=w, label=c.replace("podiel_hlasov_", ""))

    plt.xticks(x, [str(c) for c in cluster_mean.index])
    plt.xlabel("cluster")
    plt.ylabel("priemerný podiel hlasov v %")
    plt.title("výrazné strany podľa zhluku - priemery")
    plt.grid(axis="y", alpha=0.25)
    plt.legend(ncol=2, fontsize=9)
    plt.savefig(path)
    plt.close()


def main():
    edata = rename_and_engineer(load_edata_okresy(DB_PATH)) #charakteristiky okresov
    df_feat = edata[["okres_name"] + FEATURES].dropna(subset=FEATURES).copy()
    X = df_feat[FEATURES].to_numpy()
    scaler = StandardScaler() #standardizacia
    X_scaled = scaler.fit_transform(X)

    naj_k, inertie, siluety = naj_k_pre_siluetu(X_scaled, K_RANGE)
    plot_xy(K_RANGE, inertie, naj_k, "k", "Inertia SSE", "K-means elbow", "kmeans_elbow.png")
    plot_xy(K_RANGE, siluety,  naj_k, "k", "Silhouette score", "K-means silhouette", "kmeans_silhouette.png")

    #finalny kmeans:
    kmeans = KMeans(n_clusters=naj_k, init="k-means++", n_init=50, max_iter=500, random_state=RANDOM_STATE)
    labels = kmeans.fit_predict(X_scaled)
    df_feat["cluster"] = labels

    #priemery socio-ekonomických znakov podla zhluku
    feat_mean = df_feat.groupby("cluster")[FEATURES].mean(numeric_only=True).round(3)
    feat_mean.to_csv("cluster_profiles_features.csv")

    #v štandardizovaných jednotkách - viem porovnať medzi premennými
    df_scaled = pd.DataFrame(X_scaled, columns=FEATURES, index=df_feat.index)
    df_scaled["cluster"] = labels
    feat_mean_z = df_scaled.groupby("cluster")[FEATURES].mean().round(3)
    feat_mean_z.to_csv("cluster_profiles_features_zscores.csv")

    #najviac odlišné znaky medzi zhlukmi (max-min cez clustre)
    rozptyl = (feat_mean_z.max(axis=0) - feat_mean_z.min(axis=0)).sort_values(ascending=False)
    rozptyl.head(10).to_csv("top_features_by_cluster_diff.csv")
    print("Top znaky podľa rozdielu medzi zhlukmi:\n", rozptyl.head(10))

    #PCA - na vizualizaciu
    X_pca = PCA(n_components=2, random_state=RANDOM_STATE).fit_transform(X_scaled)
    df_feat["PC1"] = X_pca[:, 0]
    df_feat["PC2"] = X_pca[:, 1]
    plot_pca(df_feat, "pca_cluster.png", f"PCA zhluky k={naj_k}")

    volby_dlhsie = load_volby_a_okresy(DB_PATH)
    volby_sirsie = build_votes_wide(volby_dlhsie, MAP_STRANY)
    df_spolu = df_feat.merge(volby_sirsie, on="okres_name", how="left")

    #priemery podla zhluku: turnout + podiely hlasov
    vote_cols_all = [c for c in df_spolu.columns if c.startswith("podiel_hlasov_")]
    cols_pre_priemer = (["turnout"] if "turnout" in df_spolu.columns else []) + vote_cols_all
    cluster_mean = df_spolu.groupby("cluster")[cols_pre_priemer].mean(numeric_only=True).sort_index()
    cluster_mean.round(4).to_csv("cluster_profiles_votes_full.csv", index=True)

    vyrazne_strany = pick_significant_parties(df_spolu)
    plot_parties_by_cluster(cluster_mean, vyrazne_strany, "votes_selected_parties_by_cluster.png")
    #okres -> cluster
    df_feat[["okres_name", "cluster"]].sort_values(["cluster", "okres_name"]).to_csv("district_cluster_assignments.csv", index=False)

    print("Okresy:", len(df_feat))
    print("Vybrané k - max silhouette:", naj_k)
    print("Silhouette:", float(silhouette_score(X_scaled, labels)))
    print("Počet iterácií do konvergencie:", kmeans.n_iter_)
    print("Inertia SSE:", float(kmeans.inertia_))
    print("Zhluky a ich veľkosti:\n", df_feat["cluster"].value_counts().sort_index())
    if vyrazne_strany:
        print("Strany v grafe:", [c.replace("podiel_hlasov_", "") for c in vyrazne_strany])


if __name__ == "__main__":
    main()
