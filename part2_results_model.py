import os
import sqlite3
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GroupKFold
from sklearn.base import BaseEstimator


class Results:
    def __init__(self, db_path: os.PathLike) -> None:
        """
        self.x, self.y a self.w funguju tak, ze self.w[i] je podiel hlasov v okrese
        s parametrami self.x[i] pre politicku stranu self.y[i].
        Mozno ich predstavit ako mnoziny stlpcov dlhej tabulky tvaru
        parameter okresa 1 | ... | parameter okresa n | politicka strana | podiel hlasov.
        """
        if not os.path.isfile(db_path):
            raise FileNotFoundError("Database was not found")

        try:
            self.conn = sqlite3.connect(db_path)
            self.x, self.y, self.w = self.prepare_data()

        except sqlite3.Error as e:
            print("Database error:", e)
            raise

    def prepare_edata(self) -> pd.DataFrame:
        edata: pd.DataFrame = pd.read_sql_query(
            "SELECT * FROM edata_okresy;",
            self.conn
        )
        edata["Názov okresu"] = edata["Územná jednotka"].str[6:]
        edata = edata.sort_values(by="Názov okresu")
        edata = edata.set_index("Názov okresu")
        edata = edata.sort_index(axis=1)
        edata = edata.drop(columns=["Kód", "Územná jednotka"])
        return edata

    def prepare_elections_results(self) -> pd.DataFrame:
        elections_results: pd.DataFrame = pd.read_sql_query(
            """WITH sucet_hlasov AS (
                SELECT `Názov okresu`, SUM(`Počet platných hlasov`) AS `Súčet`
                FROM volenia_a_participacia_okres
                WHERE `Názov okresu` <> 'Zahraničie'
                GROUP BY `Názov okresu`
            )
            SELECT v.`Názov okresu`,
                v.`Názov politického subjektu`,
                v.`Počet platných hlasov` / s.`súčet` AS `Podiel`
            FROM volenia_a_participacia_okres v
                JOIN sucet_hlasov s ON v.`Názov okresu` = s.`Názov okresu`
            WHERE v.`Názov okresu` <> 'Zahraničie';""",
            self.conn,
        )
        elections_results = elections_results.sort_values(by=["Názov okresu", "Názov politického subjektu"])
        elections_results = elections_results.set_index("Názov okresu")
        return elections_results
    
    def prepare_data(self) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
        edata = self.prepare_edata()
        elections_results = self.prepare_elections_results()

        x: pd.DataFrame = edata.loc[elections_results.index]
        y: pd.Series = elections_results["Názov politického subjektu"]
        w: pd.Series = elections_results["Podiel"]
        
        return x, y, w

    def find_kfold_loss(self, x: pd.DataFrame, n_splits: int = 5) -> float:
        model = Pipeline([
            ("scaler", StandardScaler()),
            ("logreg", LogisticRegression(
                solver="lbfgs",
                max_iter=500,
            ))
        ])

        groups = x.index

        kfold = GroupKFold(n_splits=n_splits)

        losses = []

        for train_idx, test_idx in kfold.split(x, self.y, groups):
            x_train = x.iloc[train_idx]
            y_train = self.y.iloc[train_idx]
            w_train = self.w.iloc[train_idx]

            x_test = x.iloc[test_idx]
            y_test = self.y.iloc[test_idx]
            w_test = self.w.iloc[test_idx]

            model.fit(x_train, y_train, logreg__sample_weight=w_train)

            y_pred = model.predict_proba(x_test)

            log_loss = []
            num_parties = len(model.named_steps["logreg"].classes_)

            for i in range(len(x_test.index.unique())):
                t = w_test[num_parties*i : num_parties*(i+1)].tolist()
                p = y_pred[i]
                s = -np.sum([ti * np.log(pi+1e-12) for ti, pi in zip(t, p)])
                log_loss.append(s)
            
            losses.append(np.mean(log_loss))

        return np.mean(losses)

    def classification(self, criteria: pd.Series = None) -> tuple[Pipeline, pd.DataFrame, float]:
        """
        Pomocou multitriednej logistickej regresie a normalizacie vytvara model, ktory na vstupe
        ma parametry okresa (ako DataFrame rozmeru 1 x n_parametrov) a na vystupe je podiely hlasov
        politickych stran vnutri okresa s takymito parametrami.
        Ak `criteria` je None, trenuje model na vsetkych parametroch okresu.
        Inak trenuje iba na vybratych parametroch zo zoznamu `criteria`.
        Vracia natrenovany model, coeficienty modelu (ako DataFrame, kde riadky su parametre
        okresa, stlpce su politicke strany) a priemerny log-loss pre okresy.
        """
        model = Pipeline([
            ("scaler", StandardScaler()),
            ("logreg", LogisticRegression(
                solver="lbfgs",
                max_iter=500,
            ))
        ])

        if criteria is None:
            x = self.x
        else:
            mask = (self.x.columns.isin(criteria))
            x = self.x.loc[:, mask]
        
        model.fit(x, self.y, logreg__sample_weight=self.w)

        coefs = pd.DataFrame(
            model.named_steps["logreg"].coef_.T,
            index=x.columns,
            columns=model.named_steps["logreg"].classes_
        )
        kfold_loss = self.find_kfold_loss(x)

        return model, coefs, kfold_loss
    
    def find_most_impactful(self, coefs: pd.DataFrame, top_k: int = 10) -> pd.Series:
        """
        Pre DataFrame `coefs` formatu z metody Results.classification() hlada `top_k` parametrov,
        ktore maju najvacsi vplyv na model. Vplyv na model je priemer absolutnych hodnot
        koeficientov parametra pre rozne politicke strany.
        """
        importance = coefs.abs().mean(axis=1)
        return importance.sort_values(ascending=False).head(top_k)

    def predict_one(self, model: BaseEstimator, data: list) -> pd.Series:
        """
        Pre model z metody Results.classification() vracia predikciu pre vstup `data`.
        """
        input_data = pd.DataFrame(data).T
        output = model.predict_proba(input_data)[0]
        output = pd.Series(output, index=model.named_steps["logreg"].classes_)
        return output


db_path = "converted_data.db"
results = Results(db_path)
_, coefs, _ = results.classification()
impactful = results.find_most_impactful(coefs)
model, new_coefs, kfold_loss = results.classification(impactful.index)

test_okres = "Bratislava I"
index_mask = (results.x.index == test_okres)
column_mask = (results.x.columns.isin(impactful.index))

test_data = results.x.loc[index_mask, column_mask].iloc[0]

pred = results.predict_one(model, test_data)
was_true = pd.Series(results.w.loc[index_mask].values, index=results.y.loc[index_mask].values)
comparison = pd.DataFrame({
    "pred": pred,
    "was true": was_true
})

print(comparison)