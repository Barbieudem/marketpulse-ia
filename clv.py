"""Customer Lifetime Value (CLV) + segmentación RFM.

1) CLV histórico:  CLV_hist_i = Σ compras_i · margen
2) CLV predictivo (fórmula clásica con retención):
     CLV = (Ticket · Frecuencia · Margen) · r / (1 + d - r)
     r = tasa de retención anual, d = tasa de descuento anual
3) CLV con ML: Random Forest, Red Neuronal y KNN predicen el gasto de los
   próximos 6 meses a partir de RFM del periodo anterior (validación temporal),
   y se comparan con Diebold-Mariano.
"""
import numpy as np
import pandas as pd

import config
from metrics import todas_las_metricas


def rfm(tx, fecha_corte):
    t = tx[tx["fecha"] < fecha_corte]
    g = t.groupby("cliente_id").agg(ultima=("fecha", "max"), primera=("fecha", "min"),
                                    frecuencia=("fecha", "count"), monetario=("monto", "sum"))
    g["recencia_dias"] = (fecha_corte - g["ultima"]).dt.days
    g["antiguedad_dias"] = (fecha_corte - g["primera"]).dt.days
    g["ticket_prom"] = g["monetario"] / g["frecuencia"]
    return g.drop(columns=["ultima", "primera"])


def segmentar(r):
    r = r.copy()
    r["R"] = pd.qcut(r["recencia_dias"].rank(method="first"), 4, labels=[4, 3, 2, 1]).astype(int)
    r["F"] = pd.qcut(r["frecuencia"].rank(method="first"), 4, labels=[1, 2, 3, 4]).astype(int)
    r["M"] = pd.qcut(r["monetario"].rank(method="first"), 4, labels=[1, 2, 3, 4]).astype(int)
    score = r["R"] + r["F"] + r["M"]
    r["segmento"] = np.select([score >= 10, score >= 8, score >= 6, r["R"] <= 1],
                              ["Campeones", "Leales", "Potenciales", "En riesgo"], "Hibernando")
    return r


def clv_formula(tx, margen=config.MARGEN_BRUTO, descuento=0.10):
    fin = tx["fecha"].max()
    anio1 = tx[(tx["fecha"] > fin - pd.Timedelta(days=730)) & (tx["fecha"] <= fin - pd.Timedelta(days=365))]
    anio2 = tx[tx["fecha"] > fin - pd.Timedelta(days=365)]
    c1, c2 = set(anio1["cliente_id"]), set(anio2["cliente_id"])
    r = len(c1 & c2) / len(c1) if c1 else 0.5
    ticket = anio2["monto"].mean()
    frec = anio2.groupby("cliente_id").size().mean()
    valor_anual = ticket * frec * margen
    clv = valor_anual * r / (1 + descuento - r)
    out = {"retencion_anual": r, "ticket_promedio": ticket, "frecuencia_anual": frec,
           "margen": margen, "tasa_descuento": descuento, "valor_anual_cliente": valor_anual, "CLV": clv}
    return {k: float(v) for k, v in out.items()}


def dataset_clv_ml(tx, meses_obj=6):
    fin = tx["fecha"].max()
    corte = fin - pd.DateOffset(months=meses_obj)
    X = rfm(tx, corte)
    fut = tx[(tx["fecha"] >= corte)].groupby("cliente_id")["monto"].sum()
    y = fut.reindex(X.index).fillna(0.0) * config.MARGEN_BRUTO
    return X, y


def entrenar_clv_ml(tx, seed=config.SEED):
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.neural_network import MLPRegressor
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    X, y = dataset_clv_ml(tx)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=seed)
    modelos = {
        "Random Forest": RandomForestRegressor(n_estimators=300, min_samples_leaf=5, random_state=seed),
        "Red Neuronal (MLP)": make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=(32, 16), alpha=1e-2, max_iter=3000,
                                                                          early_stopping=True, random_state=seed)),
        "KNN": make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors=15, weights="distance")),
    }
    preds, filas = {}, []
    for n, m in modelos.items():
        m.fit(X_tr, y_tr)
        p = np.clip(m.predict(X_te), 0, None)
        preds[n] = p
        filas.append({"modelo": n, **todas_las_metricas(y_te, p)})
    return {"metricas": pd.DataFrame(filas), "predicciones": preds, "y_test": y_te.values,
            "modelo_rf": modelos["Random Forest"], "X": X}
