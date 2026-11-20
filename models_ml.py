"""Modelos de Machine Learning: Random Forest, Red Neuronal (MLP) y KNN.

- Regresión: pronóstico de interés semanal (y de ventas en el MMM).
- Clasificación: fase de la tendencia (Crecimiento / Estancamiento / Caída).
- Simulación de hiperparámetros con validación cruzada temporal (TimeSeriesSplit).

KNN usa distancia Euclidiana:  d(p, q) = sqrt( Σ_i (p_i - q_i)² )  sobre
variables estandarizadas (por eso va dentro de un Pipeline con StandardScaler).
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score)
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import config
from metrics import todas_las_metricas

CLASES = ["Caída", "Estancamiento", "Crecimiento"]

GRIDS_REG = {
    "Random Forest": (RandomForestRegressor(random_state=config.SEED),
                      {"m__n_estimators": [100, 300], "m__max_depth": [3, 6, None],
                       "m__min_samples_leaf": [1, 3]}),
    "Red Neuronal (MLP)": (MLPRegressor(max_iter=3000, early_stopping=True, random_state=config.SEED),
                           {"m__hidden_layer_sizes": [(16,), (32, 16)], "m__alpha": [1e-3, 1e-2],
                            "m__learning_rate_init": [1e-3, 1e-2]}),
    "KNN": (KNeighborsRegressor(metric="euclidean"),
            {"m__n_neighbors": [3, 5, 7, 11], "m__weights": ["uniform", "distance"]}),
}

GRIDS_CLF = {
    "Random Forest": (RandomForestClassifier(random_state=config.SEED, class_weight="balanced"),
                      {"m__n_estimators": [100, 300], "m__max_depth": [3, 6, None],
                       "m__min_samples_leaf": [1, 3]}),
    "Red Neuronal (MLP)": (MLPClassifier(max_iter=3000, early_stopping=True, random_state=config.SEED),
                           {"m__hidden_layer_sizes": [(16,), (32, 16)], "m__alpha": [1e-3, 1e-2]}),
    "KNN": (KNeighborsClassifier(metric="euclidean"),
            {"m__n_neighbors": [3, 5, 7, 11], "m__weights": ["uniform", "distance"]}),
}


def _pipe(modelo):
    return Pipeline([("sc", StandardScaler()), ("m", modelo)])


def _tabla_hiper(gs, nombre, metrica):
    r = pd.DataFrame(gs.cv_results_)
    cols = [c for c in r.columns if c.startswith("param_")]
    t = r[cols + ["mean_test_score", "std_test_score", "rank_test_score"]].copy()
    t.columns = [c.replace("param_m__", "") for c in t.columns]
    t.insert(0, "modelo", nombre)
    if metrica == "rmse":
        t["mean_test_score"] = -t["mean_test_score"]
        t = t.rename(columns={"mean_test_score": "RMSE_CV", "std_test_score": "std"})
    else:
        t = t.rename(columns={"mean_test_score": "F1_macro_CV", "std_test_score": "std"})
    return t.sort_values("rank_test_score")


def entrenar_regresores(X_tr, y_tr, X_te, y_te, n_splits=4):
    cv = TimeSeriesSplit(n_splits=n_splits)
    preds, metricas, hiper, mejores = {}, [], [], {}
    # Línea base ingenua (random walk): ŷ_t = y_{t-1}
    if "lag_1" in X_te:
        preds["Ingenuo (y_t-1)"] = X_te["lag_1"].values
        metricas.append({"modelo": "Ingenuo (y_t-1)", **todas_las_metricas(y_te, X_te["lag_1"])})
    for nombre, (modelo, grid) in GRIDS_REG.items():
        gs = GridSearchCV(_pipe(modelo), grid, cv=cv, scoring="neg_root_mean_squared_error", n_jobs=-1)
        gs.fit(X_tr, y_tr)
        p = gs.predict(X_te)
        preds[nombre] = p
        mejores[nombre] = {k.replace("m__", ""): v for k, v in gs.best_params_.items()}
        metricas.append({"modelo": nombre, **todas_las_metricas(y_te, p)})
        hiper.append(_tabla_hiper(gs, nombre, "rmse"))
        if nombre == "Random Forest":
            imp = pd.Series(gs.best_estimator_.named_steps["m"].feature_importances_,
                            index=X_tr.columns).sort_values(ascending=False)
    return {"predicciones": preds, "metricas": pd.DataFrame(metricas),
            "hiperparametros": pd.concat(hiper, ignore_index=True), "mejores": mejores,
            "importancias_rf": imp}


def entrenar_clasificadores(X_tr, y_tr, X_te, y_te, n_splits=4):
    cv = TimeSeriesSplit(n_splits=n_splits)
    res, hiper, filas = {}, [], []
    for nombre, (modelo, grid) in GRIDS_CLF.items():
        gs = GridSearchCV(_pipe(modelo), grid, cv=cv, scoring="f1_macro", n_jobs=-1, error_score=0)
        gs.fit(X_tr, y_tr)
        p = gs.predict(X_te)
        etiquetas = [c for c in CLASES if c in set(y_te) | set(p)]
        cm = confusion_matrix(y_te, p, labels=etiquetas)
        res[nombre] = {"pred": p, "cm": cm, "etiquetas": etiquetas,
                       "reporte": classification_report(y_te, p, labels=etiquetas, zero_division=0,
                                                        output_dict=True),
                       "modelo": gs.best_estimator_}
        filas.append({"modelo": nombre, "Accuracy": accuracy_score(y_te, p),
                      "F1_macro": f1_score(y_te, p, average="macro", zero_division=0),
                      "F1_ponderado": f1_score(y_te, p, average="weighted", zero_division=0)})
        hiper.append(_tabla_hiper(gs, nombre, "f1"))
    return {"resultados": res, "metricas": pd.DataFrame(filas),
            "hiperparametros": pd.concat(hiper, ignore_index=True)}


def errores_clasificacion_0_1(y_te, pred):
    """Pérdida 0-1 por semana, para aplicar Diebold-Mariano a los clasificadores."""
    return (np.asarray(y_te) != np.asarray(pred)).astype(float)
