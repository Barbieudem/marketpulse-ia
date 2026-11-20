"""Ingeniería de variables para series de tendencia (Opciones 3 y 7).

  Crecimiento semanal  g_t = (y_t - y_{t-1}) / y_{t-1}
  Aceleración          a_t = g_t - g_{t-1}
  Momentum 4 semanas   m_t = (MA4_t - MA4_{t-4}) / MA4_{t-4}
  Fase (etiqueta):     Crecimiento si m_{t+4} >  +umbral
                       Caída       si m_{t+4} <  -umbral
                       Estancamiento en otro caso
  (la etiqueta mira 4 semanas al FUTURO → el modelo predice la fase próxima)
"""
import numpy as np
import pandas as pd

import config


def crecimiento_y_aceleracion(serie: pd.Series) -> pd.DataFrame:
    s = serie.astype(float).replace(0, np.nan).ffill().fillna(0.5)
    df = pd.DataFrame({"interes": serie.astype(float)})
    df["crec_semanal"] = s.pct_change()
    df["aceleracion"] = df["crec_semanal"].diff()
    ma4 = s.rolling(4).mean()
    df["media_movil_4"] = ma4
    df["momentum_4"] = ma4.pct_change(4)
    return df


def features_supervisadas(serie: pd.Series, n_lags=config.N_LAGS, umbral=config.UMBRAL_FASE):
    base = crecimiento_y_aceleracion(serie)
    X = pd.DataFrame(index=serie.index)
    for k in range(1, n_lags + 1):
        X[f"lag_{k}"] = base["interes"].shift(k)
    X["crec_lag1"] = base["crec_semanal"].shift(1)
    X["acel_lag1"] = base["aceleracion"].shift(1)
    X["ma4_lag1"] = base["media_movil_4"].shift(1)
    X["mom4_lag1"] = base["momentum_4"].shift(1)
    X["std4_lag1"] = base["interes"].rolling(4).std().shift(1)
    semana = serie.index.isocalendar().week.astype(int).values
    X["sem_sin"] = np.sin(2 * np.pi * semana / 52)
    X["sem_cos"] = np.cos(2 * np.pi * semana / 52)

    y_reg = base["interes"]  # pronóstico a 1 semana
    futuro = base["media_movil_4"].shift(-4)
    mom_fut = (futuro - base["media_movil_4"]) / base["media_movil_4"]
    y_clf = pd.Series(np.where(mom_fut > umbral, "Crecimiento",
                               np.where(mom_fut < -umbral, "Caída", "Estancamiento")),
                      index=serie.index)
    y_clf[mom_fut.isna()] = np.nan

    df = X.assign(y_reg=y_reg, y_clf=y_clf).replace([np.inf, -np.inf], np.nan).dropna()
    return df


def split_temporal(df, test_size=config.TEST_SIZE):
    corte = int(len(df) * (1 - test_size))
    return df.iloc[:corte], df.iloc[corte:]
