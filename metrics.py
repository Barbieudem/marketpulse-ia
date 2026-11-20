"""Métricas de error de predicción y métricas de marketing.

Fórmulas (y = real, ŷ = pronóstico, n = observaciones):
  MSE   = (1/n) Σ (y - ŷ)²
  RMSE  = √MSE
  MAE   = (1/n) Σ |y - ŷ|
  MAPE  = (100/n) Σ |(y - ŷ) / y|
  SMAPE = (100/n) Σ 2|y - ŷ| / (|y| + |ŷ|)

Marketing:
  ROAS = Ingresos atribuidos / Inversión publicitaria
  CPA  = Inversión / Conversiones
  ROI  = (Ingresos·margen - Inversión) / Inversión
"""
import numpy as np


def mse(y, yhat):
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    return float(np.mean((y - yhat) ** 2))


def rmse(y, yhat):
    return float(np.sqrt(mse(y, yhat)))


def mae(y, yhat):
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    return float(np.mean(np.abs(y - yhat)))


def mape(y, yhat):
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    mask = y != 0
    return float(100 * np.mean(np.abs((y[mask] - yhat[mask]) / y[mask])))


def smape(y, yhat):
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    den = np.abs(y) + np.abs(yhat)
    mask = den != 0
    return float(100 * np.mean(2 * np.abs(y[mask] - yhat[mask]) / den[mask]))


def todas_las_metricas(y, yhat):
    return {
        "MSE": mse(y, yhat),
        "RMSE": rmse(y, yhat),
        "MAE": mae(y, yhat),
        "MAPE_%": mape(y, yhat),
        "SMAPE_%": smape(y, yhat),
    }


def roas(ingresos, inversion):
    return float(ingresos / inversion) if inversion else float("nan")


def cpa(inversion, conversiones):
    return float(inversion / conversiones) if conversiones else float("nan")


def roi(ingresos, inversion, margen=1.0):
    return float((ingresos * margen - inversion) / inversion) if inversion else float("nan")
