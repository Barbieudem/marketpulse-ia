"""Test de Diebold-Mariano (1995) con corrección de Harvey, Leybourne y Newbold (1997).

Compara la precisión de dos pronósticos sobre la misma serie.
  H0: ambos modelos tienen la misma precisión  (E[d_t] = 0)
  H1: la precisión es distinta

  e1_t, e2_t = errores de pronóstico de cada modelo
  d_t = L(e1_t) - L(e2_t)    con L = pérdida cuadrática (e²) o absoluta (|e|)
  DM  = d̄ / sqrt( [γ0 + 2 Σ_{k=1}^{h-1} γk] / T )
  HLN: DM* = DM · sqrt( (T + 1 - 2h + h(h-1)/T) / T ),  se compara con t de Student (T-1 g.l.)

Interpretación: DM < 0 y p < 0.05  →  el modelo 1 es significativamente MÁS preciso.
                DM > 0 y p < 0.05  →  el modelo 2 es significativamente más preciso.
"""
import itertools

import numpy as np
import pandas as pd
from scipy import stats


def dm_test(y, pred1, pred2, h=1, loss="cuadratica"):
    y, p1, p2 = (np.asarray(a, float) for a in (y, pred1, pred2))
    e1, e2 = y - p1, y - p2
    if loss == "cuadratica":
        d = e1 ** 2 - e2 ** 2
    else:
        d = np.abs(e1) - np.abs(e2)
    T = len(d)
    d_mean = d.mean()

    def autocov(k):
        return np.sum((d[k:] - d_mean) * (d[: T - k] - d_mean)) / T

    var_d = autocov(0) + 2 * sum(autocov(k) for k in range(1, h))
    if var_d <= 0:
        return {"DM": float("nan"), "p_valor": float("nan")}
    dm = d_mean / np.sqrt(var_d / T)
    hln = np.sqrt((T + 1 - 2 * h + h * (h - 1) / T) / T)
    dm_star = dm * hln
    p = 2 * stats.t.sf(np.abs(dm_star), df=T - 1)
    return {"DM": float(dm_star), "p_valor": float(p)}


def tabla_dm(y, predicciones: dict, h=1, loss="cuadratica"):
    """Aplica DM a todos los pares de modelos. predicciones = {nombre: array}."""
    filas = []
    for (n1, p1), (n2, p2) in itertools.combinations(predicciones.items(), 2):
        r = dm_test(y, p1, p2, h=h, loss=loss)
        if np.isnan(r["p_valor"]):
            concl = "No calculable"
        elif r["p_valor"] >= 0.05:
            concl = "Sin diferencia significativa"
        elif r["DM"] < 0:
            concl = f"{n1} es más preciso"
        else:
            concl = f"{n2} es más preciso"
        filas.append({"Modelo 1": n1, "Modelo 2": n2, "Estadístico DM": round(r["DM"], 3),
                      "p-valor": round(r["p_valor"], 4), "Conclusión (α=0.05)": concl})
    return pd.DataFrame(filas)
