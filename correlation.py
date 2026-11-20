"""Correlación de Pearson con significancia.

  r = Σ (x - x̄)(y - ȳ) / sqrt( Σ (x - x̄)² · Σ (y - ȳ)² )
  t = r·sqrt(n-2)/sqrt(1-r²)  ~ t(n-2)   →  p-valor
"""
import itertools

import pandas as pd
from scipy.stats import pearsonr


def matriz_pearson(df):
    cols = df.columns
    r = pd.DataFrame(index=cols, columns=cols, dtype=float)
    p = pd.DataFrame(index=cols, columns=cols, dtype=float)
    for a, b in itertools.product(cols, cols):
        res = pearsonr(df[a], df[b])
        r.loc[a, b], p.loc[a, b] = res.statistic, res.pvalue
    return r, p


def pares_significativos(r, p, objetivo, alpha=0.05):
    filas = []
    for c in r.columns:
        if c == objetivo:
            continue
        rv = r.loc[c, objetivo]
        fuerza = "fuerte" if abs(rv) >= 0.7 else "moderada" if abs(rv) >= 0.4 else "débil"
        filas.append({"variable": c, "r_Pearson": round(rv, 3), "p_valor": float(p.loc[c, objetivo]),
                      "significativa": bool(p.loc[c, objetivo] < alpha),
                      "interpretación": f"Correlación {fuerza} {'positiva' if rv > 0 else 'negativa'}"})
    return pd.DataFrame(filas).sort_values("r_Pearson", key=abs, ascending=False)
