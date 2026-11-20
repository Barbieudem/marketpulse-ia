"""Datos de marketing: inversión por canal, ventas y recorridos de cliente.

Como no hay acceso a Meta Graph API ni a datos internos de una marca, la
inversión por canal se SIMULA con parámetros conocidos ("ground truth") y la
demanda se ancla al interés REAL de Google Trends. Ventaja metodológica: al
conocer los ROAS verdaderos podemos validar que el MMM los recupera.

Ecuación generadora de ventas (semanal):
  Ventas_t = base + β_trend · Interés_t + Σ_c β_c · Hill(Adstock(Gasto_c,t)) + ε_t
  Adstock_t = Gasto_t + λ · Adstock_{t-1}
  Hill(x)   = x^s / (x^s + K^s)
"""
import numpy as np
import pandas as pd

import config

# Parámetros "verdaderos" de la simulación (para validar el MMM)
PARAMS_VERDADEROS = {
    "Meta Ads":    {"lambda": 0.50, "K": 40_000, "s": 1.5, "beta": 150_000},
    "Google Ads":  {"lambda": 0.20, "K": 22_000, "s": 1.3, "beta": 110_000},
    "TikTok Ads":  {"lambda": 0.40, "K": 25_000, "s": 1.8, "beta": 70_000},
    "Influencers": {"lambda": 0.60, "K": 30_000, "s": 2.0, "beta": 60_000},
}


def adstock(x, lam):
    out = np.zeros(len(x))
    acc = 0.0
    for i, v in enumerate(x):
        acc = v + lam * acc
        out[i] = acc
    return out


def hill(x, K, s):
    x = np.maximum(x, 0)
    return x ** s / (x ** s + K ** s)


def simular_marketing(trends: pd.DataFrame, seed=config.SEED):
    rng = np.random.default_rng(seed)
    idx = trends.index
    n = len(idx)
    interes = trends[config.KEYWORD_PRINCIPAL].values.astype(float)
    df = pd.DataFrame(index=idx)
    base_gasto = {"Meta Ads": 20_000, "Google Ads": 16_000, "TikTok Ads": 12_000, "Influencers": 8_000}
    for c in config.CANALES:
        # el gasto sigue parcialmente a la demanda (como hacen las marcas) + ruido + flights
        flights = rng.choice([0.4, 1.0, 1.7], size=n, p=[0.25, 0.5, 0.25])
        g = base_gasto[c] * flights * (0.85 + 0.3 * interes / max(interes.max(), 1))
        df[c] = np.round(g * rng.uniform(0.85, 1.15, n), -2)

    ventas = 120_000 + 2_500 * interes
    for c in config.CANALES:
        p = PARAMS_VERDADEROS[c]
        ventas += p["beta"] * hill(adstock(df[c].values, p["lambda"]), p["K"], p["s"])
    ventas += rng.normal(0, 12_000, n)
    df["interes_trends"] = interes
    df["ventas"] = np.round(ventas, 0)
    df["conversiones"] = np.round(df["ventas"] / config.TICKET_PROMEDIO).astype(int)
    return df


def simular_recorridos(n_clientes=6000, seed=config.SEED):
    """Recorridos multicanal (para modelos de atribución)."""
    rng = np.random.default_rng(seed)
    canales = config.CANALES + ["Orgánico"]
    peso_conv = {"Meta Ads": 0.10, "Google Ads": 0.16, "TikTok Ads": 0.06, "Influencers": 0.08, "Orgánico": 0.04}
    filas = []
    for cid in range(n_clientes):
        largo = rng.integers(1, 6)
        ruta = list(rng.choice(canales, size=largo, p=[0.30, 0.25, 0.20, 0.10, 0.15]))
        prob = 1 - np.prod([1 - peso_conv[c] for c in ruta])
        convirtio = int(rng.random() < prob)
        filas.append({"cliente_id": cid, "ruta": " > ".join(ruta), "convirtio": convirtio,
                      "valor": float(config.TICKET_PROMEDIO * rng.uniform(0.6, 1.8)) if convirtio else 0.0})
    return pd.DataFrame(filas)
