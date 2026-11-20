"""Modelos de atribución multicanal.

Heurísticos (reparten el valor de cada conversión entre los puntos de contacto):
  Primer clic, Último clic, Lineal, Decaimiento temporal (vida media 1 paso),
  Basado en posición (40% primero, 40% último, 20% intermedios).
Basado en datos:
  Cadena de Markov — efecto de eliminación (removal effect):
     RE_c = 1 - P(conv | sin canal c) / P(conv)
     Crédito_c = RE_c / Σ RE · Conversiones totales
"""
from collections import defaultdict

import numpy as np
import pandas as pd


def _rutas(df):
    return [r.split(" > ") for r in df["ruta"]]


def heuristicos(df):
    conv = df[df["convirtio"] == 1]
    modelos = {k: defaultdict(float) for k in
               ["Primer clic", "Último clic", "Lineal", "Decaimiento temporal", "Basado en posición"]}
    for ruta in _rutas(conv):
        n = len(ruta)
        modelos["Primer clic"][ruta[0]] += 1
        modelos["Último clic"][ruta[-1]] += 1
        for c in ruta:
            modelos["Lineal"][c] += 1 / n
        pesos = np.array([0.5 ** (n - 1 - i) for i in range(n)])
        pesos /= pesos.sum()
        for c, w in zip(ruta, pesos):
            modelos["Decaimiento temporal"][c] += w
        if n == 1:
            modelos["Basado en posición"][ruta[0]] += 1
        elif n == 2:
            modelos["Basado en posición"][ruta[0]] += 0.5
            modelos["Basado en posición"][ruta[1]] += 0.5
        else:
            modelos["Basado en posición"][ruta[0]] += 0.4
            modelos["Basado en posición"][ruta[-1]] += 0.4
            for c in ruta[1:-1]:
                modelos["Basado en posición"][c] += 0.2 / (n - 2)
    return pd.DataFrame(modelos).fillna(0)


def _transiciones(df):
    cuentas = defaultdict(lambda: defaultdict(float))
    for ruta, conv in zip(_rutas(df), df["convirtio"]):
        estados = ["INICIO"] + ruta + (["CONVERSION"] if conv else ["NULO"])
        for a, b in zip(estados[:-1], estados[1:]):
            cuentas[a][b] += 1
    return {a: {b: v / sum(d.values()) for b, v in d.items()} for a, d in cuentas.items()}


def _p_conversion(T, quitar=None, iters=200):
    estados = set(T) | {b for d in T.values() for b in d}
    p = {s: 0.0 for s in estados}
    p["CONVERSION"] = 1.0
    for _ in range(iters):
        for s in T:
            if s == quitar:
                p[s] = 0.0
                continue
            p[s] = sum(w * (0.0 if b == quitar else p[b]) for b, w in T[s].items())
    return p["INICIO"]


def markov(df):
    T = _transiciones(df)
    base = _p_conversion(T)
    canales = sorted({c for r in _rutas(df) for c in r})
    re = {c: 1 - _p_conversion(T, quitar=c) / base for c in canales}
    total = df["convirtio"].sum()
    s = sum(re.values())
    return pd.Series({c: re[c] / s * total for c in canales}, name="Markov"), re, base


def tabla_atribucion(df):
    h = heuristicos(df)
    m, re, base = markov(df)
    t = h.join(m, how="outer").fillna(0)
    pct = (t / t.sum() * 100).round(1)
    return t, pct, re, base
