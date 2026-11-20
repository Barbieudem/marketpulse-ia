"""Marketing Mix Modeling (Opción 2) + optimización de presupuesto con SciPy.

Modelo paramétrico estimado por mínimos cuadrados no lineales:
  Ventas_t = b0 + b_trend·Interés_t + Σ_c β_c · Hill(Adstock_c,t ; K_c, s_c) + ε_t

ROAS histórico por canal  = Σ_t contribución_c,t / Σ_t gasto_c,t
ROAS marginal             = ∂Ventas/∂Gasto_c  (ingreso extra por $1 adicional)

Optimización (estado estacionario, adstock_ss = gasto/(1-λ)):
  max_{g}  Σ_c β_c·Hill(g_c/(1-λ_c))
  s.a.     Σ_c g_c = Presupuesto,   0.3·ḡ_c ≤ g_c ≤ 2.5·ḡ_c
Resuelto con scipy.optimize.minimize (SLSQP).
"""
import numpy as np
import pandas as pd
from scipy.optimize import least_squares, minimize

import config
from data_marketing import adstock, hill


class MMM:
    def __init__(self, canales=None):
        self.canales = canales or config.CANALES
        self.params = None

    # vector θ = [b0, b_trend, (λ, K, s, β) × canales]
    def _desempacar(self, th):
        b0, bt = th[0], th[1]
        p = {}
        for i, c in enumerate(self.canales):
            lam, K, s, beta = th[2 + 4 * i: 6 + 4 * i]
            p[c] = {"lambda": float(lam), "K": float(K), "s": float(s), "beta": float(beta)}
        return b0, bt, p

    def _contribuciones(self, df, p):
        return pd.DataFrame({c: p[c]["beta"] * hill(adstock(df[c].values, p[c]["lambda"]), p[c]["K"], p[c]["s"])
                             for c in self.canales}, index=df.index)

    def _pred(self, th, df):
        b0, bt, p = self._desempacar(th)
        return b0 + bt * df["interes_trends"].values + self._contribuciones(df, p).sum(axis=1).values

    def fit(self, df):
        y = df["ventas"].values
        x0, lo, hi = [y.mean() * 0.5, 0.0], [0, -1e5], [y.max() * 2, 1e5]
        for c in self.canales:
            m = df[c].mean()
            x0 += [0.4, 1.5 * m, 1.5, 4 * m]
            lo += [0.0, 0.5 * m, 1.0, 0]
            hi += [0.85, 4 * m, 2.5, 15 * m]
        r = least_squares(lambda th: self._pred(th, df) - y, x0, bounds=(lo, hi), max_nfev=4000)
        self.theta = r.x
        self.b0, self.b_trend, self.params = self._desempacar(r.x)
        return self

    def predict(self, df):
        return self._pred(self.theta, df)

    def descomposicion(self, df):
        con = self._contribuciones(df, self.params)
        con.insert(0, "Base + Tendencia", self.b0 + self.b_trend * df["interes_trends"].values)
        return con

    def tabla_roas(self, df):
        con = self._contribuciones(df, self.params)
        filas = []
        for c in self.canales:
            g = df[c].mean()
            filas.append({"canal": c, "gasto_total": df[c].sum(), "ventas_atribuidas": con[c].sum(),
                          "ROAS": con[c].sum() / df[c].sum(),
                          "ROAS_marginal": self._roas_marginal(c, g),
                          "adstock_lambda": self.params[c]["lambda"], "K_saturacion": self.params[c]["K"]})
        return pd.DataFrame(filas)

    def _respuesta_ss(self, c, g):
        p = self.params[c]
        return p["beta"] * hill(np.array([g / (1 - p["lambda"])]), p["K"], p["s"])[0]

    def _roas_marginal(self, c, g, eps=100):
        return (self._respuesta_ss(c, g + eps) - self._respuesta_ss(c, g)) / eps

    def optimizar(self, df, presupuesto=config.PRESUPUESTO_SEMANAL_TOTAL):
        medias = np.array([df[c].mean() for c in self.canales])
        actual = medias / medias.sum() * presupuesto  # mismo mix histórico, escalado

        def neg_ventas(g):
            return -sum(self._respuesta_ss(c, gi) for c, gi in zip(self.canales, g))

        bounds = [(0.3 * m, 2.5 * m) for m in medias]
        cons = {"type": "eq", "fun": lambda g: g.sum() - presupuesto}
        r = minimize(neg_ventas, actual, method="SLSQP", bounds=bounds, constraints=[cons])
        opt = r.x
        filas = []
        for c, a, o in zip(self.canales, actual, opt):
            filas.append({"canal": c, "gasto_actual": a, "gasto_optimo": o,
                          "cambio_%": 100 * (o - a) / a,
                          "ventas_inc_actual": self._respuesta_ss(c, a),
                          "ventas_inc_optimo": self._respuesta_ss(c, o)})
        t = pd.DataFrame(filas)
        resumen = {
            "presupuesto": presupuesto,
            "ventas_incrementales_actual": t["ventas_inc_actual"].sum(),
            "ventas_incrementales_optimo": t["ventas_inc_optimo"].sum(),
            "ROAS_actual": t["ventas_inc_actual"].sum() / presupuesto,
            "ROAS_proyectado_optimo": t["ventas_inc_optimo"].sum() / presupuesto,
            "mejora_%": 100 * (t["ventas_inc_optimo"].sum() / t["ventas_inc_actual"].sum() - 1),
            "convergio": bool(r.success),
        }
        resumen = {k: (float(v) if isinstance(v, (np.floating, float, int)) and not isinstance(v, bool) else v)
                   for k, v in resumen.items()}
        return t, resumen

    def curvas_respuesta(self, df, puntos=40):
        filas = []
        for c in self.canales:
            for g in np.linspace(0, 3 * df[c].mean(), puntos):
                filas.append({"canal": c, "gasto": g, "ventas_incrementales": self._respuesta_ss(c, g)})
        return pd.DataFrame(filas)


def features_mmm_ml(df, params):
    """Variables para los modelos ML del MMM (adstock con λ estimados)."""
    X = pd.DataFrame(index=df.index)
    for c in config.CANALES:
        X[f"adstock_{c}"] = adstock(df[c].values, params[c]["lambda"])
    X["interes_trends"] = df["interes_trends"].values
    semana = df.index.isocalendar().week.astype(int).values
    X["sem_sin"] = np.sin(2 * np.pi * semana / 52)
    X["sem_cos"] = np.cos(2 * np.pi * semana / 52)
    return X
