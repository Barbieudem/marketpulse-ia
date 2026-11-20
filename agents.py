"""Arquitectura multi-agente con Gemini (Opción 4) + function calling.

  ┌──────────────┐   tools: resumen_tendencias()
  │ Analista de  │──────────────────────────────┐
  │ Tendencias   │                              │
  └──────────────┘                              ▼
  ┌──────────────┐   tools: optimizar_presupuesto(), simular_campana()
  │ Gestor de    │──────────────────────────────┤
  │ Presupuesto  │                              ▼
  └──────────────┘                     ┌────────────────┐
  ┌──────────────┐  (insumos de ambos)  │  Orquestador   │ → Plan de campaña
  │ Copywriter   │─────────────────────▶│  (este módulo) │
  │ Creativo     │                      └────────────────┘
  └──────────────┘
Cada agente es una llamada a Gemini con su propio system_instruction y sus
herramientas (funciones de Python que Gemini invoca automáticamente).

Evaluación: CPA simulado, ROI acumulado y coherencia (¿el plan respeta la
asignación del optimizador?).
"""
import json

import numpy as np

import config
import gemini_client
from metrics import cpa, roi

CTX = {}  # se llena con los resultados del pipeline


def configurar_contexto(resultados: dict):
    CTX.clear()
    CTX.update(resultados)


def _limpio(d):
    """Convierte tipos de numpy a tipos nativos para que Gemini pueda serializarlos."""
    return json.loads(json.dumps(d, default=lambda o: o.item() if hasattr(o, "item") else str(o)))


# ---------------- Herramientas (function calling) ----------------
def resumen_tendencias() -> dict:
    """Devuelve el estado actual de las tendencias de búsqueda de la categoría:
    crecimiento semanal, aceleración, fase pronosticada y palabra clave con más momentum."""
    return _limpio(CTX.get("resumen_tendencias", {}))


def optimizar_presupuesto(presupuesto_semanal: float) -> dict:
    """Calcula la asignación óptima del presupuesto semanal (MXN) entre canales usando
    el Marketing Mix Model y devuelve gasto por canal y ROAS proyectado."""
    mmm, df = CTX["mmm"], CTX["df_marketing"]
    t, res = mmm.optimizar(df, presupuesto=float(presupuesto_semanal))
    return _limpio({"asignacion": {r.canal: round(r.gasto_optimo) for r in t.itertuples()},
            "ROAS_proyectado": round(float(res["ROAS_proyectado_optimo"]), 2),
            "ROAS_mix_actual": round(float(res["ROAS_actual"]), 2)})


def simular_campana(presupuesto_semanal: float, semanas: int = 4) -> dict:
    """Simula una campaña de N semanas con la asignación óptima y devuelve ventas,
    conversiones, CPA simulado y ROI acumulado."""
    mmm, df = CTX["mmm"], CTX["df_marketing"]
    t, res = mmm.optimizar(df, presupuesto=float(presupuesto_semanal))
    rng = np.random.default_rng(config.SEED)
    ventas_sem = res["ventas_incrementales_optimo"] * rng.normal(1, 0.08, int(semanas))
    inversion = float(presupuesto_semanal) * int(semanas)
    ventas = float(ventas_sem.sum())
    conv = ventas / config.TICKET_PROMEDIO
    return _limpio({"semanas": int(semanas), "inversion_total": round(inversion), "ventas_incrementales": round(ventas),
            "conversiones": round(conv), "CPA_simulado": round(cpa(inversion, conv), 2),
            "ROI_acumulado_%": round(100 * roi(ventas, inversion, config.MARGEN_BRUTO), 1),
            "ROAS": round(ventas / inversion, 2)})


# ---------------- Definición de agentes ----------------
AGENTES = {
    "Analista de Tendencias": {
        "sistema": (f"Eres el Analista de Tendencias de {config.MARCA} ({config.GIRO}). Usa la herramienta "
                    "resumen_tendencias y explica en 5 viñetas qué está pasando con la demanda, la fase "
                    "de la tendencia y la oportunidad. Usa números concretos."),
        "tools": [resumen_tendencias],
    },
    "Gestor de Presupuesto": {
        "sistema": (f"Eres el Gestor de Presupuesto de {config.MARCA}. Usa optimizar_presupuesto y "
                    "simular_campana con el presupuesto indicado. Entrega una tabla de asignación por canal, "
                    "ROAS proyectado, CPA simulado y ROI acumulado, y justifica cada decisión."),
        "tools": [optimizar_presupuesto, simular_campana],
    },
    "Copywriter Creativo": {
        "sistema": (f"Eres el Copywriter de {config.MARCA}, marca mexicana. Con el análisis de tendencias, "
                    "presupuesto y sentimiento que recibes, escribe: 1 concepto creativo, 3 copies para Meta Ads, "
                    "2 guiones de 15s para TikTok y 3 asuntos de email. Tono cercano, español de México."),
        "tools": None,
    },
}


def ejecutar_agente(nombre, mensaje):
    a = AGENTES[nombre]
    return gemini_client.generar(mensaje, sistema=a["sistema"], tools=a["tools"])


def orquestar(presupuesto_semanal=config.PRESUPUESTO_SEMANAL_TOTAL, semanas=4, objetivo="ventas"):
    salida = {}
    salida["Analista de Tendencias"] = ejecutar_agente(
        "Analista de Tendencias", f"Analiza las tendencias actuales para una campaña de {objetivo}.")
    salida["Gestor de Presupuesto"] = ejecutar_agente(
        "Gestor de Presupuesto",
        f"Presupuesto semanal: {presupuesto_semanal} MXN durante {semanas} semanas. Objetivo: {objetivo}.")
    sentimiento = CTX.get("resumen_sentimiento", {})
    salida["Copywriter Creativo"] = ejecutar_agente(
        "Copywriter Creativo",
        f"ANÁLISIS DE TENDENCIAS:\n{salida['Analista de Tendencias']}\n\n"
        f"PLAN DE PRESUPUESTO:\n{salida['Gestor de Presupuesto']}\n\n"
        f"SENTIMIENTO DE MARCA: {sentimiento}")
    return salida, evaluar(presupuesto_semanal, semanas)


def evaluar(presupuesto_semanal, semanas):
    """Métricas objetivas del plan (no dependen del texto de Gemini)."""
    sim = simular_campana(presupuesto_semanal, semanas)
    opt = optimizar_presupuesto(presupuesto_semanal)
    suma = sum(opt["asignacion"].values())
    return {**sim, "ROAS_proyectado": opt["ROAS_proyectado"], "ROAS_mix_actual": opt["ROAS_mix_actual"],
            "coherencia_presupuesto_%": round(100 * (1 - abs(suma - presupuesto_semanal) / presupuesto_semanal), 2)}
