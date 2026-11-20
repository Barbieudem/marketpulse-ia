"""Ingesta de Google Trends.

Orden de prioridad (para que la demo NUNCA falle en vivo):
  1. data/trends_cache.csv         → datos ya descargados (caché)
  2. data/google_trends_export.csv → CSV exportado a mano desde trends.google.com
  3. PyTrends (API no oficial)     → con reintentos y espera ante error 429
     (con --descargar se intenta PyTrends primero y el CSV queda de respaldo)
  4. Serie de muestra sintética    → SOLO para probar el código (se marca como tal)

Nota: PyTrends fue archivado en abril de 2025 y Google responde con HTTP 429
con frecuencia; por eso guardamos caché y aceptamos el CSV manual.
"""
import io
import time

import numpy as np
import pandas as pd

import config

CACHE = config.DATA_DIR / "trends_cache.csv"
MANUAL = config.DATA_DIR / "google_trends_export.csv"


def _desde_pytrends(keywords, geo, timeframe, reintentos=4):
    from pytrends.request import TrendReq  # import tardío: es opcional

    ultimo_error = None
    for intento in range(reintentos):
        try:
            py = TrendReq(hl="es-MX", tz=360, timeout=(10, 30))
            frames = []
            # Google Trends permite máx. 5 términos por consulta
            for i in range(0, len(keywords), 5):
                py.build_payload(keywords[i:i + 5], geo=geo, timeframe=timeframe)
                df = py.interest_over_time()
                frames.append(df.drop(columns=["isPartial"], errors="ignore"))
                time.sleep(5)
            df = pd.concat(frames, axis=1)
            df.index.name = "fecha"
            return df
        except Exception as e:  # 429 u otros
            ultimo_error = e
            espera = 30 * (intento + 1)
            print(f"[PyTrends] intento {intento + 1} falló ({e}); reintento en {espera}s")
            time.sleep(espera)
    raise RuntimeError(f"PyTrends no respondió: {ultimo_error}")


def leer_csv_google_trends(ruta_o_buffer):
    """Lee el CSV que descarga el botón 'Exportar' de trends.google.com."""
    if hasattr(ruta_o_buffer, "read"):
        texto = ruta_o_buffer.read()
        texto = texto.decode("utf-8-sig") if isinstance(texto, bytes) else texto
    else:
        texto = open(ruta_o_buffer, encoding="utf-8-sig").read()
    lineas = texto.splitlines()
    # Encuentra la fila de encabezado (la primera seguida de una fila con fecha)
    inicio = 0
    for i, l in enumerate(lineas[:-1]):
        sig = lineas[i + 1].split(",")[0]
        if "," in l and pd.to_datetime(sig, errors="coerce") is not pd.NaT:
            inicio = i
            break
    df = pd.read_csv(io.StringIO("\n".join(lineas[inicio:])))
    df = df.rename(columns={df.columns[0]: "fecha"})
    df["fecha"] = pd.to_datetime(df["fecha"])
    df = df.set_index("fecha")
    df.columns = [c.split(":")[0].strip() for c in df.columns]
    df = df.replace("<1", 0.5).apply(pd.to_numeric, errors="coerce")
    return df


def serie_muestra(keywords, semanas=260, seed=config.SEED):
    """Serie sintética SOLO para pruebas de código (estacionalidad + tendencia)."""
    rng = np.random.default_rng(seed)
    fechas = pd.date_range(end=pd.Timestamp.today().normalize(), periods=semanas, freq="W-SUN")
    t = np.arange(semanas)
    data = {}
    for j, kw in enumerate(keywords):
        fase = rng.uniform(0, 2 * np.pi)
        tendencia = 30 + (0.08 + 0.04 * j) * t
        estacional = 15 * np.sin(2 * np.pi * t / 52 + fase)
        picos = np.zeros(semanas)
        for p in rng.choice(semanas, 3, replace=False):
            picos += 25 * np.exp(-0.5 * ((t - p) / 2.5) ** 2)
        y = tendencia + estacional + picos + rng.normal(0, 4, semanas)
        data[kw] = y
    df = pd.DataFrame(data, index=fechas)
    df = (100 * df / df.values.max()).clip(lower=0).round(0)
    df.index.name = "fecha"
    return df


def cargar_trends(keywords=None, geo=None, timeframe=None, forzar_descarga=False):
    """Devuelve (DataFrame semanal, fuente)."""
    keywords = keywords or config.KEYWORDS
    geo = geo or config.GEO
    timeframe = timeframe or config.TIMEFRAME

    if CACHE.exists() and not forzar_descarga:
        df = pd.read_csv(CACHE, parse_dates=["fecha"], index_col="fecha")
        fuente = open(config.DATA_DIR / "trends_fuente.txt").read().strip() \
            if (config.DATA_DIR / "trends_fuente.txt").exists() else "caché"
        return df, fuente

    df = None
    # Si existe el CSV exportado a mano y no se pidió descarga, se usa directo (rápido y seguro)
    if MANUAL.exists() and not forzar_descarga:
        df = leer_csv_google_trends(MANUAL)
        fuente = "Google Trends (CSV exportado de trends.google.com)"
    else:
        try:
            df = _desde_pytrends(keywords, geo, timeframe)
            fuente = f"Google Trends vía PyTrends ({geo}, {timeframe})"
        except Exception as e:
            print(f"[Trends] {e}")
            if MANUAL.exists():
                df = leer_csv_google_trends(MANUAL)
                fuente = "Google Trends (CSV exportado de trends.google.com)"
    if df is None:
        df = serie_muestra(keywords)
        fuente = "MUESTRA SINTÉTICA (reemplazar con datos reales antes de entregar)"

    df.to_csv(CACHE)
    (config.DATA_DIR / "trends_fuente.txt").write_text(fuente)
    return df, fuente
