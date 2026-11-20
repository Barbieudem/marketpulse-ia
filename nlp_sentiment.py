"""Análisis de sentimiento (NLP) con Gemini sobre comentarios reales de YouTube.

Fuentes, en orden:
  1. data/comentarios_youtube.csv  → descargados en tiempo real con YouTube Data API
  2. data/comentarios.csv          → comentarios de EJEMPLO (solo si no hay YouTube)

Gemini clasifica cada comentario (negativo / neutral / positivo, score −1 a 1 y tema)
en lotes de 40. El resultado se guarda en data/comentarios_analizados.csv para que
la app no tenga que volver a llamar a Gemini en cada arranque.
Si una fila trae 'etiqueta_manual', se valida a Gemini contra la etiqueta humana.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score

import config
import gemini_client, youtube_comments

EJEMPLO = config.DATA_DIR / "comentarios.csv"
ANALIZADOS = config.DATA_DIR / "comentarios_analizados.csv"
ETIQUETAS = ["negativo", "neutral", "positivo"]

POS = {"encanta", "excelente", "bueno", "buena", "recomiendo", "increíble", "funciona", "suave", "perfecto",
       "feliz", "mejor", "genial", "hidratada", "rápido", "amo", "calidad", "love", "great", "good"}
NEG = {"malo", "mala", "caro", "irritó", "irritación", "alergia", "tardó", "nunca", "peor", "decepción",
       "grasosa", "roto", "devolución", "queja", "horrible", "ardor", "bad", "worst"}

SISTEMA = ("Eres un analista de marca experto en NLP en español de México. Clasifica cada comentario de "
           "cliente sobre marcas de skincare. Devuelve SOLO JSON: una lista de objetos {id, sentimiento: "
           "negativo|neutral|positivo, score: número de -1 a 1, tema: precio|calidad|resultados en piel|"
           "irritación o alergia|disponibilidad|comparación con otra marca|otro}.")


def cargar_comentarios(descargar=False):
    """Devuelve (df, fuente)."""
    if youtube_comments.disponible() and (descargar or not youtube_comments.CACHE.exists()):
        try:
            df = youtube_comments.descargar()
            if len(df):
                return df, "YouTube Data API (tiempo real)"
        except Exception as e:
            print(f"[YouTube] {e}")
    if youtube_comments.CACHE.exists():
        return pd.read_csv(youtube_comments.CACHE), "YouTube Data API (caché)"
    df = pd.read_csv(EJEMPLO)
    if "marca" not in df:
        df["marca"] = config.MARCA
    return df, "COMENTARIOS DE EJEMPLO (configura YOUTUBE_API_KEY para datos reales)"


def _lexico(texto):
    palabras = set(str(texto).lower().replace(",", " ").replace(".", " ").split())
    s = len(palabras & POS) - len(palabras & NEG)
    return ("positivo" if s > 0 else "negativo" if s < 0 else "neutral"), float(np.tanh(s))


def analizar(df, lote=40):
    df = df.copy().reset_index(drop=True)
    df["id"] = df.index
    df["sentimiento"], df["score"], df["tema"] = None, np.nan, None
    usado = "Léxico (respaldo)"
    if gemini_client.disponible():
        for i in range(0, len(df), lote):
            parte = df.iloc[i:i + lote][["id", "texto"]].to_dict("records")
            res = gemini_client.generar_json(f"Comentarios:\n{parte}", sistema=SISTEMA)
            if isinstance(res, list):
                for r in res:
                    try:
                        k = int(r["id"])
                        df.loc[k, "sentimiento"] = str(r.get("sentimiento", "neutral")).lower()
                        df.loc[k, "score"] = float(r.get("score", 0))
                        df.loc[k, "tema"] = r.get("tema", "otro")
                    except Exception:
                        pass
        if df["sentimiento"].notna().mean() > 0.5:
            usado = f"Gemini ({config.GEMINI_MODEL})"
    faltan = df["sentimiento"].isna()
    if faltan.any():
        lex = df.loc[faltan, "texto"].apply(_lexico)
        df.loc[faltan, "sentimiento"] = [o[0] for o in lex]
        df.loc[faltan, "score"] = [o[1] for o in lex]
        df.loc[faltan, "tema"] = "otro"
    return df, usado


def analizar_con_cache(df, fuente, forzar=False):
    if ANALIZADOS.exists() and not forzar:
        prev = pd.read_csv(ANALIZADOS)
        if len(prev) == len(df) and "motor" in prev and str(prev["motor"].iloc[0]).startswith("Gemini"):
            return prev, prev["motor"].iloc[0]
    out, motor = analizar(df)
    if motor.startswith("Gemini"):
        out.assign(motor=motor, fuente_datos=fuente).to_csv(ANALIZADOS, index=False)
    return out, motor


def validar(df):
    if "etiqueta_manual" not in df or df["etiqueta_manual"].isna().all():
        return None
    d = df.dropna(subset=["etiqueta_manual"])
    y, p = d["etiqueta_manual"].str.lower(), d["sentimiento"]
    return {"cm": confusion_matrix(y, p, labels=ETIQUETAS), "etiquetas": ETIQUETAS,
            "F1_macro": f1_score(y, p, average="macro", labels=ETIQUETAS, zero_division=0),
            "accuracy": float((y == p).mean())}


def indice_sentimiento_neto(df):
    """ISN = (%positivos - %negativos), rango -100 a 100."""
    v = df["sentimiento"].value_counts(normalize=True)
    return float(100 * (v.get("positivo", 0) - v.get("negativo", 0)))


def por_marca(df):
    t = (pd.crosstab(df["marca"], df["sentimiento"], normalize="index") * 100).reindex(columns=ETIQUETAS, fill_value=0)
    t["comentarios"] = df.groupby("marca").size()
    t["ISN"] = t["positivo"] - t["negativo"]
    return t.round(1).sort_values("ISN", ascending=False)
