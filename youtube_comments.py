"""Comentarios reales de YouTube en tiempo real (YouTube Data API v3).

Sustituye a Meta Graph API: no requiere cuenta publicitaria, solo una API key
de Google Cloud con "YouTube Data API v3" habilitada (variable YOUTUBE_API_KEY).

Flujo por marca:
  1. search.list   → videos más relevantes en México sobre la marca (100 unidades de cuota)
  2. commentThreads.list → comentarios principales de cada video (1 unidad c/u)
Cuota gratuita: 10,000 unidades al día (alcanza para ~80 búsquedas diarias).
"""
import json
import os
import urllib.error
import urllib.parse
import urllib.request

import pandas as pd

import config

API = "https://www.googleapis.com/youtube/v3/"
CACHE = config.DATA_DIR / "comentarios_youtube.csv"


def disponible():
    return bool(os.getenv("YOUTUBE_API_KEY"))


def _get(endpoint, **params):
    params["key"] = os.getenv("YOUTUBE_API_KEY")
    url = API + endpoint + "?" + urllib.parse.urlencode(params)
    with urllib.request.urlopen(url, timeout=30) as r:
        return json.loads(r.read().decode("utf-8"))


def buscar_videos(marca, n=5):
    data = _get("search", part="snippet", q=marca, type="video", maxResults=n,
                regionCode=config.GEO, relevanceLanguage="es", order="relevance")
    return [{"video_id": it["id"]["videoId"], "titulo": it["snippet"]["title"],
             "canal": it["snippet"]["channelTitle"]} for it in data.get("items", [])]


def comentarios_video(video_id, n=20):
    try:
        data = _get("commentThreads", part="snippet", videoId=video_id, maxResults=n,
                    order="relevance", textFormat="plainText")
    except urllib.error.HTTPError:  # comentarios desactivados en ese video
        return []
    out = []
    for it in data.get("items", []):
        s = it["snippet"]["topLevelComment"]["snippet"]
        out.append({"fecha": s.get("publishedAt", "")[:10], "texto": s.get("textDisplay", "").strip(),
                    "likes": int(s.get("likeCount", 0))})
    return out


def descargar(marcas=None, videos_por_marca=5, comentarios_por_video=20, guardar=True):
    """Devuelve DataFrame: fecha, marca, fuente, texto, likes."""
    marcas = marcas or config.KEYWORDS
    filas = []
    for m in marcas:
        for v in buscar_videos(m, videos_por_marca):
            for c in comentarios_video(v["video_id"], comentarios_por_video):
                if c["texto"]:
                    filas.append({"fecha": c["fecha"], "marca": m,
                                  "fuente": f"YouTube: {v['titulo'][:80]}", "texto": c["texto"][:500],
                                  "likes": c["likes"]})
    df = pd.DataFrame(filas)
    if guardar and len(df):
        df.to_csv(CACHE, index=False)
    return df
