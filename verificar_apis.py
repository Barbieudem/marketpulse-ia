"""Auditoría previa a la defensa (Fase 5). Ejecuta: python verificar_apis.py"""
import config
import gemini_client

print("== Gemini ==")
ok, msg = gemini_client.verificar()
print("OK" if ok else "FALLA", "-", msg[:200])

print("\n== YouTube Data API ==")
import youtube_comments
if not youtube_comments.disponible():
    print("FALLA - falta YOUTUBE_API_KEY")
else:
    try:
        v = youtube_comments.buscar_videos(config.MARCA, 1)
        c = youtube_comments.comentarios_video(v[0]["video_id"], 3) if v else []
        print("OK -", v[0]["titulo"][:60] if v else "sin videos", "|", len(c), "comentarios de prueba")
    except Exception as e:
        print("FALLA -", e)

print("\n== Google Trends (PyTrends) ==")
try:
    from pytrends.request import TrendReq
    py = TrendReq(hl="es-MX", tz=360, timeout=(10, 30))
    py.build_payload([config.KEYWORD_PRINCIPAL], geo=config.GEO, timeframe="today 3-m")
    df = py.interest_over_time()
    print("OK -", len(df), "semanas recibidas")
except Exception as e:
    print("FALLA (normal si hay 429; la app usa caché) -", e)

print("\n== Caché de datos ==")
print("trends_cache.csv:", (config.DATA_DIR / "trends_cache.csv").exists())
print("resultados.pkl:", (config.OUT_DIR / "resultados.pkl").exists())
