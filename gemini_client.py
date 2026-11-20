"""Cliente único de Gemini (SDK google-genai).

Funciona en dos modos, según variables de entorno (.env):
  - Google AI Studio:  GEMINI_API_KEY=...
  - Vertex AI (créditos Google Cloud for Education):
        GOOGLE_GENAI_USE_VERTEXAI=true
        GOOGLE_CLOUD_PROJECT=tu-proyecto
        GOOGLE_CLOUD_LOCATION=us-central1
Si no hay credenciales, las funciones devuelven un texto marcado como
"[modo sin conexión]" para que la app nunca truene en la demo.
"""
import json
import os

import config

_client = None


def disponible():
    return bool(os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
                or os.getenv("GOOGLE_GENAI_USE_VERTEXAI", "").lower() == "true")


def cliente():
    global _client
    if _client is None:
        from google import genai
        _client = genai.Client()  # lee las variables de entorno
    return _client


def generar(prompt, sistema=None, tools=None, json_mode=False, temperatura=0.4):
    if not disponible():
        return "[modo sin conexión] Configura GEMINI_API_KEY o Vertex AI para obtener el análisis de Gemini."
    from google.genai import types

    cfg = types.GenerateContentConfig(
        system_instruction=sistema, temperature=temperatura, tools=tools,
        response_mime_type="application/json" if json_mode else None,
    )
    ultimo = None
    for modelo in [config.GEMINI_MODEL] + config.GEMINI_FALLBACK_MODELS:
        try:
            r = cliente().models.generate_content(model=modelo, contents=prompt, config=cfg)
            return r.text
        except Exception as e:  # modelo no disponible en la región, cuota, etc.
            ultimo = e
    return f"[error Gemini] {ultimo}"


def generar_json(prompt, sistema=None):
    txt = generar(prompt, sistema=sistema, json_mode=True, temperatura=0.0)
    try:
        return json.loads(txt)
    except Exception:
        return None


def verificar():
    """Auditoría previa a la defensa: confirma llave, modelo y cuota."""
    if not disponible():
        return False, "Sin credenciales de Gemini"
    r = generar("Responde solo con la palabra OK.")
    return ("OK" in r.upper() and not r.startswith("[")), r
