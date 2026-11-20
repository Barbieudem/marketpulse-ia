# MarketPulse IA — Inteligencia de marketing para CeraVe

Proyecto final · Métodos Cuantitativos para Negocios e IA en Mercadotecnia.
Combina las **Opciones 2, 4, 6 y 7**: una app web publicada en **Google Cloud Run** que pronostica
la demanda de búsqueda (Google Trends), clasifica la fase de la tendencia, optimiza el presupuesto
publicitario (MMM), calcula CLV y atribución, analiza sentimiento con Gemini y genera planes de
campaña con un sistema **multi-agente de Gemini**.

## Modelos y métricas incluidos

| Requisito del curso | Dónde está |
|---|---|
| Random Forest, Red Neuronal (MLP), KNN — regresión | `src/models_ml.py` (pronóstico de tendencia, ventas MMM, CLV) |
| Random Forest, Red Neuronal, KNN — clasificación | Fase Crecimiento / Estancamiento / Caída |
| Simulación de hiperparámetros | `GridSearchCV` + `TimeSeriesSplit` (tablas en `outputs/resultados.md`) |
| Distancia Euclidiana | KNN con `metric="euclidean"` sobre variables estandarizadas |
| Test Diebold-Mariano | `src/diebold_mariano.py` (con corrección HLN), aplicado a tendencia, fase, MMM y CLV |
| RMSE, MSE, MAE, MAPE, SMAPE | `src/metrics.py` |
| Matriz de confusión, F1 | Clasificación de fase y validación del NLP |
| Marketing Mix Modeling (adstock + saturación Hill) | `src/mmm.py` |
| Optimización de presupuesto (SciPy) | `MMM.optimizar` (SLSQP) |
| ROAS, ROAS marginal, CPA, ROI | `src/metrics.py`, `src/mmm.py`, `src/agents.py` |
| Customer Lifetime Value | `src/clv.py` (fórmula con retención + RFM + ML) |
| Modelos de atribución | `src/attribution.py` (primer/último clic, lineal, decaimiento, posición, **Markov**) |
| Correlación de Pearson | `src/correlation.py` (r y p-valor) |
| Crecimiento semanal y aceleración de tendencia | `src/features.py` |
| NLP / análisis de sentimiento | `src/nlp_sentiment.py` (Gemini) |
| Agentes con roles + Tool Use | `src/agents.py` + `agentes_gemini_enterprise.md` |
| API de Gemini | `src/gemini_client.py` (AI Studio o Vertex AI) |
| PyTrends | `src/data_trends.py` (con reintentos, caché y CSV manual) |

## Fuentes de datos

- **Google Trends (real):** interés semanal en México de CeraVe vs La Roche-Posay, Cetaphil, Neutrogena y Nivea (5 años).
- **Transacciones (real):** UCI *Online Retail II* (e-commerce, CC BY 4.0) como referencia para CLV, porque los datos de clientes de CeraVe no son públicos.
- **Comentarios de clientes:** reseñas públicas recolectadas a mano (sin Meta API).
- **Inversión por canal:** **simulada** con parámetros conocidos y anclada al interés real de Trends. Como conocemos el ROAS verdadero, podemos comprobar qué tan bien lo recupera el MMM.

---

## Paso a paso

> 📘 **Guía detallada para principiantes:** `GUIA_PASO_A_PASO.pdf` (en este repositorio).

### 1. Datos reales de Google Trends (10 min)
1. Abre: https://trends.google.com/trends/explore?date=today%205-y&geo=MX&q=CeraVe,La%20Roche-Posay,Cetaphil,Neutrogena,Nivea
2. En la gráfica "Interés a lo largo del tiempo", da clic en **Descargar (CSV)**.
3. Guarda el archivo como `data/google_trends_export.csv`.

El pipeline usa ese CSV automáticamente. Para intentar PyTrends en vivo: `python run_pipeline.py --descargar`
(si Google responde error 429, usa el CSV como respaldo).

### 2. Transacciones para CLV (automático)
El pipeline descarga solo el dataset UCI Online Retail II la primera vez (requiere internet, p. ej. Cloud Shell).

### 3. Comentarios reales (20 min)
Copia 40–100 reseñas públicas de productos CeraVe (Amazon México, Liverpool, Sephora, comentarios
públicos de Instagram/TikTok) en `data/comentarios.csv` con columnas `fecha,fuente,texto,etiqueta_manual`.
En `etiqueta_manual` pon tú `positivo`, `neutral` o `negativo`: así validamos a Gemini con F1.

### 4. Llave de Gemini
Copia `.env.example` como `.env` y pon tu `GEMINI_API_KEY` (https://aistudio.google.com/apikey),
o configura Vertex AI con tus créditos de Google Cloud for Education.

### 5. Correr todo
```bash
pip install -r requirements.txt
python run_pipeline.py        # entrena modelos, genera outputs/resultados.md y gráficas
python verificar_apis.py      # auditoría de llaves (hazlo antes de la defensa)
streamlit run app.py          # abre la app en tu navegador
```
(En Google Colab: sube la carpeta y usa `!pip install -r requirements.txt` y `!python run_pipeline.py`.)

### 6. Subir a GitHub
Crea un repositorio **público**, sube toda la carpeta. El `.gitignore` evita que se suba tu `.env`.

### 7. Publicar en Google Cloud Run (link público)
En https://console.cloud.google.com abre **Cloud Shell** (icono `>_`) y ejecuta:
```bash
git clone https://github.com/TU_USUARIO/marketpulse-ia.git
cd marketpulse-ia
gcloud services enable run.googleapis.com cloudbuild.googleapis.com artifactregistry.googleapis.com
gcloud run deploy marketpulse-ia --source . --region us-central1 \
  --allow-unauthenticated --memory 2Gi --min-instances 1 \
  --set-env-vars GEMINI_API_KEY=TU_LLAVE,GEMINI_MODEL=gemini-3.5-flash
```
Al terminar te da una URL `https://marketpulse-ia-xxxx.run.app`: ese es tu **link público**.
`--min-instances 1` evita que la app tarde en arrancar durante la defensa; después de calificarte,
bájalo a 0 para no gastar créditos. Configura también **Facturación → Presupuestos y alertas**.

### 8. Antes de la defensa
- Abre la pestaña **🔐 Estado APIs** y da clic en "Probar Gemini ahora".
- Ten abierta la app y el repositorio en pestañas separadas.
- Los datos se leen de caché: la demo no depende de que PyTrends responda en vivo.

## Estructura
```
config.py               marca, keywords, canales, presupuesto
run_pipeline.py         ejecuta todo y genera resultados + gráficas
app.py                  app Streamlit (Cloud Run)
verificar_apis.py       auditoría de llaves
src/                    módulos (datos, modelos, métricas, agentes)
data/                   datos y caché
outputs/                resultados.md y figuras/
```
