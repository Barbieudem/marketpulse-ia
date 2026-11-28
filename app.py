"""MarketPulse IA — App interactiva (Streamlit) para publicar en Google Cloud Run.

Ejecutar local:   streamlit run app.py
"""
import pickle

import pandas as pd
import streamlit as st

import config
import agents, gemini_client, nlp_sentiment, youtube_comments

st.set_page_config(page_title=f"MarketPulse IA · {config.MARCA}", page_icon="📊", layout="wide")


@st.cache_resource
def cargar():
    ruta = config.OUT_DIR / "resultados.pkl"
    if not ruta.exists():
        import run_pipeline
        run_pipeline.main()
    with open(ruta, "rb") as f:
        R = pickle.load(f)
    # Si el pipeline corrió sin llave (p. ej. al construir la imagen), completa con Gemini en vivo
    if gemini_client.disponible() and not R["motor_nlp"].startswith("Gemini"):
        com, motor = nlp_sentiment.analizar_con_cache(R["comentarios"][["fecha", "marca", "fuente", "texto"]],
                                                      R["fuente_comentarios"])
        if motor.startswith("Gemini"):
            R["comentarios"], R["motor_nlp"] = com, motor
            R["isn"] = nlp_sentiment.indice_sentimiento_neto(com)
            R["sent_marca"] = nlp_sentiment.por_marca(com)
    if gemini_client.disponible() and R["analisis_gemini"].startswith("["):
        R["analisis_gemini"] = gemini_client.generar(
            f"Interpreta para un director de marketing de {config.MARCA}: fase {R['fase_actual']}, "
            f"métricas {R['reg']['metricas'].round(3).to_dict('records')}, DM {R['dm_reg'].to_dict('records')}, "
            f"ROAS {R['roas'][['canal', 'ROAS', 'ROAS_marginal']].round(2).to_dict('records')}, "
            f"optimización {R['opt_resumen']}, CLV {R['clv_formula']}, ISN {R['isn']:.0f}. 5 recomendaciones.",
            sistema="Consultor senior de marketing cuantitativo. Español de México, conciso.")
    agents.configurar_contexto(R)
    return R


R = cargar()
FIG = config.FIG_DIR


def img(nombre):
    st.image(str(FIG / f"{nombre}.png"), use_container_width=True)


# ---------------- Barra lateral ----------------
st.sidebar.title("📊 MarketPulse IA")
# Identity-Aware Proxy agrega el correo del usuario en este encabezado
try:
    _correo = st.context.headers.get("X-Goog-Authenticated-User-Email", "")
except Exception:
    _correo = ""
if _correo:
    st.sidebar.success(f"🔒 Sesión: {_correo.split(':')[-1]}")
st.sidebar.caption(f"Marca: **{config.MARCA}** · {config.GIRO}")
st.sidebar.markdown(f"**Datos de tendencias:** {R['fuente_trends']}")
st.sidebar.markdown(f"**Transacciones (CLV):** {R['fuente_tx']}")
st.sidebar.markdown("**Inversión por canal:** simulada con parámetros conocidos, anclada a Google Trends")
estado = "🟢 Conectado" if gemini_client.disponible() else "🔴 Sin credenciales"
st.sidebar.markdown(f"**Gemini:** {estado} (`{config.GEMINI_MODEL}`)")

tabs = st.tabs(["🏠 Resumen", "📈 Tendencias", "🤖 Modelos ML", "💰 MMM y Presupuesto",
                "🔗 Atribución y Pearson", "👥 CLV", "💬 Sentimiento", "🧠 Agentes IA", "🔐 Estado APIs"])

# ---------------- Resumen ----------------
with tabs[0]:
    st.title(f"MarketPulse IA — {config.MARCA}")
    if config.VIDEO_INTRO_URL:
        st.subheader("🎬 Video introductorio")
        st.video(config.VIDEO_INTRO_URL)
    st.write("Plataforma de inteligencia de marketing: pronostica la demanda de búsqueda, clasifica la fase "
             "de la tendencia, optimiza el presupuesto entre canales, calcula el valor del cliente y genera "
             "planes de campaña con agentes de Gemini.")
    o = R["opt_resumen"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Fase pronosticada", R["fase_actual"], help=f"Modelo: {R['mejor_clf']}")
    c2.metric("ROAS proyectado (óptimo)", f"{o['ROAS_proyectado_optimo']:.2f}",
              f"{o['mejora_%']:+.1f}% vs mix actual")
    c3.metric("CLV (fórmula)", f"${R['clv_formula']['CLV']:,.0f}")
    c4.metric("Índice de sentimiento neto", f"{R['isn']:.0f}")
    st.subheader("Interpretación automática (Gemini)")
    st.markdown(R["analisis_gemini"])
    if st.button("🔄 Regenerar interpretación con Gemini"):
        with st.spinner("Gemini analizando…"):
            st.markdown(gemini_client.generar(
                f"Interpreta para un director de marketing de {config.MARCA}: fase {R['fase_actual']}, "
                f"ROAS {R['roas'][['canal', 'ROAS']].round(2).to_dict('records')}, optimización {o}, "
                f"CLV {R['clv_formula']}, sentimiento {R['resumen_sentimiento']}. 5 recomendaciones.",
                sistema="Consultor senior de marketing cuantitativo. Español de México, conciso."))

# ---------------- Tendencias ----------------
with tabs[1]:
    st.header("Google Trends: marca vs competencia")
    marcas = st.multiselect("Marcas a comparar", list(R["trends"].columns), default=list(R["trends"].columns))
    if marcas:
        st.line_chart(R["trends"][marcas])
    st.subheader("Ciclo de vida: crecimiento semanal y aceleración")
    st.latex(r"g_t=\frac{y_t-y_{t-1}}{y_{t-1}}\qquad a_t=g_t-g_{t-1}")
    st.dataframe(R["ciclo_resumen"].round(2), use_container_width=True)
    kw = st.selectbox("Keyword", list(R["ciclo"].keys()))
    c = R["ciclo"][kw]
    st.line_chart(c[["crec_semanal", "aceleracion"]].clip(-1, 1))

# ---------------- ML ----------------
with tabs[2]:
    st.header(f"Pronóstico de interés — '{config.KEYWORD_PRINCIPAL}'")
    st.dataframe(R["reg"]["metricas"].round(3), use_container_width=True)
    modelos_sel = st.multiselect("Modelos a mostrar", list(R["reg"]["predicciones"].keys()),
                                 default=list(R["reg"]["predicciones"].keys()))
    serie = pd.DataFrame({"Real": R["reg_y_test"], **{m: R["reg"]["predicciones"][m] for m in modelos_sel}},
                         index=R["reg_fechas"])
    st.line_chart(serie)
    st.subheader("Test de Diebold-Mariano")
    st.latex(r"DM=\frac{\bar d}{\sqrt{\hat\gamma_0/T}},\quad d_t=e_{1,t}^2-e_{2,t}^2")
    st.dataframe(R["dm_reg"], use_container_width=True)
    st.subheader("Simulación de hiperparámetros (TimeSeriesSplit)")
    st.dataframe(R["reg"]["hiperparametros"], use_container_width=True)
    img("03_importancia_rf")
    st.header("Clasificación de fase: Crecimiento / Estancamiento / Caída")
    st.dataframe(R["clf"]["metricas"].round(3), use_container_width=True)
    img("04_matrices_confusion")
    st.subheader("Distribución KNN (semanas proyectadas en 2D con PCA)")
    st.caption("Cada punto es una semana; KNN clasifica una semana nueva según sus vecinos más cercanos (distancia Euclidiana).")
    st.scatter_chart(R["knn_pca"], x="PC1", y="PC2", color="fase")
    st.dataframe(R["dm_clf"], use_container_width=True)

# ---------------- MMM ----------------
with tabs[3]:
    st.header("Marketing Mix Modeling")
    st.latex(r"Ventas_t=b_0+b_{T}\,Interés_t+\sum_c\beta_c\,\frac{A_{c,t}^{s_c}}{A_{c,t}^{s_c}+K_c^{s_c}},"
             r"\quad A_{c,t}=G_{c,t}+\lambda_c A_{c,t-1}")
    st.dataframe(R["mmm_metricas"].round(2), use_container_width=True)
    st.dataframe(R["dm_mmm"], use_container_width=True)
    img("05_mmm_descomposicion")
    st.subheader("ROAS por canal")
    st.dataframe(R["roas"].round(2), use_container_width=True)
    img("06_curvas_respuesta")
    st.subheader("Optimizador de presupuesto (SciPy SLSQP)")
    pres = st.slider("Presupuesto semanal (MXN)", 20_000, 150_000, int(config.PRESUPUESTO_SEMANAL_TOTAL), 5_000)
    t, res = R["mmm"].optimizar(R["df_marketing"], presupuesto=pres)
    c1, c2, c3 = st.columns(3)
    c1.metric("ROAS mix actual", f"{res['ROAS_actual']:.2f}")
    c2.metric("ROAS óptimo", f"{res['ROAS_proyectado_optimo']:.2f}")
    c3.metric("Ventas incrementales extra", f"${res['ventas_incrementales_optimo'] - res['ventas_incrementales_actual']:,.0f}")
    st.bar_chart(t.set_index("canal")[["gasto_actual", "gasto_optimo"]])
    st.dataframe(t.round(1), use_container_width=True)

# ---------------- Atribución / Pearson ----------------
with tabs[4]:
    st.header("Modelos de atribución (% del crédito de conversión)")
    st.dataframe(R["atrib_pct"], use_container_width=True)
    img("09_atribucion")
    st.caption("Markov: efecto de eliminación RE_c = 1 − P(conv sin c)/P(conv). "
               f"P(conv) base = {R['atrib_pconv']:.3f}")
    st.header("Correlación de Pearson")
    img("08_pearson")
    st.dataframe(R["pearson_ventas"], use_container_width=True)

# ---------------- CLV ----------------
with tabs[5]:
    st.header("Customer Lifetime Value")
    st.latex(r"CLV=\frac{(Ticket\cdot Frecuencia\cdot Margen)\, r}{1+d-r}")
    st.json({k: round(v, 3) for k, v in R["clv_formula"].items()})
    st.subheader("CLV a 6 meses con ML (RF / MLP / KNN)")
    st.dataframe(R["clv_ml_metricas"].round(2), use_container_width=True)
    st.dataframe(R["dm_clv"], use_container_width=True)
    st.subheader("Segmentos RFM")
    st.dataframe(R["segmentos"].round(1), use_container_width=True)
    img("10_clv_segmentos")

# ---------------- Sentimiento ----------------
with tabs[6]:
    st.header("Sentimiento de mercado por marca (YouTube + Gemini)")
    st.caption(f"Fuente: {R['fuente_comentarios']} · {len(R['comentarios'])} comentarios · Motor: {R['motor_nlp']}")
    sm = R["sent_marca"]
    st.bar_chart(sm[["negativo", "neutral", "positivo"]], horizontal=True, stack=True,
                 color=["#d55555", "#bbbbbb", "#44aa44"])
    st.dataframe(sm, use_container_width=True)
    if R.get("nlp_validacion"):
        v = R["nlp_validacion"]
        st.write(f"Validación vs etiquetas humanas → Accuracy **{v['accuracy']:.2f}**, F1 macro **{v['F1_macro']:.2f}**")
    cols = [c for c in ["marca", "fecha", "fuente", "texto", "sentimiento", "score", "tema", "likes"] if c in R["comentarios"]]
    st.dataframe(R["comentarios"][cols], use_container_width=True)

    st.subheader("🔴 Cargar comentarios en tiempo real")
    if not youtube_comments.disponible():
        st.info("Configura YOUTUBE_API_KEY para descargar comentarios en vivo.")
    else:
        c1, c2 = st.columns([3, 1])
        marcas_live = c1.multiselect("Marcas", config.KEYWORDS, default=config.KEYWORDS[:3])
        extra = c1.text_input("Agregar otra marca (opcional)")
        n_vid = c2.number_input("Videos por marca", 1, 10, 3)
        if st.button("Descargar y analizar ahora"):
            lista = marcas_live + ([extra.strip()] if extra.strip() else [])
            with st.spinner("Descargando comentarios de YouTube…"):
                crudos = youtube_comments.descargar(lista, videos_por_marca=int(n_vid), comentarios_por_video=15,
                                                    guardar=False)
            if crudos.empty:
                st.warning("No se encontraron comentarios.")
            else:
                with st.spinner(f"Gemini analizando {len(crudos)} comentarios…"):
                    out, motor = nlp_sentiment.analizar(crudos)
                st.caption(f"{len(out)} comentarios · {motor}")
                vivo = nlp_sentiment.por_marca(out)
                st.bar_chart(vivo[["negativo", "neutral", "positivo"]], horizontal=True, stack=True,
                             color=["#d55555", "#bbbbbb", "#44aa44"])
                st.dataframe(out[["marca", "texto", "sentimiento", "score", "tema"]], use_container_width=True)

    st.subheader("Prueba manual")
    texto = st.text_area("Pega comentarios (uno por línea)")
    if st.button("Analizar con Gemini") and texto.strip():
        df = pd.DataFrame({"texto": [t for t in texto.splitlines() if t.strip()]})
        with st.spinner("Clasificando…"):
            out, motor = nlp_sentiment.analizar(df)
        st.caption(motor)
        st.dataframe(out[["texto", "sentimiento", "score", "tema"]], use_container_width=True)

# ---------------- Agentes ----------------
with tabs[7]:
    st.header("Sistema multi-agente de campañas (Gemini + function calling)")
    st.markdown("**Analista de Tendencias** → `resumen_tendencias()` · **Gestor de Presupuesto** → "
                "`optimizar_presupuesto()`, `simular_campana()` · **Copywriter Creativo** → copies y guiones")
    c1, c2, c3 = st.columns(3)
    pres = c1.number_input("Presupuesto semanal (MXN)", 10_000, 500_000, int(config.PRESUPUESTO_SEMANAL_TOTAL), 5_000)
    sem = c2.number_input("Semanas", 1, 12, 4)
    obj = c3.selectbox("Objetivo", ["ventas", "awareness", "lanzamiento de producto", "temporada de calor"])
    if st.button("🚀 Generar plan de campaña"):
        with st.spinner("Los agentes están trabajando…"):
            salida, ev = agents.orquestar(pres, sem, obj)
        st.subheader("Evaluación cuantitativa del plan")
        c = st.columns(4)
        c[0].metric("CPA simulado", f"${ev['CPA_simulado']:,.2f}")
        c[1].metric("ROI acumulado", f"{ev['ROI_acumulado_%']:.1f}%")
        c[2].metric("ROAS", f"{ev['ROAS']:.2f}")
        c[3].metric("Coherencia presupuesto", f"{ev['coherencia_presupuesto_%']:.1f}%")
        for nombre, txt in salida.items():
            with st.expander(nombre, expanded=True):
                st.markdown(txt)

# ---------------- Estado ----------------
with tabs[8]:
    st.header("Auditoría de llaves e infraestructura")
    st.write("Verifica antes de la defensa que Gemini responde y tiene cuota.")
    if st.button("Probar Gemini ahora"):
        ok, msg = gemini_client.verificar()
        (st.success if ok else st.error)(f"{'OK' if ok else 'FALLA'} — {msg}")
    st.write(f"YouTube Data API: {'🟢 llave configurada' if youtube_comments.disponible() else '🔴 sin YOUTUBE_API_KEY'}")
    st.write("Google Trends: los datos se leen de caché (no dependen de PyTrends en vivo, evita error 429).")
