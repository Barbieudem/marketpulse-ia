"""Pipeline completo de MarketPulse IA.

Uso:
    python run_pipeline.py              # usa caché si existe
    python run_pipeline.py --descargar  # fuerza descarga nueva de Google Trends

Genera:
    outputs/resultados.pkl      → lo lee la app (Streamlit)
    outputs/resultados.md       → tablas listas para el Informe Técnico
    outputs/figuras/*.png       → gráficas para el informe y la presentación
"""
import argparse
import pickle
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import config
import agents, attribution, clv, correlation, gemini_client, nlp_sentiment
from data_customers import cargar_transacciones
from data_marketing import PARAMS_VERDADEROS, simular_marketing, simular_recorridos
from data_trends import cargar_trends
from diebold_mariano import tabla_dm
from features import crecimiento_y_aceleracion, features_supervisadas, split_temporal
from metrics import todas_las_metricas
from mmm import MMM, features_mmm_ml
from models_ml import entrenar_clasificadores, entrenar_regresores, errores_clasificacion_0_1

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")
FIG = config.FIG_DIR


def guardar(fig, nombre):
    fig.tight_layout()
    fig.savefig(FIG / f"{nombre}.png", dpi=150)
    plt.close(fig)


def md(df, dec=3):
    return df.round(dec).to_markdown(index=False) if isinstance(df, pd.DataFrame) else str(df)


def main(descargar=False):
    R = {}
    print("1) Google Trends…")
    trends, fuente_t = cargar_trends(forzar_descarga=descargar)
    trends = trends.astype(float)
    R["trends"], R["fuente_trends"] = trends, fuente_t
    print("   fuente:", fuente_t, "| semanas:", len(trends))

    fig, ax = plt.subplots(figsize=(11, 4))
    trends.plot(ax=ax)
    ax.set_title(f"Interés de búsqueda en Google Trends ({config.GEO})")
    ax.set_ylabel("Índice 0-100")
    guardar(fig, "01_tendencias")

    # --- Ciclo de vida: crecimiento y aceleración (Opción 7) ---
    ciclo = {kw: crecimiento_y_aceleracion(trends[kw]) for kw in trends.columns}
    R["ciclo"] = ciclo
    ultimo = pd.DataFrame({kw: {"interes_actual": c["interes"].iloc[-1],
                                "crec_semanal_%": 100 * c["crec_semanal"].iloc[-1],
                                "aceleracion_pp": 100 * c["aceleracion"].iloc[-1],
                                "momentum_4s_%": 100 * c["momentum_4"].iloc[-1],
                                "semana_pico": str(c["interes"].idxmax().date())}
                           for kw, c in ciclo.items()}).T.reset_index(names="keyword")
    R["ciclo_resumen"] = ultimo

    # --- ML sobre la keyword principal ---
    print("2) Modelos ML de tendencia (RF / MLP / KNN)…")
    ds = features_supervisadas(trends[config.KEYWORD_PRINCIPAL])
    tr, te = split_temporal(ds)
    Xcols = [c for c in ds.columns if c not in ("y_reg", "y_clf")]
    reg = entrenar_regresores(tr[Xcols], tr["y_reg"], te[Xcols], te["y_reg"])
    R["reg"] = reg
    R["reg_y_test"], R["reg_fechas"] = te["y_reg"].values, te.index
    dm_reg = tabla_dm(te["y_reg"].values, reg["predicciones"])
    R["dm_reg"] = dm_reg

    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(te.index, te["y_reg"], "k-", lw=2, label="Real")
    for n, p in reg["predicciones"].items():
        ax.plot(te.index, p, label=n, alpha=0.8)
    ax.set_title(f"Pronóstico 1 semana adelante — '{config.KEYWORD_PRINCIPAL}' (conjunto de prueba)")
    ax.legend()
    guardar(fig, "02_pronostico_tendencia")

    fig, ax = plt.subplots(figsize=(7, 4))
    reg["importancias_rf"].head(10).sort_values().plot.barh(ax=ax, color="#2a7")
    ax.set_title("Importancia de variables — Random Forest")
    guardar(fig, "03_importancia_rf")

    print("3) Clasificación de fase (Crecimiento / Estancamiento / Caída)…")
    clf = entrenar_clasificadores(tr[Xcols], tr["y_clf"], te[Xcols], te["y_clf"])
    R["clf"] = clf
    perdidas = {n: errores_clasificacion_0_1(te["y_clf"], r["pred"]) for n, r in clf["resultados"].items()}
    # DM sobre pérdida 0-1: comparamos y=0 contra la pérdida de cada modelo
    dm_clf = tabla_dm(np.zeros(len(te)), perdidas, loss="absoluta")
    R["dm_clf"] = dm_clf
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, (n, r) in zip(axes, clf["resultados"].items()):
        sns.heatmap(r["cm"], annot=True, fmt="d", cmap="Blues", ax=ax,
                    xticklabels=r["etiquetas"], yticklabels=r["etiquetas"], cbar=False)
        ax.set_title(n)
        ax.set_xlabel("Predicho")
        ax.set_ylabel("Real")
    fig.suptitle("Matrices de confusión — fase de la tendencia")
    guardar(fig, "04_matrices_confusion")

    # Distribución KNN: proyección PCA 2D de las semanas (distancia Euclidiana)
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    Z = PCA(n_components=2, random_state=config.SEED).fit_transform(StandardScaler().fit_transform(ds[Xcols]))
    R["knn_pca"] = pd.DataFrame({"PC1": Z[:, 0], "PC2": Z[:, 1], "fase": ds["y_clf"].values,
                                 "conjunto": ["entrenamiento"] * len(tr) + ["prueba"] * len(te)}, index=ds.index)
    fig, ax = plt.subplots(figsize=(7, 5))
    for f, col in zip(["Caída", "Estancamiento", "Crecimiento"], ["#d55", "#999", "#2a7"]):
        d = R["knn_pca"][R["knn_pca"]["fase"] == f]
        ax.scatter(d["PC1"], d["PC2"], s=18, alpha=0.75, color=col, label=f)
    ax.set_title("Distribución KNN: semanas en el espacio de variables (PCA)")
    ax.set_xlabel("Componente 1")
    ax.set_ylabel("Componente 2")
    ax.legend()
    guardar(fig, "12_distribucion_knn")

    mejor_clf = clf["metricas"].sort_values("F1_macro", ascending=False).iloc[0]["modelo"]
    modelo_fase = clf["resultados"][mejor_clf]["modelo"]
    fase_actual = modelo_fase.predict(ds[Xcols].iloc[[-1]])[0]
    # para la semana más reciente (sin etiqueta futura aún) recalculamos features completas
    R["fase_actual"], R["mejor_clf"] = fase_actual, mejor_clf

    # --- MMM (Opción 2) ---
    print("4) Marketing Mix Model + optimización de presupuesto…")
    dfm = simular_marketing(trends)
    R["df_marketing"] = dfm
    corte = int(len(dfm) * (1 - config.TEST_SIZE))
    mmm_tr = MMM().fit(dfm.iloc[:corte])
    pred_mmm_te = mmm_tr.predict(dfm)[corte:]
    Xm = features_mmm_ml(dfm, mmm_tr.params)
    Xm["lag_1"] = dfm["ventas"].shift(1).bfill()
    reg_mmm = entrenar_regresores(Xm.iloc[:corte], dfm["ventas"].iloc[:corte],
                                  Xm.iloc[corte:], dfm["ventas"].iloc[corte:])
    preds_mmm = {"MMM paramétrico": pred_mmm_te, **reg_mmm["predicciones"]}
    met_mmm = pd.concat([pd.DataFrame([{"modelo": "MMM paramétrico",
                                        **todas_las_metricas(dfm["ventas"].iloc[corte:], pred_mmm_te)}]),
                         reg_mmm["metricas"]], ignore_index=True)
    R["mmm_metricas"], R["mmm_hiper"] = met_mmm, reg_mmm["hiperparametros"]
    R["dm_mmm"] = tabla_dm(dfm["ventas"].iloc[corte:].values, preds_mmm)
    R["mmm_preds"], R["mmm_y_test"], R["mmm_fechas"] = preds_mmm, dfm["ventas"].iloc[corte:].values, dfm.index[corte:]

    mmm = MMM().fit(dfm)  # modelo final con toda la historia
    R["mmm"] = mmm
    R["roas"] = mmm.tabla_roas(dfm)
    R["roas"]["ROAS_verdadero_simulacion"] = [
        (PARAMS_VERDADEROS[c]["beta"] * __import__("data_marketing", fromlist=["hill"]).hill(
            __import__("data_marketing", fromlist=["adstock"]).adstock(dfm[c].values, PARAMS_VERDADEROS[c]["lambda"]),
            PARAMS_VERDADEROS[c]["K"], PARAMS_VERDADEROS[c]["s"])).sum() / dfm[c].sum() for c in config.CANALES]
    opt_t, opt_r = mmm.optimizar(dfm)
    R["opt_tabla"], R["opt_resumen"] = opt_t, opt_r
    R["curvas"] = mmm.curvas_respuesta(dfm)
    R["descomposicion"] = mmm.descomposicion(dfm)

    fig, ax = plt.subplots(figsize=(11, 4))
    R["descomposicion"].plot.area(ax=ax, alpha=0.75)
    ax.plot(dfm.index, dfm["ventas"], "k-", lw=1, label="Ventas reales")
    ax.set_title("Descomposición de ventas — Marketing Mix Model")
    ax.legend(loc="upper left", fontsize=8)
    guardar(fig, "05_mmm_descomposicion")

    fig, ax = plt.subplots(figsize=(8, 4))
    for c, g in R["curvas"].groupby("canal"):
        ax.plot(g["gasto"], g["ventas_incrementales"], label=c)
    ax.set_title("Curvas de respuesta (saturación) por canal")
    ax.set_xlabel("Gasto semanal (MXN)")
    ax.set_ylabel("Ventas incrementales (MXN)")
    ax.legend()
    guardar(fig, "06_curvas_respuesta")

    fig, ax = plt.subplots(figsize=(8, 4))
    opt_t.set_index("canal")[["gasto_actual", "gasto_optimo"]].plot.bar(ax=ax, rot=0)
    ax.set_title(f"Presupuesto actual vs óptimo (ROAS {opt_r['ROAS_actual']:.2f} → {opt_r['ROAS_proyectado_optimo']:.2f})")
    guardar(fig, "07_presupuesto_optimo")

    # --- Pearson ---
    print("5) Correlación de Pearson…")
    pear_df = dfm[config.CANALES + ["interes_trends", "ventas"]]
    r, p = correlation.matriz_pearson(pear_df)
    R["pearson_r"], R["pearson_p"] = r, p
    R["pearson_ventas"] = correlation.pares_significativos(r, p, "ventas")
    fig, ax = plt.subplots(figsize=(7, 5.5))
    sns.heatmap(r, annot=True, fmt=".2f", cmap="RdBu_r", vmin=-1, vmax=1, ax=ax)
    ax.set_title("Matriz de correlación de Pearson")
    guardar(fig, "08_pearson")

    # --- Atribución ---
    print("6) Modelos de atribución…")
    rec = simular_recorridos()
    abs_, pct, re, base = attribution.tabla_atribucion(rec)
    R["atrib_abs"], R["atrib_pct"], R["atrib_removal"], R["atrib_pconv"] = abs_, pct, re, base
    fig, ax = plt.subplots(figsize=(10, 4))
    pct.plot.bar(ax=ax, rot=0)
    ax.set_title("Crédito de conversión por canal según modelo de atribución (%)")
    ax.legend(fontsize=8)
    guardar(fig, "09_atribucion")

    # --- CLV ---
    print("7) Customer Lifetime Value…")
    tx, fuente_tx = cargar_transacciones()
    R["fuente_tx"] = fuente_tx
    corte_rfm = tx["fecha"].max() + pd.Timedelta(days=1)
    seg = clv.segmentar(clv.rfm(tx, corte_rfm))
    R["clv_formula"] = clv.clv_formula(tx)
    clv_ml = clv.entrenar_clv_ml(tx)
    R["clv_ml_metricas"] = clv_ml["metricas"]
    R["dm_clv"] = tabla_dm(clv_ml["y_test"], clv_ml["predicciones"])
    pred_clv = clv_ml["modelo_rf"].predict(clv.rfm(tx, corte_rfm)[clv_ml["X"].columns])
    seg["CLV_6m_pred"] = pred_clv
    R["segmentos"] = seg.groupby("segmento").agg(clientes=("frecuencia", "size"),
                                                 frecuencia_prom=("frecuencia", "mean"),
                                                 monetario_prom=("monetario", "mean"),
                                                 CLV_6m_prom=("CLV_6m_pred", "mean")).reset_index()
    fig, ax = plt.subplots(figsize=(8, 4))
    R["segmentos"].set_index("segmento")["CLV_6m_prom"].sort_values().plot.barh(ax=ax, color="#a5d")
    ax.set_title("CLV predicho a 6 meses por segmento RFM (Random Forest)")
    guardar(fig, "10_clv_segmentos")

    # --- NLP ---
    print("8) Sentimiento (comentarios de YouTube + Gemini)…")
    crudos, fuente_com = nlp_sentiment.cargar_comentarios(descargar=descargar)
    com, motor = nlp_sentiment.analizar_con_cache(crudos, fuente_com, forzar=descargar)
    R["comentarios"], R["motor_nlp"], R["fuente_comentarios"] = com, motor, fuente_com
    R["nlp_validacion"] = nlp_sentiment.validar(com)
    R["isn"] = nlp_sentiment.indice_sentimiento_neto(com)
    R["sent_marca"] = nlp_sentiment.por_marca(com)
    print("   fuente:", fuente_com, "| comentarios:", len(com), "| motor:", motor)
    sm = R["sent_marca"]
    fig, ax = plt.subplots(figsize=(9, 0.6 * len(sm) + 1.8))
    izq = np.zeros(len(sm))
    for et, col in zip(nlp_sentiment.ETIQUETAS, ["#d55", "#bbb", "#4a4"]):
        ax.barh(sm.index, sm[et], left=izq, color=col, label=et)
        izq += sm[et].values
    ax.set_xlabel("% de comentarios")
    ax.set_title("Sentimiento de mercado por marca (comentarios de YouTube)")
    ax.legend(loc="lower right", fontsize=8)
    guardar(fig, "11_sentimiento")

    # --- Contexto para agentes ---
    kw_mom = ultimo.sort_values("momentum_4s_%", ascending=False).iloc[0]
    R["resumen_tendencias"] = {
        "keyword_principal": config.KEYWORD_PRINCIPAL,
        "fase_pronosticada": fase_actual, "modelo_fase": mejor_clf,
        "tabla": ultimo.round(2).to_dict("records"),
        "keyword_con_mas_momentum": kw_mom["keyword"],
    }
    R["resumen_sentimiento"] = {"ISN": round(R["isn"], 1),
                                "ISN_por_marca": R["sent_marca"]["ISN"].to_dict(),
                                "distribucion": com["sentimiento"].value_counts().to_dict(),
                                "temas": com["tema"].value_counts().to_dict()}
    agents.configurar_contexto(R)
    R["eval_agentes"] = agents.evaluar(config.PRESUPUESTO_SEMANAL_TOTAL, 4)

    # --- Análisis de Gemini sobre los resultados ---
    print("9) Interpretación con Gemini…")
    R["analisis_gemini"] = gemini_client.generar(
        "Interpreta para un director de marketing estos resultados y da 5 recomendaciones accionables:\n"
        f"Métricas pronóstico: {reg['metricas'].round(3).to_dict('records')}\n"
        f"Diebold-Mariano: {dm_reg.to_dict('records')}\nFase actual: {fase_actual}\n"
        f"ROAS por canal: {R['roas'][['canal', 'ROAS', 'ROAS_marginal']].round(2).to_dict('records')}\n"
        f"Optimización: {opt_r}\nCLV: {R['clv_formula']}\nSentimiento: {R['resumen_sentimiento']}",
        sistema="Eres consultor senior de marketing cuantitativo. Responde en español de México, conciso.")

    with open(config.OUT_DIR / "resultados.pkl", "wb") as f:
        pickle.dump(R, f)
    escribir_md(R)
    print("✔ Listo. Revisa outputs/resultados.md y outputs/figuras/")
    return R


def escribir_md(R):
    L = [f"# Resultados — MarketPulse IA ({config.MARCA})\n",
         f"- Fuente tendencias: **{R['fuente_trends']}**",
         f"- Fuente transacciones (CLV): **{R['fuente_tx']}**",
         f"- Motor NLP: **{R['motor_nlp']}**\n",
         "## 1. Ciclo de vida de tendencias (crecimiento y aceleración)\n", md(R["ciclo_resumen"], 2),
         "\n## 2. Pronóstico de interés (regresión)\n", md(R["reg"]["metricas"]),
         "\n### Mejores hiperparámetros\n", str(R["reg"]["mejores"]),
         "\n### Simulación de hiperparámetros (top 5 por modelo)\n",
         md(R["reg"]["hiperparametros"].groupby("modelo").head(5)),
         "\n### Test Diebold-Mariano (pérdida cuadrática)\n", md(R["dm_reg"]),
         "\n## 3. Clasificación de fase\n", md(R["clf"]["metricas"]),
         f"\nFase pronosticada actual: **{R['fase_actual']}** (modelo {R['mejor_clf']})\n",
         "\n### Diebold-Mariano sobre pérdida 0-1\n", md(R["dm_clf"]),
         "\n## 4. Marketing Mix Model\n", md(R["mmm_metricas"]),
         "\n### Diebold-Mariano (ventas)\n", md(R["dm_mmm"]),
         "\n### ROAS por canal (estimado vs verdadero de la simulación)\n", md(R["roas"], 2),
         "\n### Presupuesto óptimo\n", md(R["opt_tabla"], 1), "\n", str({k: (round(v, 3) if isinstance(v, float) else v)
                                                                       for k, v in R["opt_resumen"].items()}),
         "\n## 5. Pearson vs ventas\n", md(R["pearson_ventas"], 4),
         "\n## 6. Atribución (% del crédito)\n", R["atrib_pct"].reset_index(names="canal").to_markdown(index=False),
         "\n## 7. CLV\n", str({k: round(v, 3) for k, v in R["clv_formula"].items()}),
         "\n", md(R["clv_ml_metricas"]), "\n", md(R["dm_clv"]), "\n", md(R["segmentos"], 1),
         "\n## 8. Sentimiento\n", f"Fuente: {R['fuente_comentarios']} · {len(R['comentarios'])} comentarios · ISN global = {R['isn']:.1f}\n",
         R["sent_marca"].reset_index().to_markdown(index=False),
         f"\nValidación vs etiquetas manuales: accuracy={R['nlp_validacion']['accuracy']:.3f}, "
         f"F1_macro={R['nlp_validacion']['F1_macro']:.3f}" if R["nlp_validacion"] else "",
         "\n## 9. Evaluación del sistema multi-agente\n", str(R["eval_agentes"]),
         "\n## 10. Interpretación de Gemini\n", R["analisis_gemini"]]
    (config.OUT_DIR / "resultados.md").write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--descargar", action="store_true")
    main(ap.parse_args().descargar)
