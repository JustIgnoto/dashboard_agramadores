"""
Página 2 — Modelo predictivo
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go


# Tema Plotly global: texto oscuro, ejes legibles y rejilla con contraste.
import plotly.io as pio
_agro_theme = pio.templates["plotly_white"]
_agro_theme.layout.font = dict(family="Arial, sans-serif", size=12, color="#17251B")
_agro_theme.layout.title = dict(font=dict(color="#10291C", size=16))
_agro_theme.layout.xaxis = dict(
    color="#263D2D", title_font=dict(color="#17251B"),
    tickfont=dict(color="#263D2D"), gridcolor="#D6DFD8",
    linecolor="#87988C", zerolinecolor="#A9B7AD"
)
_agro_theme.layout.yaxis = dict(
    color="#263D2D", title_font=dict(color="#17251B"),
    tickfont=dict(color="#263D2D"), gridcolor="#D6DFD8",
    linecolor="#87988C", zerolinecolor="#A9B7AD"
)
pio.templates["agrocebada_contraste"] = _agro_theme
pio.templates.default = "agrocebada_contraste"
st.set_page_config(page_title="Modelo · AgroCebada 2026", layout="wide",
                   initial_sidebar_state="expanded")

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
[data-testid="stSidebar"] { background-color: #0F2318; }
[data-testid="stSidebar"] * { color: #C8DCC8 !important; }
[data-testid="stSidebar"] hr { border-color: #1E3E2A !important; }
footer { visibility: hidden; }

.page-header {
    background: linear-gradient(135deg, #0D47A1, #1565C0);
    border-radius: 10px; padding: 18px 26px; margin-bottom: 20px; color:white;
}
.page-header h2 { margin:0; font-size:1.4em; color:#fff; }
.page-header p  { margin:3px 0 0; font-size:.8em; color:#90CAF9; }

.sec { font-size:.76em; font-weight:700; letter-spacing:.07em; color:#0D47A1;
       text-transform:uppercase; border-bottom:2px solid #90CAF9;
       padding-bottom:3px; margin:20px 0 10px; }

.metric-pill {
    background:#E3F2FD; border:1px solid #90CAF9; border-radius:8px;
    padding:12px 16px; text-align:center;
}
.metric-pill .val { font-size:1.7em; font-weight:800; color:#0D47A1; }
.metric-pill .lbl { font-size:.72em; color:#1565C0; margin-top:2px; }

.algo-card {
    border:1.5px solid #BBDEFB; border-radius:10px;
    padding:18px 20px; background:#F8FBFF;
}
.algo-card h4 { margin:0 0 8px; color:#0D47A1; font-size:.95em; }

.info-box {
    background:#E3F2FD; border-left:4px solid #1565C0;
    border-radius:0 8px 8px 0; padding:14px 18px; margin:10px 0;
    font-size:.87em; color:#0D1A3A; line-height:1.6;
}
.warn-box {
    background:#FFF8E1; border-left:4px solid #F9A825;
    border-radius:0 8px 8px 0; padding:14px 18px; margin:10px 0;
    font-size:.87em; color:#4A3A00; line-height:1.6;
}
.placeholder-box {
    background:#F3F4F6; border:2px dashed #9CA3AF;
    border-radius:10px; padding:32px; text-align:center;
    color:#6B7280; font-size:.88em; line-height:1.7;
}

.condicion-bueno    { color:#fff; background:#27AE60; padding:2px 8px; border-radius:4px; font-size:.78em; }
.condicion-alerta1  { color:#000; background:#F1C40F; padding:2px 8px; border-radius:4px; font-size:.78em; }
.condicion-alerta2  { color:#fff; background:#E67E22; padding:2px 8px; border-radius:4px; font-size:.78em; }
.condicion-emerg    { color:#fff; background:#C0392B; padding:2px 8px; border-radius:4px; font-size:.78em; }

/* Contraste global y accesibilidad */
html, body, [data-testid="stAppViewContainer"], [data-testid="stMain"] {
    color: #17251B !important;
}
[data-testid="stSidebar"] { background-color: #10291C !important; }
[data-testid="stSidebar"] * { color: #F0F7F1 !important; }
[data-testid="stSidebar"] hr { border-color: #42634D !important; }
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
[data-testid="stSidebar"] label { color: #F0F7F1 !important; }
[data-testid="stWidgetLabel"] p, [data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] li, [data-testid="stMarkdownContainer"] td,
[data-testid="stMarkdownContainer"] th, label, small {
    color: #17251B;
}
[data-testid="stMetricLabel"] { color: #34483A !important; }
[data-testid="stMetricValue"] { color: #10291C !important; }
[data-testid="stDataFrame"] { border: 1px solid #B6C5BA; border-radius: 8px; }
[data-testid="stTabs"] button { color: #263D2D !important; }
[data-testid="stTabs"] button[aria-selected="true"] {
    color: #14532D !important; border-bottom-color: #167A3E !important;
}
button[kind="secondary"], button[kind="primary"] { font-weight: 600; }
footer { visibility: hidden; }

</style>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("##  AgroCebada 2026")
    st.divider()
    st.page_link("Inicio.py",            label="Inicio")
    st.page_link("pages/1_Datos.py",     label="Datos y procesamiento")
    st.page_link("pages/2_Modelo.py",    label="Modelo predictivo")
    st.page_link("pages/3_Mapa.py",      label="Mapa de parcelas")
    st.divider()

# ── Carga de datos ────────────────────────────────────────────────────────────
import json, os

def cond(v):
    """Semáforo de condición según rendimiento (t/ha); mismo criterio que el mapa."""
    if pd.isna(v): return "Sin dato"
    if v >= 4.3: return "Bueno"
    if v >= 3.5: return "Alerta 1"
    if v >= 2.8: return "Alerta 2"
    return "Emergencia"

def _csv(path, **kw):
    return pd.read_csv(path, **kw) if os.path.exists(path) else None

@st.cache_data
def load_results():
    r = _csv("data/resultados_modelo.csv", encoding="utf-8-sig")
    if r is None:
        return None
    r = r.rename(columns={"RENDIMIENTO_PREDICHO_T_HA": "rendimiento_predicho"})
    r["condicion"] = r["rendimiento_predicho"].apply(cond)
    return r

@st.cache_data
def load_imp():      return _csv("data/importancia_features.csv")
@st.cache_data
def load_metricas(): return _csv("data/metricas_cv.csv")
@st.cache_data
def load_folds():    return _csv("data/metricas_folds.csv")
@st.cache_data
def load_oof():      return _csv("data/oof_entrenamiento.csv")

@st.cache_data
def load_resumen():
    if not os.path.exists("data/resumen_modelo.json"):
        return None
    return json.load(open("data/resumen_modelo.json", encoding="utf-8"))

@st.cache_data
def load_ext():
    df = pd.read_csv("data/Conjunto_datos_EXT_AgroCebada.csv", encoding="latin1")
    df["Municipio"] = df["Municipio"].str.strip()
    return df

@st.cache_data
def load_obs():
    return pd.read_csv("data/ID_area_rendimiento_70_30_Reto_AgroCebada.csv", encoding="utf-8-sig")

res_df  = load_results()
imp_df  = load_imp()
met_df  = load_metricas()
fold_df = load_folds()
oof_df  = load_oof()
resumen = load_resumen()
ext_df  = load_ext()

SIN_ARTEFACTOS = (imp_df is None or met_df is None or fold_df is None or resumen is None)
AVISO_ARTEFACTOS = """<div class="warn-box">Faltan archivos del modelo en <code>data/</code>
(<code>importancia_features.csv</code>, <code>metricas_cv.csv</code>, <code>metricas_folds.csv</code>,
<code>resumen_modelo.json</code>). Genéralos con
<code>python scripts/generar_artefactos_modelo.py --tabla &lt;Tabla_Maestra_Mensual_Escalada_MinMax.csv&gt;</code>.</div>"""

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="page-header">
  <h2> Modelo predictivo</h2>
  <p>Arquitectura, métricas de desempeño, importancia de variables y análisis de resultados</p>
</div>
""", unsafe_allow_html=True)

# ── Tabs ──────────────────────────────────────────────────────────────────────
t1, t2, t3, t4, t5 = st.tabs([
    "  Arquitectura",
    "  Métricas CV",
    "  Variables importantes",
    "  Resultados",
    "  Notas del modelo",
])


# ════════════════════════════════════════════════════════════════════════════════
# TAB 1 — ARQUITECTURA
# ════════════════════════════════════════════════════════════════════════════════
with t1:
    if SIN_ARTEFACTOS:
        st.markdown(AVISO_ARTEFACTOS, unsafe_allow_html=True)
    else:
        n_cand = resumen["n_variables_candidatas"]; n_sel = resumen["n_variables_seleccionadas"]
        col_a, col_b = st.columns([3,2], gap="large")

        with col_a:
            st.markdown('<p class="sec">Modelo final</p>', unsafe_allow_html=True)
            st.markdown(f"""
<div class="algo-card">
<h4> SVR (kernel RBF) con selección de variables por Lasso</h4>

El modelo final es una <strong>máquina de vectores de soporte para regresión (SVR)</strong> entrenada
sobre las <strong>{n_sel} variables</strong> que Lasso conserva de {n_cand:,} candidatas
(clima CHIRPS, topografía INEGI e índices espectrales Sentinel-2 / Landsat 8 y Planet).
Lasso actúa como filtro: anula los coeficientes de las variables que no aportan y el SVR
aprende la relación no lineal con el rendimiento sobre las que sobreviven.
</div>
            """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown(f"""
| Etapa | Detalle |
|---|---|
| **Preprocesamiento** | `MinMaxScaler` ajustado en entrenamiento · imputación por mediana |
| **Selección de variables** | Lasso, α = {resumen["lasso_alpha_seleccion"]} (GridSearch, CV-5) → {n_sel} variables |
| **Regresor** | SVR RBF · C = {resumen["svr_C"]} · γ = {resumen["svr_gamma"]} · ε = {resumen["svr_epsilon"]} |
| **Validación** | K-Fold k = 5 con barajado · `random_state = 42` |
| **Datos** | {resumen["n_entrenamiento"]} parcelas de entrenamiento · {resumen["n_prediccion"]} de predicción |
            """)

        with col_b:
            st.markdown('<p class="sec">Flujo del pipeline</p>', unsafe_allow_html=True)
            st.markdown(f"""
```
Tabla maestra mensual ({n_cand:,} variables)
    │
    ├─► MinMaxScaler + imputación (mediana)
    │
    ├─► Lasso → {n_sel} variables activas
    │
    ├─► SVR RBF (GridSearch, CV-5)
    │
    └─► Predicción de {resumen["n_prediccion"]} parcelas
```
            """)

        st.markdown('<p class="sec">Variables candidatas vs. seleccionadas por grupo</p>', unsafe_allow_html=True)
        cand = pd.Series(resumen["candidatas_por_grupo"], name="Candidatas")
        selg = imp_df["grupo"].value_counts().rename("Seleccionadas")
        grp = pd.concat([cand, selg], axis=1).fillna(0).astype(int).reset_index()
        grp.columns = ["Grupo", "Candidatas", "Seleccionadas"]
        fig_g = px.bar(grp.melt(id_vars="Grupo", var_name="Tipo", value_name="Variables"),
                       x="Grupo", y="Variables", color="Tipo", barmode="group", text_auto=True,
                       color_discrete_map={"Candidatas":"#90A4AE","Seleccionadas":"#2E7D32"},
                       log_y=True, labels={"Grupo":""})
        fig_g.update_layout(plot_bgcolor="white", paper_bgcolor="white",
                            height=280, margin=dict(t=10,b=30,l=10,r=10),
                            legend=dict(orientation="h",y=1.08))
        st.plotly_chart(fig_g, use_container_width=True)
        st.caption("Escala logarítmica en el eje vertical.")


# ════════════════════════════════════════════════════════════════════════════════
# TAB 2 — MÉTRICAS CV
# ════════════════════════════════════════════════════════════════════════════════
with t2:
    if SIN_ARTEFACTOS:
        st.markdown(AVISO_ARTEFACTOS, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="info-box">
        Métricas de la validación cruzada K-Fold (k=5) sobre las {resumen["n_entrenamiento"]} parcelas de
        entrenamiento, calculadas con predicciones <strong>fuera de muestra</strong> (OOF).
        Se comparan los tres modelos entrenados con las mismas {resumen["n_variables_seleccionadas"]} variables.
        </div>
        """, unsafe_allow_html=True)

        mt = met_df.rename(columns={"R2": "R²"})
        mt_long = mt.melt(id_vars="Modelo", var_name="Métrica", value_name="Valor")
        mt_long["Métrica"] = mt_long["Métrica"].replace({"RMSE":"RMSE (t/ha)","MAE":"MAE (t/ha)"})

        col_m, col_t = st.columns([2,3], gap="medium")
        with col_m:
            st.markdown('<p class="sec">Comparación de modelos</p>', unsafe_allow_html=True)
            fig_met = px.bar(mt_long, x="Métrica", y="Valor", color="Modelo", barmode="group",
                             color_discrete_map={"Lasso":"#1565C0","Poly + Ridge":"#F57C00","SVR (RBF)":"#2E7D32"},
                             text_auto=".3f")
            fig_met.update_layout(plot_bgcolor="white", paper_bgcolor="white",
                                  height=320, margin=dict(t=10,b=40,l=10,r=10),
                                  legend=dict(orientation="h",y=1.08))
            st.plotly_chart(fig_met, use_container_width=True)

        with col_t:
            st.markdown('<p class="sec">Tabla de métricas CV-5</p>', unsafe_allow_html=True)
            st.dataframe(mt, use_container_width=True, hide_index=True,
                         column_config={c: st.column_config.NumberColumn(format="%.3f")
                                        for c in ["RMSE","MAE","R²"]})
            svr_row = met_df[met_df["Modelo"] == "SVR (RBF)"].iloc[0]
            media_obs = float(load_obs()["RENDIMIENTO_T_HA"].mean())
            st.markdown(f"""
| Métrica | Interpretación |
|---|---|
| **RMSE** | Error cuadrático medio — penaliza errores grandes |
| **MAE** | Error absoluto medio — más intuitivo en t/ha |
| **R²** | Proporción de varianza explicada por el modelo |

Con un MAE de {svr_row["MAE"]:.2f} t/ha sobre un rendimiento medio observado de {media_obs:.2f} t/ha,
el error relativo del modelo final (SVR) es de aproximadamente **{svr_row["MAE"]/media_obs*100:.0f} %**.
            """)

        st.markdown('<p class="sec">Variabilidad del error entre folds</p>', unsafe_allow_html=True)
        fig_folds = px.bar(fold_df, x="Fold", y="RMSE", color="Modelo", barmode="group",
                           color_discrete_map={"Lasso":"#1565C0","Poly + Ridge":"#F57C00","SVR (RBF)":"#2E7D32"},
                           labels={"RMSE":"RMSE (t/ha)"})
        fig_folds.update_layout(plot_bgcolor="white", paper_bgcolor="white",
                                height=260, margin=dict(t=10,b=30,l=10,r=10),
                                legend=dict(orientation="h",y=1.08))
        st.plotly_chart(fig_folds, use_container_width=True)

        st.markdown("""
        <div class="warn-box">
          <strong>Lectura con cautela:</strong> Lasso eligió las variables usando las
        138 parcelas completas y después se validó el SVR con CV-5 sobre esas mismas parcelas.
        Esa selección no se repite dentro de cada fold, por lo que las métricas pueden ser algo
        optimistas frente al desempeño en parcelas realmente nuevas.
        </div>
        """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════════
# TAB 3 — IMPORTANCIA DE VARIABLES
# ════════════════════════════════════════════════════════════════════════════════
with t3:
    if imp_df is None:
        st.markdown(AVISO_ARTEFACTOS, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="info-box">
          Coeficientes de Lasso (variables escaladas con MinMax, por lo que son comparables entre sí).
        Se muestran las variables que Lasso conservó; el signo indica si, en el ajuste lineal de selección,
        la variable se asocia con más (+) o menos (−) rendimiento. El SVR final es no lineal, así que
        esto es un indicador de selección, no una importancia exacta del modelo final.
        </div>""", unsafe_allow_html=True)

        top = imp_df.sort_values("importancia")
        fig_imp = px.bar(top, x="coeficiente", y="feature", orientation="h",
                         color="grupo",
                         color_discrete_map={"Clima (CHIRPS)":"#1565C0","Topografía (INEGI)":"#558B2F",
                                             "Sentinel-2 / Landsat 8":"#6A1B9A","Planet":"#E65100"},
                         labels={"coeficiente":"Coeficiente Lasso","feature":"Variable","grupo":"Grupo"})
        fig_imp.update_layout(plot_bgcolor="white", paper_bgcolor="white",
                              height=max(380, 22*len(top)), margin=dict(t=10,b=20,l=10,r=10),
                              yaxis=dict(tickfont_size=10), legend=dict(orientation="h",y=1.04))
        st.plotly_chart(fig_imp, use_container_width=True)

    # Correlación real de las variables climáticas mensuales con el rendimiento observado
    st.markdown('<p class="sec">Correlación mensual de clima vs. rendimiento observado (entrenamiento)</p>',
                unsafe_allow_html=True)
    obs = load_obs().rename(columns={"ID_POLIGONO": "ID_POLIGON"})
    dtr = ext_df.merge(obs[["ID_POLIGON","RENDIMIENTO_T_HA"]], on="ID_POLIGON", how="left")
    dtr = dtr[dtr["RENDIMIENTO_T_HA"].notna()]
    anios = sorted({c[2:4] for c in dtr.columns if c[:2] in ("Tx","Tm","Pr") and c[2:4].isdigit()})
    anio = st.selectbox("Año de las variables climáticas", anios,
                        index=anios.index("25") if "25" in anios else len(anios)-1,
                        format_func=lambda a: f"20{a}")
    meses_lbl = ["Ene","Feb","Mar","Abr","May","Jun","Jul","Ago","Sep","Oct","Nov","Dic"]
    corr = pd.DataFrame({"Mes": meses_lbl})
    for pref, nom in [("Tx","T máx (Tx)"),("Tm","T mín (Tm)"),("Pr","Precipitación (Pr)")]:
        corr[nom] = [dtr[f"{pref}{anio}{m:02d}_mea"].corr(dtr["RENDIMIENTO_T_HA"])
                     if f"{pref}{anio}{m:02d}_mea" in dtr.columns else np.nan for m in range(1,13)]
    fig_corr = px.line(corr, x="Mes", y=["T máx (Tx)","T mín (Tm)","Precipitación (Pr)"], markers=True,
                       color_discrete_map={"T máx (Tx)":"#D32F2F","T mín (Tm)":"#F57C00",
                                           "Precipitación (Pr)":"#1565C0"},
                       labels={"value":"Correlación de Pearson (r)","variable":"Variable"})
    fig_corr.add_hline(y=0, line_color="#90A4AE", line_width=1)
    fig_corr.update_layout(plot_bgcolor="white", paper_bgcolor="white",
                           height=300, margin=dict(t=10,b=40,l=10,r=10),
                           legend=dict(orientation="h",y=1.08))
    st.plotly_chart(fig_corr, use_container_width=True)
    st.caption(f"r de Pearson entre la variable climática mensual de 20{anio} de cada parcela y su "
               f"rendimiento observado (n = {len(dtr)} parcelas de entrenamiento).")


# ════════════════════════════════════════════════════════════════════════════════
# TAB 4 — RESULTADOS
# ════════════════════════════════════════════════════════════════════════════════
with t4:
    if res_df is None:
        st.markdown("""<div class="warn-box">
          No se encontró <code>data/resultados_modelo.csv</code>.
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown(f"""<div class="info-box">
          Predicciones de rendimiento para las {len(res_df)} parcelas del conjunto de predicción,
        cargadas desde <code>data/resultados_modelo.csv</code>.
        </div>""", unsafe_allow_html=True)

        C = {"Bueno":"#27AE60","Alerta 1":"#F1C40F","Alerta 2":"#E67E22","Emergencia":"#C0392B"}

        col_kpi = st.columns(5)
        col_kpi[0].metric("Parcelas predichas", len(res_df))
        col_kpi[1].metric("Bueno",      int((res_df["condicion"]=="Bueno").sum()))
        col_kpi[2].metric("Alerta 1",   int((res_df["condicion"]=="Alerta 1").sum()))
        col_kpi[3].metric("Alerta 2",   int((res_df["condicion"]=="Alerta 2").sum()))
        col_kpi[4].metric("Emergencia", int((res_df["condicion"]=="Emergencia").sum()))

        col_l, col_r = st.columns([3,2], gap="medium")
        with col_l:
            st.markdown('<p class="sec">Real vs. predicho (entrenamiento, validación cruzada)</p>',
                        unsafe_allow_html=True)
            if oof_df is not None:
                oof = oof_df.copy()
                oof["condicion"] = oof["rendimiento_real"].apply(cond)
                fig_sc = px.scatter(oof, x="rendimiento_real", y="rendimiento_predicho_cv",
                                    color="condicion", color_discrete_map=C,
                                    hover_data=["ID_POLIGONO"],
                                    labels={"rendimiento_real":"Real (t/ha)",
                                            "rendimiento_predicho_cv":"Predicho CV (t/ha)",
                                            "condicion":"Condición real"})
                lo = min(oof["rendimiento_real"].min(), oof["rendimiento_predicho_cv"].min()) - .1
                hi = max(oof["rendimiento_real"].max(), oof["rendimiento_predicho_cv"].max()) + .1
                fig_sc.add_shape(type="line", x0=lo, y0=lo, x1=hi, y1=hi,
                                 line=dict(dash="dash", color="#90A4AE", width=1.5))
                fig_sc.update_layout(plot_bgcolor="white", paper_bgcolor="white",
                                     height=340, margin=dict(t=10,b=40,l=10,r=10))
                st.plotly_chart(fig_sc, use_container_width=True)
            else:
                st.info("No se encontró data/oof_entrenamiento.csv.")

        with col_r:
            st.markdown('<p class="sec">Condición de las parcelas predichas</p>', unsafe_allow_html=True)
            fig_pie = px.pie(res_df, names="condicion", color="condicion",
                             color_discrete_map=C, hole=.42)
            fig_pie.update_layout(paper_bgcolor="white", height=340,
                                  margin=dict(t=10,b=10,l=0,r=0), showlegend=False)
            fig_pie.update_traces(textinfo="percent+label", textfont_size=11)
            st.plotly_chart(fig_pie, use_container_width=True)

        tabla = res_df.merge(
            ext_df.rename(columns={"ID_POLIGON":"ID_POLIGONO"})[["ID_POLIGONO","Municipio","área_ha"]],
            on="ID_POLIGONO", how="left")
        tabla = tabla[["ID_POLIGONO","Municipio","área_ha","rendimiento_predicho","condicion"]]
        st.dataframe(tabla, use_container_width=True, hide_index=True, height=300,
                     column_config={
                         "rendimiento_predicho": st.column_config.NumberColumn("Predicho (t/ha)", format="%.3f"),
                         "área_ha": st.column_config.NumberColumn("Área (ha)", format="%.2f")})
        st.download_button("⬇  Descargar predicciones (.csv)",
                           tabla.to_csv(index=False).encode("utf-8"), "predicciones.csv", "text/csv")


# ════════════════════════════════════════════════════════════════════════════════
# TAB 5 — NOTAS DEL MODELO
# ════════════════════════════════════════════════════════════════════════════════
with t5:
    st.markdown("""
    <div class="info-box">
    ℹ  <strong>Sección editable</strong> — Usa los bloques de expansión para
    documentar decisiones metodológicas, referencias bibliográficas y limitaciones del modelo.
    </div>
    """, unsafe_allow_html=True)

    with st.expander("  Justificación de la arquitectura del modelo", expanded=True):
        st.markdown("""
**Por qué SVR con selección de variables por Lasso**

- Con solo 138 parcelas de entrenamiento y más de 1,700 variables candidatas, la dimensionalidad
  es el principal riesgo de sobreajuste. Lasso reduce el conjunto a unas pocas variables activas
  antes de entrenar el regresor final.
- Las relaciones entre clima, índices espectrales y rendimiento no son lineales; el kernel RBF del SVR
  las captura sin necesidad de especificar interacciones a mano.
- En la validación cruzada, el SVR sobre las variables de Lasso obtuvo el menor RMSE y MAE entre los tres
  modelos comparados (Lasso, Poly + Ridge y SVR); ver la pestaña *Métricas CV*.
        """)
        st.markdown("""
        <div class="warn-box">
          <strong>Pendiente de completar:</strong> las versiones 3 a 5 del repositorio
        (fusión lineal/PCA de GDD, regresión polinómica y Random Forest) no se resumen aquí;
        agrega su comparación si quieres documentar por qué se descartaron.
        </div>
        """, unsafe_allow_html=True)

    with st.expander("  Limitaciones y consideraciones"):
        st.markdown("""
**Limitaciones identificadas del modelo:**

- El conjunto de entrenamiento tiene solo 138 parcelas, lo que limita la generalización
  a zonas muy diferentes a las representadas en los datos.
- La arquitectura de acarreo en cascada de variables fenológicas asume que el ciclo
  de siembra–cosecha es uniforme entre municipios; variaciones locales no están capturadas.
- La señal de las variables Tx y Tm puede estar correlacionada con la elevación (TpEl),
  introduciendo un efecto de confusión no descompuesto explícitamente.
        """)
        st.markdown("""
        <div class="warn-box">
          <strong>Agrega aquí:</strong> Limitaciones específicas que identificaste
        durante el entrenamiento, advertencias sobre la interpretación de los resultados
        en zonas sin datos de entrenamiento, y posibles mejoras futuras.
        </div>
        """, unsafe_allow_html=True)

    with st.expander("  Referencias bibliográficas"):
        st.markdown("""
- Jeong, J. H., et al. (2016). *Random forests for global and regional crop yield predictions.*
  PLOS ONE, 11(6), e0156571.
- Cao, J., et al. (2020). *Improving the accuracy of crop yield prediction using remote sensing data.*
  Agricultural and Forest Meteorology, 287, 107930.
- Mano, M. M., & Ciletti, M. D. (2018). *Digital design* (6th ed.). Pearson.
        """)
        st.markdown("""
        <div class="warn-box">
          <strong>Agrega aquí:</strong> Las referencias específicas que usaste para
        justificar el modelo, las variables elegidas y el proceso de validación.
        </div>
        """, unsafe_allow_html=True)

    with st.expander("  Registro de versiones del modelo"):
        versions_df = pd.DataFrame({
            "Versión": ["Version_1", "Version_2 (final)"],
            "Cambio":  ["SVR + Lasso con dataset filtrado manualmente (53 variables)",
                        "SVR + Lasso sobre tabla maestra mensual MinMax (22 variables de 1,755)"],
            "RMSE CV": [0.5251, 0.4500],
        })
        st.dataframe(versions_df, use_container_width=True, hide_index=True)
        st.markdown("""
        <div class="warn-box">
          Las predicciones de <code>resultados_modelo.csv</code> corresponden a Version_2
        (coinciden con <code>Predicciones_Finales_SVR_Opt.csv</code> del repositorio, salvo redondeo).
        El RMSE de Version_1 sale de su <code>Description.txt</code>. Agrega aquí las versiones 3 a 5 si las quieres documentar.
        </div>
        """, unsafe_allow_html=True)
