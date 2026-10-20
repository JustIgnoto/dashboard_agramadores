"""
Página 3 — Mapa interactivo de parcelas tipo DENUE/INEGI
Usa:
  data/Coordenadas_Parcelas_Dashboard.csv   -> ubicación de las 197 parcelas
  data/resultados_modelo.csv                -> rendimiento predicho (59 parcelas de PREDICCION)
  data/ID_area_rendimiento_70_30_...csv     -> rendimiento observado (138 parcelas de ENTRENAMIENTO)
  data/oof_entrenamiento.csv (opcional)     -> predicción por validación cruzada de ENTRENAMIENTO
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
st.set_page_config(page_title="Mapa · AgroCebada 2026", layout="wide",
                   initial_sidebar_state="expanded")

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
[data-testid="stSidebar"] { background-color: #0F2318; }
[data-testid="stSidebar"] * { color: #C8DCC8 !important; }
[data-testid="stSidebar"] hr { border-color: #1E3E2A !important; }
footer { visibility: hidden; }

.page-header {
    background: linear-gradient(135deg, #4A148C, #6A1B9A);
    border-radius: 10px; padding: 18px 26px; margin-bottom: 20px;
    color: white;
}
.page-header h2 { margin:0; font-size:1.4em; color:#fff; }
.page-header p  { margin:3px 0 0; font-size:.8em; color:#CE93D8; }

.sec { font-size:.76em; font-weight:700; letter-spacing:.07em; color:#4A148C;
       text-transform:uppercase; border-bottom:2px solid #CE93D8;
       padding-bottom:3px; margin:20px 0 10px; }

.demo-badge {
    display:inline-block; background:#FF6F00; color:white;
    padding:3px 10px; border-radius:12px; font-size:.72em;
    font-weight:700; letter-spacing:.05em; margin-bottom:8px;
}
.info-box {
    background:#F3E5F5; border-left:4px solid #6A1B9A;
    border-radius:0 8px 8px 0; padding:14px 18px; margin:10px 0;
    font-size:.87em; color:#1A0033; line-height:1.6;
}

/* Panel de detalle de parcela */
.detail-card {
    background: #FAFAFA;
    border: 1.5px solid #CE93D8;
    border-radius: 12px;
    padding: 18px 20px;
}
.detail-card h4 { margin: 0 0 10px; font-size:1.0em; color:#4A148C; }
.cond-pill {
    display:inline-block; padding:4px 14px; border-radius:20px;
    font-size:.85em; font-weight:700; margin-bottom:12px;
}

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
    st.markdown("**Controles del mapa**")

# ── Carga de datos reales ─────────────────────────────────────────────────────
@st.cache_data
def load_coords():
    df = pd.read_csv("data/Coordenadas_Parcelas_Dashboard.csv", encoding="utf-8-sig")
    df["ID_POLIGONO"] = df["ID_POLIGONO"].str.strip()
    return df

@st.cache_data
def load_ext():
    df = pd.read_csv("data/Conjunto_datos_EXT_AgroCebada.csv", encoding="latin1")
    df["Municipio"] = df["Municipio"].str.strip()
    df = df.rename(columns={"ID_POLIGON": "ID_POLIGONO"})
    df["ID_POLIGONO"] = df["ID_POLIGONO"].str.strip()
    return df[["ID_POLIGONO", "área_ha", "Municipio", "TpEl_mean", "TpPe_mean"]]

def cond(v):
    """Semáforo de condición según rendimiento (t/ha)."""
    if pd.isna(v): return "Sin dato"
    if v >= 4.3: return "Bueno"
    if v >= 3.5: return "Alerta 1"
    if v >= 2.8: return "Alerta 2"
    return "Emergencia"

@st.cache_data
def build_map_data():
    """
    Une coordenadas + atributos del EXT + rendimiento.
      * PREDICCION   -> rendimiento = predicho (resultados_modelo.csv)
      * ENTRENAMIENTO-> rendimiento = observado (ID_area_rendimiento...csv);
                        el 'predicho' es la predicción fuera de muestra (CV) si existe.
    Devuelve (df, lista_de_avisos).
    """
    avisos = []
    df = load_coords().merge(load_ext(), on="ID_POLIGONO", how="left")

    # Rendimiento observado (solo ENTRENAMIENTO tiene valor)
    obs = pd.read_csv("data/ID_area_rendimiento_70_30_Reto_AgroCebada.csv", encoding="utf-8-sig")
    obs = obs.rename(columns={"RENDIMIENTO_T_HA": "rendimiento_real"})[["ID_POLIGONO", "rendimiento_real"]]
    df = df.merge(obs, on="ID_POLIGONO", how="left")
    df.loc[df["CONJUNTO"] != "ENTRENAMIENTO", "rendimiento_real"] = np.nan

    # Predicciones del modelo (resultados_modelo.csv)
    res = pd.read_csv("data/resultados_modelo.csv", encoding="utf-8-sig")
    res = res.rename(columns={"RENDIMIENTO_PREDICHO_T_HA": "rendimiento_predicho"})
    res["ID_POLIGONO"] = res["ID_POLIGONO"].str.strip()
    df = df.merge(res[["ID_POLIGONO", "rendimiento_predicho"]], on="ID_POLIGONO", how="left")

    # Predicción fuera de muestra para ENTRENAMIENTO (opcional)
    try:
        oof = pd.read_csv("data/oof_entrenamiento.csv")[["ID_POLIGONO", "rendimiento_predicho_cv"]]
        df = df.merge(oof, on="ID_POLIGONO", how="left")
        m = df["CONJUNTO"] == "ENTRENAMIENTO"
        df.loc[m, "rendimiento_predicho"] = df.loc[m, "rendimiento_predicho_cv"]
        df = df.drop(columns="rendimiento_predicho_cv")
    except FileNotFoundError:
        pass

    # Valor que se dibuja en el mapa
    es_pred = df["CONJUNTO"] == "PREDICCION"
    df["rendimiento"] = np.where(es_pred, df["rendimiento_predicho"], df["rendimiento_real"])
    df["tipo_valor"]  = np.where(es_pred, "Predicho", "Observado")
    df["error_absoluto"] = (df["rendimiento_real"] - df["rendimiento_predicho"]).abs()
    df["condicion"] = df["rendimiento"].apply(cond)

    # Controles de consistencia
    n_sin = int(((df["CONJUNTO"] == "PREDICCION") & df["rendimiento_predicho"].isna()).sum())
    if n_sin:
        avisos.append(f"{n_sin} parcela(s) de PREDICCION no aparecen en resultados_modelo.csv.")
    extra = set(res["ID_POLIGONO"]) - set(df.loc[es_pred, "ID_POLIGONO"])
    if extra:
        avisos.append(f"resultados_modelo.csv trae {len(extra)} ID que no son de PREDICCION "
                      f"en las coordenadas: {', '.join(sorted(extra)[:5])}...")
    n_coord = int(df[["Latitud", "Longitud"]].isna().any(axis=1).sum())
    if n_coord:
        avisos.append(f"{n_coord} parcela(s) sin coordenadas; no se dibujan.")
        df = df.dropna(subset=["Latitud", "Longitud"])
    return df, avisos

try:
    df_map, avisos = build_map_data()
except FileNotFoundError as e:
    st.error(f"Falta un archivo de datos: `{e.filename}`. Revisa la carpeta `data/`.")
    st.stop()

MEDIA_REF = float(df_map["rendimiento_real"].mean())   # media observada (referencia del indicador)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="page-header">
  <h2> Mapa de parcelas</h2>
  <p>Visualización geoespacial de las 197 parcelas · Estado de Hidalgo, México</p>
</div>
""", unsafe_allow_html=True)

# Estado de los datos
n_pred = int((df_map["CONJUNTO"] == "PREDICCION").sum())
n_ent  = int((df_map["CONJUNTO"] == "ENTRENAMIENTO").sum())
st.markdown(f"""
<div class="info-box">
<strong>{n_pred} parcelas de PREDICCION</strong>: rendimiento <strong>predicho</strong> por el modelo
(<code>resultados_modelo.csv</code>). &nbsp;
<strong>{n_ent} parcelas de ENTRENAMIENTO</strong>: rendimiento <strong>observado</strong>;
en el tooltip se muestra además su predicción por validación cruzada y el error.
</div>
""", unsafe_allow_html=True)
for av in avisos:
    st.warning(av)

# ── Sidebar — controles ───────────────────────────────────────────────────────
with st.sidebar:
    # Filtro: conjunto
    conj_opts = ["Todos"] + sorted(df_map["CONJUNTO"].unique().tolist())
    conj_sel  = st.selectbox("Conjunto", conj_opts)

    # Filtro: condición
    cond_opts = sorted(df_map["condicion"].unique().tolist())
    cond_sel  = st.multiselect("Condición", cond_opts, default=cond_opts)

    # Variable de color
    color_var = st.radio("Colorear por",
                         ["Condición (semáforo)", "Rendimiento (gradiente)"])

    # Tamaño del marcador
    marker_sz = st.slider("Tamaño de marcadores", 6, 20, 10)

    st.divider()
    st.markdown("**Mapa base**")
    mapa_base = st.selectbox("Estilo",
                              ["carto-positron",
                               "open-street-map",
                               "carto-voyager",
                               "carto-darkmatter"])

# ── Filtrado ──────────────────────────────────────────────────────────────────
df_f = df_map.copy()
if conj_sel != "Todos":
    df_f = df_f[df_f["CONJUNTO"] == conj_sel]
df_f = df_f[df_f["condicion"].isin(cond_sel)]
if df_f.empty:
    st.warning("No hay parcelas con los filtros seleccionados.")
    st.stop()

df_f = df_f.copy()
df_f["hover_extra"] = np.where(
    df_f["CONJUNTO"] == "ENTRENAMIENTO",
    "Predicho (validación cruzada): " + df_f["rendimiento_predicho"].map("{:.3f}".format) + " t/ha<br>"
    + "Error absoluto: " + df_f["error_absoluto"].map("{:.3f}".format) + " t/ha",
    "Observado: no disponible",
)
CUSTOM = ["ID_POLIGONO","CONJUNTO","rendimiento","tipo_valor","condicion",
          "área_ha","Municipio","TpEl_mean","TpPe_mean","hover_extra"]

# ── Layout: mapa + panel lateral ──────────────────────────────────────────────
col_map, col_det = st.columns([4, 1], gap="medium")

COLORES = {
    "Bueno":      "#27AE60",
    "Alerta 1":   "#F1C40F",
    "Alerta 2":   "#E67E22",
    "Emergencia": "#C0392B",
}

with col_map:
    st.markdown('<p class="sec">Haz clic en una parcela para ver el detalle</p>',
                unsafe_allow_html=True)

    if color_var == "Condición (semáforo)":
        fig = px.scatter_map(
            df_f,
            lat="Latitud", lon="Longitud",
            color="condicion",
            color_discrete_map=COLORES,
            hover_name="ID_POLIGONO",
            custom_data=CUSTOM,
            size_max=marker_sz,
            zoom=10,
            center={"lat": df_f["Latitud"].mean(), "lon": df_f["Longitud"].mean()},
            map_style=mapa_base,
            height=560,
        )
    else:
        fig = px.scatter_map(
            df_f,
            lat="Latitud", lon="Longitud",
            color="rendimiento",
            color_continuous_scale=[
                (0.00, "#C0392B"),
                (0.35, "#E67E22"),
                (0.55, "#F1C40F"),
                (1.00, "#27AE60"),
            ],
            range_color=[df_f["rendimiento"].min(),
                         df_f["rendimiento"].max()],
            hover_name="ID_POLIGONO",
            custom_data=CUSTOM,
            zoom=10,
            center={"lat": df_f["Latitud"].mean(), "lon": df_f["Longitud"].mean()},
            map_style=mapa_base,
            height=560,
        )

    # Personalizar hover
    fig.update_traces(
        marker=dict(size=marker_sz, opacity=0.88),
        hovertemplate=(
            "<b>%{customdata[0]}</b><br>"
            "Municipio: %{customdata[6]}<br>"
            "Conjunto: %{customdata[1]}<br>"
            "Área: %{customdata[5]:.2f} ha<br>"
            "──────────────<br>"
            "<b>Rendimiento %{customdata[3]}: %{customdata[2]:.3f} t/ha</b><br>"
            "%{customdata[9]}<br>"
            "Condición: %{customdata[4]}<br>"
            "──────────────<br>"
            "Elevación: %{customdata[7]:.0f} m<br>"
            "Pendiente: %{customdata[8]:.2f}°"
            "<extra></extra>"
        )
    )
    fig.update_layout(
        margin=dict(t=0, b=0, l=0, r=0),
        legend=dict(
            title="Condición",
            bgcolor="rgba(255,255,255,0.85)",
            bordercolor="#ccc", borderwidth=1,
            font=dict(size=12, color="#17251B"),
            x=0.01, y=0.99,
        ),
        coloraxis_colorbar=dict(title="t/ha", len=0.6, y=0.5),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Selector de parcela para el panel de detalle
    st.markdown('<p class="sec">Seleccionar parcela manualmente</p>', unsafe_allow_html=True)
    parcela_sel = st.selectbox(
        "Parcela",
        options=df_f["ID_POLIGONO"].tolist(),
        help="También puedes pasar el cursor sobre el mapa para ver el detalle en el tooltip"
    )

with col_det:
    st.markdown('<p class="sec">Detalle</p>', unsafe_allow_html=True)

    if parcela_sel:
        row = df_f[df_f["ID_POLIGONO"] == parcela_sel].iloc[0]

        # Color de la condición
        cond_colors = {"Bueno":"#27AE60","Alerta 1":"#F1C40F",
                       "Alerta 2":"#E67E22","Emergencia":"#C0392B"}
        bg  = cond_colors.get(row["condicion"], "#95A5A6")
        txt = "#000" if row["condicion"] == "Alerta 1" else "#fff"

        st.markdown(f"""
        <div class="detail-card">
          <h4> {row['ID_POLIGONO']}</h4>
          <span class="cond-pill" style="background:{bg};color:{txt}">
            {row['condicion']}
          </span><br>
          <small><b>Municipio:</b> {row.get('Municipio','—')}</small><br>
          <small><b>Conjunto:</b> {row['CONJUNTO']}</small><br>
          <small><b>Área:</b> {row.get('área_ha','—'):.2f} ha</small>
          <hr style="margin:10px 0">
          <small><b>Lat:</b> {row['Latitud']:.6f}</small><br>
          <small><b>Lon:</b> {row['Longitud']:.6f}</small>
          <hr style="margin:10px 0">
          <b style="font-size:.95em"> Rendimiento</b><br>
          <big style="color:{bg};font-weight:800">{row['rendimiento']:.3f} t/ha</big>
          <p style="font-size:.75em;margin:2px 0">{row['tipo_valor'].lower()}</p>
        """, unsafe_allow_html=True)

        if row["CONJUNTO"] == "ENTRENAMIENTO":
            st.markdown(f"""
          <small><b>Predicho (CV):</b> {row['rendimiento_predicho']:.3f} t/ha</small><br>
          <small><b>Error abs.:</b> {row['error_absoluto']:.3f} t/ha</small>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
          <small style="color:#555">Rendimiento observado: no disponible (parcela de predicción)</small>
            """, unsafe_allow_html=True)

        st.markdown("""
          <hr style="margin:10px 0">
          <b style="font-size:.85em"> Topografía</b>
        """, unsafe_allow_html=True)
        st.markdown(f"""
          <small><b>Elevación:</b> {row.get('TpEl_mean',0):.0f} m</small><br>
          <small><b>Pendiente:</b> {row.get('TpPe_mean',0):.2f}°</small>
        </div>
        """, unsafe_allow_html=True)

        # Mini indicador de rendimiento
        st.markdown("<br>", unsafe_allow_html=True)
        rend_v = row["rendimiento"]
        fig_g = go.Figure(go.Indicator(
            mode="gauge+number",
            value=rend_v,
            number={"suffix":" t/ha","font":{"size":16,"color":"#17251B"}},
            gauge={
                "axis":{"range":[1.5, 6.0],"tickwidth":1,"tickfont_size":8},
                "bar":{"color":bg},
                "steps":[
                    {"range":[1.5, 2.8], "color":"#FFEBEE"},
                    {"range":[2.8, 3.5], "color":"#FFF3E0"},
                    {"range":[3.5, 4.3], "color":"#FFFDE7"},
                    {"range":[4.3, 6.0], "color":"#E8F5E9"},
                ],
                "threshold":{"line":{"color":"#1B5E20","width":2},
                             "thickness":.75,"value":MEDIA_REF},
            },
            domain={"x":[0,1],"y":[0,1]}
        ))
        fig_g.update_layout(height=160, margin=dict(t=10,b=10,l=10,r=10),
                             paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_g, use_container_width=True)

# ── Estadísticos del mapa ─────────────────────────────────────────────────────
st.divider()
col_s1, col_s2, col_s3 = st.columns(3, gap="medium")

with col_s1:
    st.markdown('<p class="sec">Distribución por condición (vista filtrada)</p>',
                unsafe_allow_html=True)
    cnt = df_f["condicion"].value_counts().reset_index()
    cnt.columns = ["Condición","Parcelas"]
    cnt["%"] = (cnt["Parcelas"]/cnt["Parcelas"].sum()*100).round(1)
    st.dataframe(cnt, hide_index=True, use_container_width=True,
                 column_config={"%": st.column_config.NumberColumn(format="%.1f %%")})

with col_s2:
    st.markdown('<p class="sec">Rendimiento promedio por municipio (vista filtrada)</p>',
                unsafe_allow_html=True)
    if "Municipio" in df_f.columns:
        mun_g = (df_f.groupby("Municipio")["rendimiento"]
                 .mean().sort_values(ascending=False).reset_index())
        fig_mun = px.bar(mun_g, x="rendimiento", y="Municipio",
                         orientation="h",
                         color="rendimiento",
                         color_continuous_scale=["#C0392B","#F1C40F","#27AE60"],
                         range_color=[2.5,5.0],
                         text_auto=".2f",
                         labels={"rendimiento":"Rendimiento promedio (t/ha)",
                                 "Municipio":""})
        fig_mun.update_layout(plot_bgcolor="white", paper_bgcolor="white",
                               height=260, margin=dict(t=5,b=10,l=10,r=10),
                               coloraxis_showscale=False,
                               yaxis=dict(tickfont_size=9))
        st.plotly_chart(fig_mun, use_container_width=True)

with col_s3:
    st.markdown('<p class="sec">Histograma de rendimiento (predicho u observado)</p>',
                unsafe_allow_html=True)
    fig_hist = px.histogram(df_f, x="rendimiento", nbins=20,
                             color="condicion", color_discrete_map=COLORES,
                             opacity=.82, barmode="overlay",
                             labels={"rendimiento":"Rendimiento (t/ha)",
                                     "condicion":"Condición"})
    fig_hist.add_vline(x=df_f["rendimiento"].mean(),
                       line_dash="dash", line_color="#37474F", line_width=1.5,
                       annotation_text=f"Media {df_f['rendimiento'].mean():.2f}",
                       annotation_font_size=9)
    fig_hist.update_layout(plot_bgcolor="white", paper_bgcolor="white",
                            height=260, margin=dict(t=10,b=40,l=10,r=10),
                            showlegend=False,
                            yaxis_title="Parcelas")
    st.plotly_chart(fig_hist, use_container_width=True)

# ── Tabla completa ────────────────────────────────────────────────────────────
with st.expander("  Tabla de todas las parcelas visibles"):
    cols_tab = ["ID_POLIGONO","Municipio","CONJUNTO","área_ha",
                "Latitud","Longitud","tipo_valor","rendimiento",
                "rendimiento_predicho","rendimiento_real","error_absoluto","condicion"]
    cols_tab = [c for c in cols_tab if c in df_f.columns]
    st.dataframe(df_f[cols_tab].round(3),
                 hide_index=True, use_container_width=True, height=350,
                 column_config={
                     "tipo_valor": st.column_config.TextColumn("Tipo de valor"),
                     "rendimiento": st.column_config.NumberColumn(
                         "Rendimiento mostrado (t/ha)", format="%.3f"),
                     "rendimiento_predicho": st.column_config.NumberColumn(
                         "Predicho (t/ha)", format="%.3f"),
                     "error_absoluto": st.column_config.NumberColumn(
                         "Error abs. (t/ha)", format="%.3f"),
                     "rendimiento_real": st.column_config.NumberColumn(
                         "Real (t/ha)", format="%.3f"),
                     "área_ha": st.column_config.NumberColumn(
                         "Área (ha)", format="%.2f"),
                 })
    csv_b = df_f[cols_tab].to_csv(index=False).encode("utf-8")
    st.download_button("⬇  Descargar datos del mapa (.csv)",
                       csv_b, "mapa_parcelas.csv", "text/csv")
