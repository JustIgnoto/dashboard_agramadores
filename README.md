# AgroCebada 2026 — Dashboard web

Dashboard multipágina creado con Streamlit. El proyecto está preparado para ejecutarse localmente o publicarse en Streamlit Community Cloud. Se retiraron los emojis del contenido visible y se mantuvieron los ajustes de contraste.

## Ejecutar localmente

1. Instala Python 3.10 o posterior.
2. En una terminal, entra en la carpeta del proyecto.
3. Instala las dependencias: `pip install -r requirements.txt`.
4. Ejecuta desde la raíz del proyecto: `streamlit run Inicio.py`.

## Publicar como página web

1. Descomprime el ZIP y sube el contenido de esta carpeta a un repositorio de GitHub. Incluye `Inicio.py`, `pages/`, `data/`, `.streamlit/config.toml` y `requirements.txt`.
2. Abre https://share.streamlit.io/ e inicia sesión con GitHub.
3. Selecciona **Create app** y elige el repositorio, la rama `main` y el archivo principal `Inicio.py`.
4. Pulsa **Deploy**. Streamlit generará una dirección pública terminada en `streamlit.app`.

No se puede crear la dirección pública desde este paquete sin acceso a tu cuenta de GitHub/Streamlit; esos pasos requieren que inicies sesión y autorices la publicación.

## Datos

Todo vive en `data/`:

| Archivo | Lo usa | Origen |
|---|---|---|
| `Conjunto_datos_EXT_AgroCebada.csv` | Inicio, Datos, Mapa, Modelo | `DataSet/` del repo Agramadores |
| `Coordenadas_Parcelas_Dashboard.csv` | Mapa | `Code/Data/Data_Proto4/Ubicacion/` |
| `resultados_modelo.csv` | Mapa, Modelo | Predicciones finales (59 parcelas de PREDICCION) |
| `ID_area_rendimiento_70_30_Reto_AgroCebada.csv` | Mapa, Modelo | Rendimiento observado (138 parcelas de ENTRENAMIENTO) |
| `importancia_features.csv`, `metricas_cv.csv`, `metricas_folds.csv`, `oof_entrenamiento.csv`, `resumen_modelo.json` | Modelo, Mapa | Generados con `scripts/generar_artefactos_modelo.py` |

`resultados_modelo.csv` corresponde a **Prediccion/Version_2** del repo (SVR RBF + selección de variables con Lasso).

### Regenerar los artefactos del modelo

Requiere `scikit-learn` (solo para este script, el dashboard no lo necesita):

```
pip install scikit-learn
python scripts/generar_artefactos_modelo.py --tabla ruta/a/Version_2/Tabla_Maestra_Mensual_Escalada_MinMax.csv
```

El script reentrena el pipeline de Version_2 con la misma semilla y verifica que sus predicciones coincidan con `data/resultados_modelo.csv`.
