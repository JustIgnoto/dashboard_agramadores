"""
Regenera los archivos de data/ que usa la página "Modelo" a partir de la tabla
maestra del repositorio Agramadores (Code/Prediccion/Version_2).

Replica EXACTAMENTE el pipeline de Version_2 (filter_lasso.py + models_opt.py):
  1) Lasso (alpha por GridSearch, CV 5-fold, seed 42) sobre todas las variables
  2) Se conservan las variables con coeficiente != 0
  3) Lasso / Poly+Ridge / SVR(RBF) sobre las variables seleccionadas
  4) Predicciones OOF (5-fold) para métricas y predicho-vs-real

Uso (desde la raíz del dashboard):
  python scripts/generar_artefactos_modelo.py \
      --tabla ruta/Version_2/Tabla_Maestra_Mensual_Escalada_MinMax.csv \
      [--resultados data/resultados_modelo.csv]

Requiere: scikit-learn (no está en requirements.txt porque el dashboard no lo usa).
"""
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import SVR

ap = argparse.ArgumentParser()
ap.add_argument("--tabla", required=True)
ap.add_argument("--resultados", default="data/resultados_modelo.csv")
ap.add_argument("--salida", default="data")
a = ap.parse_args()
out = Path(a.salida)

df = pd.read_csv(a.tabla)
tr = df[df.CONJUNTO == "ENTRENAMIENTO"]
pr = df[df.CONJUNTO == "PREDICCION"]
EXCL = ["ID_POLIGONO", "CONJUNTO", "RENDIMIENTO_T_HA", "AREA_HA"]
X = tr.drop(columns=EXCL, errors="ignore").select_dtypes(include=[np.number])
y = tr["RENDIMIENTO_T_HA"]
cv = KFold(n_splits=5, shuffle=True, random_state=42)
imp = lambda est: Pipeline([("imputer", SimpleImputer(strategy="median")), est])


def entrenar(pipe, grid, Xt, yt):
    g = GridSearchCV(pipe, grid, cv=cv, scoring="neg_mean_squared_error", n_jobs=-1).fit(Xt, yt)
    m, oof = g.best_estimator_, np.zeros(len(yt))
    for i, j in cv.split(Xt):
        m.fit(Xt.iloc[i], yt.iloc[i])
        oof[j] = m.predict(Xt.iloc[j])
    m.fit(Xt, yt)
    return m, oof, g.best_params_


# 1) Selección de variables con Lasso
lasso_b, _, p_lasso = entrenar(
    imp(("lasso", Lasso(random_state=42, max_iter=10000))),
    {"lasso__alpha": [0.001, 0.01, 0.1, 1, 10.0]}, X, y)
coef = pd.Series(lasso_b.named_steps["lasso"].coef_, index=X.columns)
coef = coef[coef != 0]
sel = coef.abs().sort_values(ascending=False).index.tolist()
Xo, Xp = tr[sel], pr[sel]

# 2) Modelos sobre variables seleccionadas
modelos = {
    "Lasso": (imp(("lasso", Lasso(random_state=42, max_iter=10000))),
              {"lasso__alpha": [0.001, 0.01, 0.1, 1]}),
    "Poly + Ridge": (Pipeline([("imputer", SimpleImputer(strategy="median")),
                               ("poly", PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)),
                               ("ridge", Ridge(random_state=42, max_iter=10000))]),
                     {"ridge__alpha": [0.01, 0.1, 1, 10.0]}),
    "SVR (RBF)": (imp(("svr", SVR(kernel="rbf"))),
                  {"svr__C": [0.1, 1, 10, 50, 100], "svr__gamma": ["scale", "auto", 0.01, 0.05],
                   "svr__epsilon": [0.01, 0.05, 0.1]}),
}
met, folds, fin = [], [], {}
for nombre, (pipe, grid) in modelos.items():
    m, oof, bp = entrenar(pipe, grid, Xo, y)
    met.append({"Modelo": nombre,
                "RMSE": np.sqrt(mean_squared_error(y, oof)),
                "MAE": mean_absolute_error(y, oof), "R2": r2_score(y, oof)})
    for k, (_, j) in enumerate(cv.split(Xo), 1):
        folds.append({"Fold": f"Fold {k}", "Modelo": nombre,
                      "RMSE": float(np.sqrt(mean_squared_error(y.iloc[j], oof[j])))})
    fin[nombre] = (m, oof, bp)

svr, oof_svr, bp_svr = fin["SVR (RBF)"]


def grupo(v):
    if v.startswith(("Tx", "Tm", "Pr")): return "Clima (CHIRPS)"
    if v.startswith("Tp"): return "Topografía (INEGI)"
    if v.startswith("S2_L8"): return "Sentinel-2 / Landsat 8"
    if v.startswith("Planet"): return "Planet"
    return "Otras"


pd.DataFrame({"feature": sel, "importancia": coef[sel].abs().values, "coeficiente": coef[sel].values,
              "grupo": [grupo(v) for v in sel]}).to_csv(out / "importancia_features.csv", index=False)
pd.DataFrame(met).to_csv(out / "metricas_cv.csv", index=False)
pd.DataFrame(folds).to_csv(out / "metricas_folds.csv", index=False)
pd.DataFrame({"ID_POLIGONO": tr.ID_POLIGONO.values, "rendimiento_real": y.values,
              "rendimiento_predicho_cv": oof_svr}).to_csv(out / "oof_entrenamiento.csv", index=False)
(out / "resumen_modelo.json").write_text(json.dumps({
    "n_variables_candidatas": int(X.shape[1]), "n_variables_seleccionadas": len(sel),
    "candidatas_por_grupo": pd.Series([grupo(c) for c in X.columns]).value_counts().to_dict(),
    "lasso_alpha_seleccion": p_lasso["lasso__alpha"],
    "svr_C": bp_svr["svr__C"], "svr_gamma": bp_svr["svr__gamma"], "svr_epsilon": bp_svr["svr__epsilon"],
    "n_entrenamiento": int(len(tr)), "n_prediccion": int(len(pr)),
}, indent=2), encoding="utf-8")

# 3) Verificación contra resultados_modelo.csv (si existe)
if Path(a.resultados).exists():
    r = pd.read_csv(a.resultados)
    p = pd.DataFrame({"ID_POLIGONO": pr.ID_POLIGONO.values, "p": svr.predict(Xp)})
    m = r.merge(p, on="ID_POLIGONO")
    print(f"Verificación vs {a.resultados}: {len(m)}/{len(r)} parcelas, "
          f"dif. máx = {(m.RENDIMIENTO_PREDICHO_T_HA - m.p).abs().max():.4f} t/ha")
print(pd.DataFrame(met).round(4).to_string(index=False))
