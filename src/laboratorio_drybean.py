# =============================================================
# LABORATORIO PRÁCTICO DE MACHINE LEARNING — DRY BEAN DATASET
# CRISP-DM + TDSP + Scrum ML
# =============================================================
# Equipo:
#   - Daniel Felipe Zamora   | ML Engineer
#   - Diego Ortiz            | Data Engineer / Analyst
#   - Jairo Andrés Pérez     | Scrum Master
#   - Manuel Enrique Luna    | Product Owner
# Dataset: https://archive.ics.uci.edu/dataset/602/dry+bean+dataset
# =============================================================

# ── PASO 1: INSTALACIÓN (ejecutar solo una vez en terminal) ──
# pip install ucimlrepo pandas numpy matplotlib scikit-learn joblib

# ── PASO 2: IMPORTAR LIBRERÍAS ────────────────────────────────
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, f1_score,
    classification_report, ConfusionMatrixDisplay
)
import joblib
import os

# Crear carpetas de salida si no existen
os.makedirs("outputs/models",  exist_ok=True)
os.makedirs("outputs/reports", exist_ok=True)

# ── PASO 3: CARGAR EL DATASET DESDE UCI ──────────────────────
# (CRISP-DM Fase 2: Comprensión de datos)
print("=" * 60)
print("  CARGANDO DATASET DESDE UCI...")
print("=" * 60)

dry_bean = fetch_ucirepo(id=602)
X = dry_bean.data.features
y = dry_bean.data.targets
df = pd.concat([X, y], axis=1)

print("\nPrimeras filas del dataset:")
print(df.head())

# ── ACTIVIDAD: Responder preguntas del paso 3 ─────────────────
numero_columnas = df.shape[1]
nombre_objetivo = y.columns[0]
valores_unicos  = y[nombre_objetivo].unique()

print(f"\n1. El dataset completo tiene {numero_columnas} columnas.")
print(f"2. La variable objetivo se llama: '{nombre_objetivo}'")
print(f"3. Tipo de dato de la variable objetivo: {y[nombre_objetivo].dtype}")
print(f"   Valores únicos: {valores_unicos}")
print("   → Es un problema de CLASIFICACIÓN MULTICLASE")

# ── PASO 4: COMPRENSIÓN DE DATOS — CRISP-DM ──────────────────
print("\n" + "=" * 60)
print("  PASO 4: COMPRENSIÓN DE DATOS")
print("=" * 60)

print(f"\nForma del dataset: {df.shape}")
print("\nInformación general:")
print(df.info())
print("\nEstadísticas descriptivas:")
print(df.describe().round(2))

# ── PASO 5: INTERPRETACIÓN DETALLADA ─────────────────────────
num_registros = df.shape[0]
num_variables = df.shape[1]
print(f"\n1. Número de registros: {num_registros}")
print(f"2. Número de variables: {num_variables}")

print("\n3. Tipos de datos:")
print(df.dtypes.value_counts().to_string())

# Detección de outliers con método IQR
print("\n4. Columnas con posibles valores extremos (IQR):")
df_numerico = df.select_dtypes(include=["float64", "int64"])
Q1  = df_numerico.quantile(0.25)
Q3  = df_numerico.quantile(0.75)
IQR = Q3 - Q1
outliers = ((df_numerico < (Q1 - 1.5 * IQR)) | (df_numerico > (Q3 + 1.5 * IQR))).sum()
columnas_con_outliers = outliers[outliers > 0]
if not columnas_con_outliers.empty:
    print(columnas_con_outliers.to_string())
else:
    print("No se detectaron valores extremos significativos.")

# Columnas no numéricas aparte de Class
columnas_no_numericas = df.select_dtypes(exclude=["float64", "int64"]).columns.tolist()
if "Class" in columnas_no_numericas:
    columnas_no_numericas.remove("Class")

print("\n5. ¿Hay columnas no numéricas aparte de la clase?")
if len(columnas_no_numericas) == 0:
    print("   → No, la clase es la única columna con texto/categorías.")
else:
    print(f"   → Sí: {columnas_no_numericas}")

# ── PASO 6: REVISAR NULOS Y DUPLICADOS ───────────────────────
print("\n" + "=" * 60)
print("  PASO 6: CALIDAD DE DATOS")
print("=" * 60)

missing_values  = df.isna().sum()
duplicated_rows = df.duplicated().sum()

print(f"\nValores nulos por columna:\n{missing_values}")
print(f"\nFilas duplicadas: {duplicated_rows}")

# Decisión técnica: eliminar duplicados
if duplicated_rows > 0:
    df = df.drop_duplicates().reset_index(drop=True)
    print(f"✔ Duplicados eliminados. Nuevo tamaño: {df.shape}")
else:
    print("✔ Sin duplicados. Dataset limpio.")

# ── PASO 7: ANALIZAR LA VARIABLE OBJETIVO ────────────────────
print("\n" + "=" * 60)
print("  PASO 7: DISTRIBUCIÓN DE CLASES")
print("=" * 60)

class_counts = df["Class"].value_counts()
print(f"\n{class_counts}")

# Gráfica de distribución
class_counts.plot(kind="bar", color="#c0392b", edgecolor="white")
plt.title("Distribución de clases — Dry Bean Dataset")
plt.xlabel("Clase de frijol")
plt.ylabel("Cantidad de registros")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("outputs/reports/distribucion_clases.png", dpi=150)
plt.show()

# Análisis de balance
total_registros = len(df)
porcentajes = (df["Class"].value_counts() / total_registros) * 100

print("\nPorcentaje por clase:")
print(porcentajes.round(2).astype(str) + " %")

clase_mayor = porcentajes.max()
clase_menor = porcentajes.min()
ratio       = clase_mayor / clase_menor

print(f"\nRatio mayoritaria/minoritaria: {ratio:.1f}x")
if ratio > 2:
    print("→ Las clases NO están balanceadas.")
    print("→ Se usará F1 Macro como métrica principal y class_weight='balanced'.")

# ── PASO 8: SEPARAR VARIABLES ────────────────────────────────
print("\n" + "=" * 60)
print("  PASO 8: SEPARACIÓN DE VARIABLES")
print("=" * 60)

X = df.drop(columns="Class")
y = df["Class"]

print(f"  X (predictoras): {X.shape}")
print(f"  y (objetivo):    {y.shape}")

# ── PASO 9: TRAIN / TEST SPLIT ───────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,   # Semilla fija → reproducibilidad (TDSP)
    stratify=y         # Mantiene proporciones de clases
)

print(f"\n  Train: {X_train.shape} | Test: {X_test.shape}")

# ── PASO 10: MODELO BASELINE — LOGISTIC REGRESSION ───────────
# (CRISP-DM Fase 4: Modelado)
print("\n" + "=" * 60)
print("  PASO 10: MODELO BASELINE — LOGISTIC REGRESSION")
print("=" * 60)

baseline_model = Pipeline(steps=[
    ("scaler",     StandardScaler()),
    ("classifier", LogisticRegression(max_iter=1000, random_state=42))
])
baseline_model.fit(X_train, y_train)

# ── PASO 11: EVALUAR BASELINE ────────────────────────────────
y_pred_baseline   = baseline_model.predict(X_test)
accuracy_baseline = accuracy_score(y_test, y_pred_baseline)
f1_baseline       = f1_score(y_test, y_pred_baseline, average="macro")

print(f"\n  Accuracy baseline : {accuracy_baseline:.4f}")
print(f"  F1 macro baseline : {f1_baseline:.4f}")
print("\nReporte completo:")
print(classification_report(y_test, y_pred_baseline))

# Nota: F1 Macro es más informativo que Accuracy en datasets desbalanceados
# porque evalúa el rendimiento dándole la misma importancia a todas las clases.
# El Accuracy puede parecer alto aunque el modelo falle con las clases minoritarias.

# ── PASO 12: MODELO MEJORADO — RANDOM FOREST ─────────────────
print("\n" + "=" * 60)
print("  PASO 12: MODELO MEJORADO — RANDOM FOREST")
print("=" * 60)

forest_model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced"   # Compensa el desbalance entre clases
)
forest_model.fit(X_train, y_train)

# ── PASO 13: EVALUAR RANDOM FOREST ───────────────────────────
y_pred_forest  = forest_model.predict(X_test)
accuracy_forest = accuracy_score(y_test, y_pred_forest)
f1_forest       = f1_score(y_test, y_pred_forest, average="macro")

print(f"\n  Accuracy Random Forest : {accuracy_forest:.4f}")
print(f"  F1 macro Random Forest : {f1_forest:.4f}")
print("\nReporte completo:")
print(classification_report(y_test, y_pred_forest))

# ── PASO 14: COMPARAR MODELOS ────────────────────────────────
# (CRISP-DM Fase 5: Evaluación)
print("\n" + "=" * 60)
print("  PASO 14: COMPARACIÓN DE MODELOS")
print("=" * 60)

results = pd.DataFrame({
    "modelo":   ["Logistic Regression", "Random Forest"],
    "accuracy": [accuracy_baseline, accuracy_forest],
    "f1_macro": [f1_baseline, f1_forest]
})
print(f"\n{results.to_string(index=False)}")

mejor = "Logistic Regression" if f1_baseline >= f1_forest else "Random Forest"
print(f"\n🏆 Mejor modelo según F1 Macro: {mejor}")

# ── PASO 15: MATRIZ DE CONFUSIÓN ─────────────────────────────
print("\n  Generando matriz de confusión...")
ConfusionMatrixDisplay.from_predictions(
    y_test,
    y_pred_forest,
    xticks_rotation=45
)
plt.title("Matriz de confusión - Random Forest")
plt.tight_layout()
plt.savefig("outputs/reports/matriz_confusion.png", dpi=150)
plt.show()

# Las clases que más se confunden son DERMASON y SIRA, ya que comparten
# perfiles geométricos similares (área, perímetro, redondez, longitud de ejes).

# ── PASO 16: IMPORTANCIA DE VARIABLES ────────────────────────
print("\n" + "=" * 60)
print("  PASO 16: IMPORTANCIA DE VARIABLES")
print("=" * 60)

feature_importance = pd.DataFrame({
    "feature":    X.columns,
    "importance": forest_model.feature_importances_
}).sort_values(by="importance", ascending=False)

print(f"\nTop 10 variables más importantes:")
print(feature_importance.head(10).to_string(index=False))

feature_importance.head(10).plot(
    x="feature", y="importance", kind="bar",
    color="#2980b9", edgecolor="white", legend=False
)
plt.title("Top 10 variables más importantes — Random Forest")
plt.xlabel("Variable")
plt.ylabel("Importancia")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("outputs/reports/importancia_variables.png", dpi=150)
plt.show()

# ── PASO 17: GUARDAR MODELO ──────────────────────────────────
# (CRISP-DM Fase 6: Despliegue — TDSP)
print("\n" + "=" * 60)
print("  PASO 17: GUARDAR MODELO")
print("=" * 60)

joblib.dump(forest_model, "outputs/models/random_forest_drybean.joblib")
print("  ✔ Modelo guardado en outputs/models/random_forest_drybean.joblib")

# ── PASO 18: CARGAR MODELO Y HACER PREDICCIÓN ────────────────
print("\n" + "=" * 60)
print("  PASO 18: CARGAR MODELO Y PREDECIR")
print("=" * 60)

loaded_model = joblib.load("outputs/models/random_forest_drybean.joblib")
sample       = X_test.iloc[[0]]
prediction   = loaded_model.predict(sample)

print(f"  Predicción : {prediction[0]}")
print(f"  Valor real : {y_test.iloc[0]}")

# Función reutilizable de predicción
def predict_bean_class(model, input_data):
    """Clasifica una muestra de frijol y retorna la variedad predicha."""
    prediction = model.predict(input_data)
    return prediction[0]

example_prediction = predict_bean_class(loaded_model, sample)
print(f"  Función predict_bean_class → {example_prediction}")

# ── RESUMEN FINAL ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("  ✅ LABORATORIO COMPLETADO")
print("=" * 60)
print(f"\n  Baseline (Logistic Regression)")
print(f"    Accuracy : {accuracy_baseline:.4f}")
print(f"    F1 Macro : {f1_baseline:.4f}")
print(f"\n  Mejorado (Random Forest)")
print(f"    Accuracy : {accuracy_forest:.4f}")
print(f"    F1 Macro : {f1_forest:.4f}")
print(f"\n  🏆 Mejor modelo: {mejor}")
print("\n  Archivos generados:")
print("    outputs/reports/distribucion_clases.png")
print("    outputs/reports/matriz_confusion.png")
print("    outputs/reports/importancia_variables.png")
print("    outputs/models/random_forest_drybean.joblib")
print("=" * 60)
