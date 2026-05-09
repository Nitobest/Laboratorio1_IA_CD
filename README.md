# 🫘 Clasificación de Variedades de Frijol Seco
### Laboratorio de Machine Learning con Datos Reales
#### CRISP-DM + TDSP + Scrum ML — Dry Bean Dataset (UCI)

---

> **Institución:** Universidad Autónoma de Occidente
> **Curso:** Machine Learning / Inteligencia Artificial
> **Dataset:** [Dry Bean Dataset — UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/602/dry+bean+dataset)
> **Lenguaje:** Python 3.x
> **Duración estimada:** 2 a 4 sesiones de laboratorio
> **Integrantes:**
> - Daniel Felipe Zamora
> - Diego Ortiz
> - Jairo Andrés Pérez
> - Manuel Enrique Luna

---

## 📋 Tabla de Contenidos

1. [Descripción del problema](#-descripción-del-problema)
2. [Dataset](#-dataset)
3. [Equipo de trabajo](#-equipo-de-trabajo)
4. [Metodologías aplicadas](#-metodologías-aplicadas)
5. [Estructura del proyecto](#-estructura-del-proyecto-tdsp)
6. [Product Backlog](#-product-backlog--scrum-ml)
7. [Planeación de Sprints](#-planeación-de-sprints)
8. [Resultados obtenidos](#-resultados-obtenidos)
9. [Instalación y ejecución](#-instalación-y-ejecución)
10. [Archivos generados](#-archivos-generados)
11. [Conclusiones](#-conclusiones)
12. [Definition of Done](#-definition-of-done)

---

## 📌 Descripción del problema

Una empresa agrícola desea **automatizar la clasificación de granos de frijol seco** para mejorar su proceso de control de calidad. Actualmente, parte de la clasificación se realiza de forma manual, lo cual genera errores y demoras.

El equipo de datos debe construir un modelo de Machine Learning que **prediga la variedad de frijol** a partir de características geométricas extraídas de imágenes digitales de los granos.

| Detalle | Valor |
|---|---|
| Variable objetivo | `Class` (variedad del frijol) |
| Tipo de problema | Clasificación multiclase |
| Número de clases | 7 variedades |
| Métrica principal | F1 Macro |

---

## 🌱 Dataset

- **Nombre:** Dry Bean Dataset
- **Fuente:** UCI Machine Learning Repository
- **URL:** https://archive.ics.uci.edu/dataset/602/dry+bean+dataset
- **Instancias:** 13.611
- **Variables predictoras:** 16 (características geométricas)
- **Variable objetivo:** `Class`

### Variedades de frijol (clases)

| Clase | Instancias |
|---|---|
| DERMASON | 3.546 |
| SIRA | 2.636 |
| SEKER | 2.027 |
| HOROZ | 1.928 |
| CALI | 1.630 |
| BARBUNYA | 1.322 |
| BOMBAY | 522 |

### Variables del dataset

| Variable | Descripción |
|---|---|
| `Area` | Área del grano en píxeles |
| `Perimeter` | Perímetro del grano |
| `MajorAxisLength` | Longitud del eje mayor |
| `MinorAxisLength` | Longitud del eje menor |
| `AspectRation` | Relación entre ejes |
| `Eccentricity` | Excentricidad de la elipse |
| `ConvexArea` | Área del casco convexo |
| `EquivDiameter` | Diámetro equivalente |
| `Extent` | Relación área / caja contenedora |
| `Solidity` | Solidez del grano |
| `roundness` | Redondez del contorno |
| `Compactness` | Compacidad del grano |
| `ShapeFactor1–4` | Factores de forma derivados |

---

## 👥 Equipo de trabajo

Equipo de 4 integrantes organizado bajo el marco **Scrum ML**.

| # | Integrante | Rol Scrum ML | Responsabilidades |
|---|---|---|---|
| 1 | **Daniel Felipe Zamora** | 🤖 ML Engineer | Entrenamiento, evaluación, comparación de modelos y despliegue |
| 2 | **Diego Ortiz** | 🔧 Data Engineer / Analyst | Carga, limpieza, análisis exploratorio y preparación del dataset |
| 3 | **Jairo Andrés Pérez** | 🛡️ Scrum Master | Coordina reuniones, gestiona el tablero y elimina bloqueos del equipo |
| 4 | **Manuel Enrique Luna** | 🎯 Product Owner | Define objetivos, prioriza el backlog y valida entregables finales |

---

## 🔄 Metodologías aplicadas

### CRISP-DM
Estructura del ciclo de vida del proyecto de minería de datos:

```
1. Comprensión del negocio  →  Problema agrícola de clasificación automática
2. Comprensión de datos     →  EDA, calidad, distribución de clases
3. Preparación de datos     →  Limpieza, escalamiento, train/test split
4. Modelado                 →  Logistic Regression + Random Forest
5. Evaluación               →  Accuracy, F1 Macro, matriz de confusión
6. Despliegue               →  Modelo guardado (.joblib) + función de predicción
```

### TDSP — Team Data Science Process
- Repositorio estructurado con carpetas estándar
- Datos versionados en `data/raw/` y `data/processed/`
- Modelos guardados en `outputs/models/`
- Reproducibilidad garantizada con `random_state=42`
- Documentación técnica centralizada en `README.md`

### Scrum ML
- Roles definidos: Product Owner, Scrum Master, Data Engineer, ML Engineer
- Backlog con historias de usuario priorizadas
- 3 Sprints con entregables concretos y criterios de aceptación
- Definition of Done por tarea
- Daily Scrum y retrospectiva por sprint

---

## 📁 Estructura del proyecto (TDSP)

```
laboratorio_drybean_ml/
│
├── data/
│   ├── raw/
│   │   └── dry_bean_raw.csv               # Dataset original sin modificar
│   └── processed/
│       └── dry_bean_clean.csv             # Dataset limpio y transformado
│
├── notebooks/
│   └── 01_laboratorio_drybean.ipynb       # Notebook de exploración (opcional)
│
├── outputs/
│   ├── models/
│   │   └── random_forest_drybean.joblib   # Modelo entrenado serializado
│   └── reports/
│       ├── distribucion_clases.png        # Gráfica de balance de clases
│       ├── matriz_confusion.png           # Matriz de confusión Random Forest
│       ├── importancia_variables.png      # Top 10 variables más importantes
│       ├── comparacion_modelos.csv        # Tabla comparativa de métricas
│       └── predicciones_drybean.csv       # Predicciones reales vs predichas
│
├── src/
│   └── laboratorio_drybean.py             # Script principal (18 pasos completos)
│
├── requirements.txt                       # Dependencias del proyecto
├── .gitignore                             # Archivos excluidos del repositorio
└── README.md                              # Este archivo
```

---

## 📝 Product Backlog — Scrum ML

| ID | Historia de usuario | Prioridad | Responsable | Estado |
|---|---|---|---|---|
| PB-01 | Como analista, quiero cargar el dataset real para analizarlo | 🔴 Alta | Jairo Pérez | ✅ Hecho |
| PB-02 | Como científico de datos, quiero revisar la calidad de los datos | 🔴 Alta | Diego Ortiz | ✅ Hecho |
| PB-03 | Como equipo, queremos un modelo base para tener punto de comparación | 🔴 Alta | Daniel Felipe Zamora | ✅ Hecho |
| PB-04 | Como equipo, queremos mejorar el desempeño con un modelo alternativo | 🟡 Media | Daniel Felipe Zamora | ✅ Hecho |
| PB-05 | Como equipo, queremos comparar modelos con métricas claras | 🔴 Alta | Todo el equipo | ✅ Hecho |
| PB-06 | Como usuario final, quiero entender el resultado visualmente | 🟡 Media | Diego Ortiz | ✅ Hecho |
| PB-07 | Como equipo técnico, queremos reproducibilidad del modelo | 🔴 Alta | Daniel Felipe Zamora | ✅ Hecho |
| PB-08 | Como equipo, queremos documentación profesional del proyecto | 🔴 Alta | Manuel Enrique Luna | ✅ Hecho |

---

## 🏃 Planeación de Sprints

### Sprint 1 — Comprensión y Preparación de Datos
**Responsable principal:** Diego Ortiz + Jairo Pérez
**Duración estimada:** 1 sesión

| Tarea | Responsable | Estado |
|---|---|---|
| Crear repositorio y estructura TDSP | Jairo Pérez | ✅ |
| Configurar entorno virtual e instalar dependencias | Todo el equipo | ✅ |
| Cargar dataset desde fuente oficial | Diego Ortiz | ✅ |
| Análisis exploratorio: forma, tipos, estadísticas | Diego Ortiz | ✅ |
| Revisar valores nulos y duplicados | Diego Ortiz | ✅ |
| Graficar distribución de clases | Diego Ortiz| ✅ |
| Separar variables predictoras y objetivo | Diego Ortiz| ✅ |
| Dividir datos en train/test (80/20, stratify) | Diego Ortiz | ✅ |

**Entregables Sprint 1:**
- Dataset limpio guardado en `data/processed/`
- Análisis exploratorio documentado
- Separación train/test reproducible

---

### Sprint 2 — Modelado y Evaluación
**Responsable principal:** Zamora, Daniel Felipe  
**Duración estimada:** 1–2 sesiones

| Tarea | Responsable | Estado |
|---|---|---|
| Entrenar modelo baseline (Logistic Regression + Pipeline) | Daniel Felipe Zamora | ✅ |
| Evaluar baseline: Accuracy y F1 Macro | Daniel Felipe Zamora| ✅ |
| Entrenar modelo mejorado (Random Forest) | Daniel Felipe Zamora| ✅ |
| Evaluar Random Forest: Accuracy y F1 Macro | Daniel Felipe Zamora | ✅ |
| Comparar ambos modelos en tabla | Todo el equipo | ✅ |
| Generar matriz de confusión | Daniel Felipe Zamora| ✅ |
| Analizar importancia de variables | Daniel Felipe Zamora| ✅ |

**Entregables Sprint 2:**
- Dos modelos entrenados y evaluados
- Tabla comparativa de métricas
- Gráficas de confusión e importancia de variables

---

### Sprint 3 — Despliegue y Documentación
**Responsable principal:** Luna, Manuel Enrique  
**Duración estimada:** 1 sesión

| Tarea | Responsable | Estado |
|---|---|---|
| Guardar modelo final con `joblib` | Daniel Felipe Zamora| ✅ |
| Cargar modelo y probar predicción individual | Daniel Felipe Zamora | ✅ |
| Crear función reutilizable `predict_bean_class()` | Daniel Felipe Zamora | ✅ |
| Exportar predicciones a CSV | Diego Ortiz| ✅ |
| Redactar README profesional | Manuel Enrique Luna| ✅ |
| Preparar evidencia Scrum ML | Jairo Pérez | ✅ |

**Entregables Sprint 3:**
- Modelo `.joblib` listo para reutilizar
- Función de predicción documentada
- README profesional completo
- Reporte y evidencias Scrum ML

---

## 📊 Resultados obtenidos

### Comparación de modelos

| Modelo | Accuracy | F1 Macro |
|---|---|---|
| Logistic Regression (baseline) | 86.2% | 84.5% |
| Random Forest | 84.4% | 82.4% |

### 🏆 Modelo seleccionado: Logistic Regression

**Criterio de decisión (CRISP-DM — Fase de Evaluación):**
- Se priorizó el **F1 Macro** por encima del Accuracy, ya que las clases no están perfectamente balanceadas (BOMBAY tiene solo 522 instancias frente a 3.546 de DERMASON).
- La Regresión Logística obtuvo mejor F1 Macro y es un modelo más simple e interpretable.
- Random Forest fue útil para extraer la **importancia de variables**.

### Variables más importantes según Random Forest

| # | Variable | Importancia |
|---|---|---|
| 1 | ConvexArea | 14.6% |
| 2 | EquivDiameter | 14.5% |
| 3 | Area | 14.2% |
| 4 | MajorAxisLength | 8.6% |
| 5 | MinorAxisLength | 8.0% |

> Las variables relacionadas con el **tamaño del grano** (área y diámetro) son las más discriminativas entre variedades, lo cual es coherente con las diferencias físicas visibles entre tipos de frijol.

---

## ⚙️ Instalación y ejecución

### 1. Clonar el repositorio
```bash
git clone <url-del-repositorio>
cd laboratorio_drybean_ml
```

### 2. Crear entorno virtual

**Linux / Mac:**
```bash
python -m venv .venv
source .venv/bin/activate
```

**Windows PowerShell:**
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 3. Instalar dependencias
```bash
pip install -r requirements.txt
```

### 4. Ejecutar el laboratorio completo
```bash
python src/laboratorio_drybean.py
```

### 5. (Opcional) Abrir notebook interactivo
```bash
jupyter notebook notebooks/01_laboratorio_drybean.ipynb
```

---

## 📦 Archivos generados al ejecutar

| Archivo | Descripción |
|---|---|
| `data/processed/dry_bean_clean.csv` | Dataset sin duplicados ni nulos |
| `outputs/reports/distribucion_clases.png` | Gráfica de balance de clases |
| `outputs/reports/matriz_confusion.png` | Matriz de confusión del mejor modelo |
| `outputs/reports/importancia_variables.png` | Top 10 variables más importantes |
| `outputs/reports/comparacion_modelos.csv` | Tabla con Accuracy y F1 Macro |
| `outputs/reports/predicciones_drybean.csv` | Predicciones reales vs predichas |
| `outputs/models/random_forest_drybean.joblib` | Modelo Random Forest serializado |

---

## 💡 Conclusiones

1. **CRISP-DM** estructuró el proyecto de forma lógica y ordenada, dejando claro en cada momento qué fase se estaba ejecutando y por qué.
2. **TDSP** permitió que cualquier integrante del equipo pudiera entender y reproducir el trabajo de otro, gracias a la organización de carpetas y la semilla `random_state=42`.
3. **Scrum ML** facilitó distribuir el trabajo por roles y sprints, evitando que todas las tareas cayeran sobre una sola persona y haciendo visible el avance del equipo.
4. El modelo baseline (Logistic Regression) superó al Random Forest en F1 Macro, lo que demuestra que **más complejidad no siempre significa mejor rendimiento**.
5. Las variables de **tamaño del grano** (Área, Diámetro equivalente, Área convexa) son las más útiles para distinguir variedades, lo cual tiene sentido desde el punto de vista biológico y agrícola.

---

## ✅ Definition of Done

Una tarea se considera **terminada** cuando cumple todos estos criterios:

- [x] El script corre de inicio a fin sin errores
- [x] El dataset se carga correctamente y se verifica su calidad
- [x] Se reportan valores nulos y duplicados
- [x] Se entrena al menos un modelo baseline y un modelo alternativo
- [x] Se comparan modelos con Accuracy y F1 Macro
- [x] Se incluye matriz de confusión
- [x] Se guarda el modelo final en `outputs/models/`
- [x] Se explican los resultados en lenguaje comprensible
- [x] Se documenta la relación con CRISP-DM, TDSP y Scrum ML

---

## 🔧 Dependencias

```
pandas
numpy
matplotlib
scikit-learn
ucimlrepo
joblib
openpyxl
```

---

## 📚 Referencias

- Koklu, M. & Ozkan, I.A. (2020). *Multiclass Classification of Dry Beans Using Computer Vision and Machine Learning Techniques.* Computers and Electronics in Agriculture.
- UCI Machine Learning Repository: https://archive.ics.uci.edu/dataset/602/dry+bean+dataset
- CRISP-DM Reference Guide: https://www.datascience-pm.com/crisp-dm-2/
- Microsoft TDSP: https://learn.microsoft.com/en-us/azure/architecture/data-science-process/overview
- Scikit-learn Documentation: https://scikit-learn.org/stable/

---

*Proyecto desarrollado como laboratorio práctico de Machine Learning aplicando metodologías profesionales de ciencia de datos.*
