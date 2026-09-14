# Kaggle – Data on the top 💻📈

Reto de predicción del precio de portátiles (`Price_in_euros`) a partir de sus
características técnicas (marca, CPU, GPU, RAM, almacenamiento, pantalla...).

Métrica de evaluación: **RMSE** (Root Mean Squared Error) — cuanto menor, mejor.

## Estructura del repo

```
.
├── TC-Kaggle_submission.ipynb   # Notebook con la solución completa
├── data/
│   ├── train.csv                # Datos de entrenamiento (con Price_in_euros)
│   ├── test.csv                 # Datos de test (sin Price_in_euros, a predecir)
│   └── sample_submission.csv    # Formato exacto que espera Kaggle
├── requirements.txt
└── README.md
```

## Cómo ejecutarlo

```bash
python -m venv venv
source venv/bin/activate        # En Windows: venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook TC-Kaggle_submission.ipynb
```

Ejecuta las celdas de arriba a abajo. Al final se generará un archivo
`submission_<timestamp>.csv` listo para subir a Kaggle.

## Resumen de la solución

1. **Exploración (2.1):** revisión de tipos, nulos y cardinalidad de cada columna.
2. **Feature engineering (2.2):** las columnas vienen "sucias" (texto libre) y se
   transforman en variables útiles:
   - `Ram`: `"8GB"` → `8` (int)
   - `Weight`: `"1.86kg"` → `1.86` (float)
   - `ScreenResolution`: se extraen `Touchscreen`, `IPS`, ancho/alto de pantalla
     y la densidad de píxeles (`PPI`)
   - `Cpu`: se separa en marca/gama (`CpuBrand`) y velocidad en GHz (`CpuSpeedGHz`)
   - `Memory`: puede combinar varias unidades (p. ej. `"256GB SSD + 1TB HDD"`),
     se separa en `SSD_GB`, `HDD_GB`, `Flash_GB`, `Hybrid_GB`
   - `Gpu`: se queda solo con la marca (`GpuBrand`)
   - Se descarta `Product` (texto libre de muy alta cardinalidad)
3. **Train/test split (2.3):** 80/20 para validar localmente.
4. **Procesado (3):** `OneHotEncoder` para categóricas y `StandardScaler` para
   numéricas, ajustados (`.fit()`) **solo con `X_train`** para evitar *data
   leakage*, y aplicados con `.transform()` tanto a train como a test.
5. **Modelado (4):**
   - Se entrenan y comparan `RandomForestRegressor` y `SVR`.
   - Se optimiza el Random Forest con `GridSearchCV` (nº de árboles, profundidad,
     `min_samples_split`).
6. **Reentrenamiento final (5):** el modelo ganador se reentrena con el 100% de
   `train.csv` (encoder y scaler también se reajustan sobre todos los datos).
7. **Predicción sobre `test.csv` (6-7):** se aplica exactamente el mismo
   `feature_engineer()` y luego solo `.transform()` (nunca `.fit_transform()`)
   con el encoder/scaler ya ajustados. No se elimina ninguna fila.
8. **Submission (8):** se genera el CSV con las columnas `laptop_ID` y
   `Price_in_euros`, y se valida con la función `checker()` (mismo shape,
   mismas columnas y mismo orden de IDs que `sample_submission.csv`) antes de
   guardarlo.

## Resultado local

Con el split de validación (80/20) y el Random Forest optimizado por
`GridSearchCV`, el RMSE en validación cruzada ronda los **~290-380 €**
(dependerá de la semilla/aleatoriedad de cada ejecución). Hay margen de mejora
probando otros modelos (Gradient Boosting, XGBoost/LightGBM), más
feature engineering (p. ej. extraer el modelo concreto de CPU/GPU) o técnicas
de *stacking/ensembling*.
