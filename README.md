# Singns

Clasificador de imágenes de huellas de calzado para evidencia forense usando Bag of Visual Words (BoVW) con descriptores locales (SIFT/KAZE) y un clasificador supervisado de scikit-learn. El dataset es **shoes**, con dos clases: **Nike** y **Adidas**.

## Resumen
- Lee imágenes organizadas por carpetas (una carpeta = una clase).
- Extrae keypoints y descriptores con SIFT, KAZE o un híbrido.
- Construye un diccionario visual con KMeans/MiniBatchKMeans.
- Representa cada imagen como un histograma BoVW.
- Entrena y evalúa un clasificador (por defecto RandomForest).
- Guarda el modelo BoVW y el clasificador en `models/`.

## Requisitos
- Python 3.9+
- OpenCV con SIFT disponible (`opencv-contrib-python` si tu build no trae SIFT)
- scikit-learn, numpy, joblib

Instalación rápida:
```bash
pip install opencv-contrib-python scikit-learn numpy joblib
```

## Uso rápido
1. Asegura tu dataset **shoes** con esta estructura:
```text
shoes/
  Nike/
  Adidas/
```

2. Edita la ruta del dataset en `main.py`:
```python
X, y = bvw.load_dataset("/ruta/a/shoes")
# X, y = bvw.load_dataset("./images")
```

3. Ejecuta:
```bash
python main.py
```

Al finalizar verás métricas de evaluación y se guardarán los modelos en `models/`.

## Configuración clave
En `main.py`:
- `BagOfVisualWords(n_clusters=100, detector_type="SIFT", nfeatures=100)`
  - `n_clusters`: tamaño del diccionario visual.
  - `detector_type`: `SIFT`, `KAZE` o `SIFT+KAZE`.
  - `nfeatures`: límite de keypoints para SIFT.

- Clasificador (por defecto):
```python
clf = RandomForestClassifier(
    n_estimators=300, max_depth=None, random_state=42, n_jobs=-1
)
```
Puedes cambiar a SVM o GradientBoosting (ya hay opciones comentadas en `main.py`).

## Predicción de una imagen nueva
Ejemplo usando el modelo ya entrenado en memoria:
```python
new_img = cv2.imread('./predict/mi_imagen.jpg', cv2.IMREAD_GRAYSCALE)
new_bovw = bovw.transform([new_img])
pred = clf.predict(new_bovw)
print(f'Clase predicha: {pred[0]}')
```

Para usar modelos guardados, hay un bloque de ejemplo al final de `main.py` para reconstruir `BagOfVisualWords` desde `joblib`.

## Estructura del proyecto
```text
Singns/
├─ bvw.py                 # Lógica BoVW: carga, detectores, extracción y transformaciones
├─ main.py                # Entrenamiento, evaluación y guardado de modelos
├─ shoes/                 # Dataset principal (Nike/Adidas)
├─ predict/               # Imágenes para pruebas manuales
├─ models/                # Modelos guardados (se crea automáticamente)
└─ util.py, reuse_model.py, show.py, rezise.py, bvw.py
```

## Detalles de implementación
- `bvw.py`:
  - `load_dataset` lee imágenes en gris por clase.
  - `get_detector` expone SIFT/KAZE/SIFT+KAZE.
  - `BagOfVisualWords.fit` construye el diccionario visual con KMeans/MiniBatchKMeans.
  - `BagOfVisualWords.transform` genera histogramas normalizados por imagen.

- `main.py`:
  - Hace split train/test con estratificación.
  - Entrena BoVW y el clasificador.
  - Evalúa con `classification_report`, accuracy, precision y recall.
  - Guarda modelos en `models/` con timestamp.

## Problemas comunes
- “No se encontraron keypoints”: revisa la calidad o el tamaño de las imágenes.
- Pocas imágenes por clase: el script avisa si alguna clase tiene menos de 2 imágenes.
- SIFT no disponible: instala `opencv-contrib-python`.

## Licencia
Por definir.
