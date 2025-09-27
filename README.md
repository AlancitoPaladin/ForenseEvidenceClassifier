# Singns

Proyecto de reconocimiento de letras usando Bag of Visual Words (BoVW) y un clasificador SVM sobre descriptores locales (SIFT/KAZE) con OpenCV y scikit‑learn.

## ¿Qué hace este proyecto?
- Carga un conjunto de imágenes organizado por carpetas (una carpeta por clase/etiqueta).
- Extrae puntos clave y descriptores con SIFT o KAZE.
- Construye un diccionario visual (KMeans) para representar cada imagen como un histograma de “palabras visuales”.
- Entrena una SVM para clasificar las imágenes según su letra/clase.
- Evalúa el modelo (classification_report y exactitud).
- (Opcional) Compara y registra en CSV coordenadas de keypoints “coincidentes” entre SIFT y KAZE para una imagen por clase.

## Estructura del proyecto
```
Singns/
├─ bvw.py                # Lógica de BoVW: carga, detectores, extracción y transformaciones
├─ main.py               # Script principal: entrenamiento, evaluación y exportación de CSV
├─ dataset/              # (Opcional) Otro dataset
├─ images/               # Dataset principal organizado por clases
│  ├─ a/
│  ├─ b/
│  └─ ...
├─ predict/              # Imágenes para pruebas de predicción manual
└─ coordenadas_matches.csv (generado por main.py)
```

## Requisitos
- Python 3.9+
- OpenCV (cv2) con SIFT disponible. En OpenCV moderno, SIFT ya está integrado; si no, instale `opencv-contrib-python`.
- scikit-learn, numpy, matplotlib

Instalación rápida (ejemplo):
```
pip install opencv-contrib-python scikit-learn numpy matplotlib
```

## Cómo ejecutar
Desde la raíz del proyecto:
```
python main.py
```
El script:
1. Carga imágenes desde `./images` (carpetas por clase).
2. Divide en train/test (25% test).
3. Construye BoVW (por defecto KAZE en main.py, configurable) y extrae características.
4. Entrena una SVM (kernel lineal por defecto).
5. Imprime reporte de clasificación y exactitud.
6. Genera `coordenadas_matches.csv` con coincidencias geométricas SIFT↔KAZE para una imagen por clase.

## Parámetros relevantes
En `main.py`:
- `detector_type = "KAZE"` (puede ser "SIFT" o "KAZE").
- `nfeatures = 150` (límite de keypoints al usar SIFT).
- `BagOfVisualWords(n_clusters=150, ...)` controla el tamaño del diccionario visual.

## Predicción de una imagen nueva
Descomente el bloque de ejemplo al final de `main.py` y ajuste la ruta del archivo:
```python
# new_img = cv2.imread('./predict/mi_imagen.jpg', cv2.IMREAD_GRAYSCALE)
# new_bovw = bovw.transform([new_img])
# pred = clf.predict(new_bovw)
# print(f'Letra predecida: {pred[0]}')
```

## Notas
- El script `main.py` asume que `./images` existe y contiene carpetas con imágenes en escala de grises o RGB (se convierten a gris al leer).
- Si no se detectan keypoints suficientes, BoVW lanzará un error. Verifique la calidad/resolución de imágenes.
- Para la comparación geométrica de SIFT↔KAZE, se usa una distancia euclídea simple en píxeles; ajuste `distance_thresh` en `bvw.py`/`main.py` según el caso.

## ¿Qué hace cada módulo?
- `bvw.py`: 
  - `load_dataset`, `load_images`: cargan imágenes y etiquetas.
  - `get_detector`: crea SIFT/KAZE.
  - `feature_extraction`: extrae descriptores.
  - `BagOfVisualWords`: `fit` agrupa descriptores con KMeans; `transform` genera histogramas normalizados.
  - `get_geom_matches_keypoints` y `plot_overlap_keypoints`: emparejan geométricamente keypoints SIFT↔KAZE y visualizan.
- `main.py`: orquesta el flujo completo de entrenamiento, evaluación y exportación a CSV.

## FAQ
- ¿Qué hago si falta SIFT? Instale `opencv-contrib-python` y asegúrese de que su versión de OpenCV incluya SIFT.
- ¿Puedo usar RBF en SVM? Cambie `SVC(kernel='linear')` por `SVC(kernel='rbf')` y ajuste `C`/`gamma`.
# ForenseEvidenceClassifier
