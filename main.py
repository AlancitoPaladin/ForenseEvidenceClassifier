from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier  # Nuevo: RF y GBT

import bvw as bvw
from bvw import BagOfVisualWords

if __name__ == "__main__":
    # 1. Carga dataset
    #X, y = bvw.load_dataset("/Volumes/AlanDisk/Shoes/Results")
    X, y = bvw.load_dataset("./images")
    print(f'Dataset cargado: images={len(X)}')

    # 2. Split en train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=0.25, random_state=42
    )
    print(f'Train: {len(X_train)}, Test: {len(X_test)}')

    # 3. Inicialización y entrenamiento de la BoVW con SIFT+KAZE
    # Hacer escaneo desde 100 hasta 500 clusters e ir guardando todos y cada uno de los modelos que se tengan.
    bovw = BagOfVisualWords(n_clusters=250, detector_type="SIFT", nfeatures=150)
    print(f'Creando Bag of Visual Words usando {bovw.detector_type}...')
    bovw.fit(X_train, y_train)
    print('Bag of Visual Words creada.')

    # 4. Transformación de imágenes en vectores BoVW
    X_train_bovw = bovw.transform(X_train)
    X_test_bovw = bovw.transform(X_test)
    print('Extracción de características finalizada.')

    # 5. Clasificador
    # Opción A: Random Forest (comentado)
    # rf_clf = RandomForestClassifier(
    #     n_estimators=300, max_depth=None, random_state=42, n_jobs=-1
    # )

    # Opción B: Árboles impulsados por gradiente (Gradient Boosting) — ACTIVADO
    clf = GradientBoostingClassifier(random_state=42)

    # Opción C: SVM (comentado)
    # clf = SVC(kernel='linear')  # 'linear', 'rbf', 'poly', 'sigmoid'

    print('Entrenando clasificador...')
    clf.fit(X_train_bovw, y_train)
    print('Clasificador entrenado.')

    # 6. Evaluación
    y_pred = clf.predict(X_test_bovw)
    print('Reporte de clasificación:')
    print(classification_report(y_test, y_pred))
    print('Exactitud:', accuracy_score(y_test, y_pred))

    # 7. Ejemplo: predicción de una imagen
    # new_img = cv2.imread('./predict/predictE.jpg', cv2.IMREAD_GRAYSCALE)
    # new_bovw = bovw.transform([new_img])
    # pred = clf.predict(new_bovw)
    # print(f'Letra predecida: {pred[0]}')

    # 8.- Formación de nuevas bolsas de palabras (BOVW) a partir de la concatenación de SIFT y KAZE.
    # bovw_concat = BagOfVisualWords(n_clusters=350, detector_type="SIFT+KAZE", nfeatures=150)
    # bovw_concat.fit(compare_imgs, compare_labels)
    # X_concat_bovw = bovw_concat.transform(compare_imgs)
    # print(f'BOVW concatenada creada con {len(X_concat_bovw)} muestras.')

"""
 
- **Macro average: el promedio de la métrica (precisión, recall, etc.) 
  sin importar cuántas muestras hay de cada etiqueta.
  Es decir, se calcula la métrica para cada clase (etiqueta) y luego se promedian todos esos valores.

- **Weighted average: es similar, pero cada métrica de etiqueta se pondera según cuántas muestras 
  hay de esa etiqueta ("support" o soporte).

- Macro avg: promedia las métricas de cada etiqueta/clase.
- Weighted avg: promedia las métricas de cada etiqueta/clase pero ponderando según el número de muestras por clase
"""
