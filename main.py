import os
from datetime import datetime

import joblib
import numpy as np
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

import bvw as bvw
from bvw import BagOfVisualWords

if __name__ == "__main__":
    # 1. Carga dataset
    X, y = bvw.load_dataset("/Volumes/AlanDisk/Shoes/ResultsGel")
    # X, y = bvw.load_dataset("./images")
    print(f'Dataset cargado: images={len(X)}')

    # Sanity checks de dataset
    if len(X) == 0:
        raise ValueError("El dataset está vacío o no se pudieron leer imágenes.")
    classes, counts = np.unique(y, return_counts=True)
    if len(classes) < 2:
        raise ValueError("Se requieren al menos 2 clases para clasificar.")
    low_count = classes[counts < 2]
    if len(low_count) > 0:
        print(f"Aviso: clases con menos de 2 imágenes: {', '.join(low_count)}")

    # 2. Split en train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=0.25, random_state=42
    )
    print(f'Train: {len(X_train)}, Test: {len(X_test)}')

    # 3. Inicialización y entrenamiento de la BoVW con SIFT+KAZE
    # Hacer escaneo desde 100 hasta 500 clusters e ir guardando todos y cada uno de los modelos que se tengan.
    bovw = BagOfVisualWords(n_clusters=100, detector_type="SIFT", nfeatures=100)
    print(f'Creando Bag of Visual Words usando {bovw.detector_type}...')
    bovw.fit(X_train, y_train)
    print('Bag of Visual Words creada.')

    # 4. Transformación de imágenes en vectores BoVW
    X_train_bovw = bovw.transform(X_train)
    X_test_bovw = bovw.transform(X_test)
    print('Extracción de características finalizada.')

    # 5. Clasificador
    # Clasificador usado en el artículo: SVM lineal
    clf = SVC(kernel='linear')
    # Opciones alternativas (no usadas en los resultados reportados):
    # clf = RandomForestClassifier(n_estimators=300, max_depth=None, random_state=42, n_jobs=-1)
    # clf = GradientBoostingClassifier(random_state=42)

    print('Entrenando clasificador...')
    clf.fit(X_train_bovw, y_train)
    print('Clasificador entrenado.')

    # Guardado de modelos
    os.makedirs("models", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    clf_name = clf.__class__.__name__
    tag = f"{timestamp}_bovw{bovw.n_clusters}_{bovw.detector_type}_{clf_name}"
    bovw_path = f"models/bovw_{tag}.joblib"
    clf_path = f"models/clf_{tag}.joblib"

    bovw_state = {
        "n_clusters": bovw.n_clusters,
        "detector_type": bovw.detector_type,
        "nfeatures": bovw.nfeatures,
        "bag": bovw.bag,  # KMeans serializable
    }
    joblib.dump(bovw_state, bovw_path)
    joblib.dump(clf, clf_path)
    print(f"Modelos guardados:\n- {bovw_path}\n- {clf_path}")

    # 6. Evaluación
    y_pred = clf.predict(X_test_bovw)
    print('\n' + '=' * 60)
    print('RESULTADOS DE EVALUACIÓN')
    print('=' * 60)

    # Reporte completo (ya incluye precisión y recall por clase)
    print('\nReporte de clasificación:')
    print(classification_report(y_test, y_pred))

    # Métricas globales
    print('\n--- MÉTRICAS GLOBALES ---')
    print(f'Exactitud (Accuracy): {accuracy_score(y_test, y_pred):.4f}')
    print(f'Precisión promedio (weighted): {precision_score(y_test, y_pred, average="weighted"):.4f}')
    print(f'Recall promedio (weighted): {recall_score(y_test, y_pred, average="weighted"):.4f}')

    # Obtener las clases únicas
    classes = sorted(set(y_test))

    # Calcular precisión y recall por clase
    precision_per_class = precision_score(y_test, y_pred, average=None, labels=classes)
    recall_per_class = recall_score(y_test, y_pred, average=None, labels=classes)

    print('\n--- MÉTRICAS POR CLASE ---')
    for i, class_name in enumerate(classes):
        print(f'{class_name}:')
        print(f'  Precisión: {precision_per_class[i]:.4f}')
        print(f'  Recall: {recall_per_class[i]:.4f}')

    print('=' * 60 + '\n')

    # Carga (reconstrucción de BoVW)
    # state = joblib.load(bovw_path)
    # bovw_loaded = BagOfVisualWords(
    #     n_clusters=state["n_clusters"],
    #     detector_type=state["detector_type"],
    #     nfeatures=state["nfeatures"],
    # )
    # bovw_loaded.bag = state["bag"]
    # bovw_loaded.detector = get_detector(bovw_loaded.detector_type, bovw_loaded.nfeatures)
