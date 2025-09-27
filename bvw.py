import os

import cv2
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans


def load_dataset(dataset_path):
    X = []
    y = []
    # Revisa primero si hay imágenes directamente en la carpeta
    for fname in os.listdir(dataset_path):
        img_path = os.path.join(dataset_path, fname)
        if os.path.isfile(img_path) and fname.lower().endswith((".jpg", ".png", ".tif", ".tiff")):
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                X.append(img)
                y.append("default")  # etiqueta genérica

    # Revisa si hay subcarpetas (varias clases)
    for label in sorted(os.listdir(dataset_path)):
        folder_path = os.path.join(dataset_path, label)
        if not os.path.isdir(folder_path):
            continue
        for fname in os.listdir(folder_path):
            img_path = os.path.join(folder_path, fname)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                X.append(img)
                y.append(label)

    y = np.array(y)
    return X, y


def load_images(dataset_path):
    """
    Retorna una imagen de cada clase encontrada (máximo 2, útil para pruebas rápidas).
    """
    X = []
    y = []
    labels_seen = set()
    for label in sorted(os.listdir(dataset_path)):
        folder_path = os.path.join(dataset_path, label)
        if not os.path.isdir(folder_path):
            continue
        img_files = sorted(os.listdir(folder_path))
        for fname in img_files:
            img_path = os.path.join(folder_path, fname)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is not None and label not in labels_seen:
                X.append(img)
                y.append(label)
                labels_seen.add(label)
            if len(labels_seen) >= 21:
                return X, y
    return X, y


def get_detector(detector_type="SIFT", nfeatures=0, kaze_threshold=0.001):
    detector_type = detector_type.upper()
    if detector_type == "SIFT":
        return cv2.SIFT_create(nfeatures=nfeatures) if nfeatures > 0 else cv2.SIFT_create()
    elif detector_type == "SURF":
        if hasattr(cv2, 'xfeatures2d'):
            return cv2.xfeatures2d.SURF_create(hessianThreshold=400)
        else:
            raise AttributeError("SURF no está disponible")
    elif detector_type == "KAZE":
        return cv2.KAZE_create(threshold=kaze_threshold)
    elif detector_type == "SIFT+KAZE":
        class SIFTKAZEWrapper:
            @staticmethod
            def detectAndCompute(img, mask=None):
                return sift_and_kaze(img)

        return SIFTKAZEWrapper()
    else:
        raise ValueError(f"Detector '{detector_type}' no soportado.")


def feature_extraction(X, detector):
    """
    Extrae todos los keypoints y descriptores de un conjunto de imágenes.
    """
    kp_vector = []
    des_vector = []
    for img in X:
        kp, des = detector.detectAndCompute(img, None)
        if (kp is not None) and (des is not None):
            kp_vector.extend(kp)
            des_vector.append(des)
    kp_vector = np.array(kp_vector)
    des_vector = np.vstack(des_vector) if des_vector else np.array([])
    return kp_vector, des_vector


class BagOfVisualWords(BaseEstimator, TransformerMixin):
    def __init__(self, n_clusters, detector_type="SIFT", nfeatures=0):
        self.n_clusters = n_clusters
        self.detector_type = detector_type
        self.nfeatures = nfeatures
        self.detector = get_detector(detector_type, nfeatures)

    def fit(self, X, y=None):
        self.labels_list = np.unique(y)
        self.labels_dict = {label: dict.fromkeys(['kp', 'des']) for label in self.labels_list}
        self.bneck_value = float("inf")
        for label in self.labels_list:
            idx = np.where(y == label)
            imgs = [X[i] for i in idx[0]]
            kp_vec, des_vec = feature_extraction(imgs, self.detector)
            if self.bneck_value > len(kp_vec) > 0:
                self.bneck_value = len(kp_vec)
            self.labels_dict[label]['kp'] = kp_vec
            self.labels_dict[label]['des'] = des_vec

        if self.bneck_value == float("inf"):
            raise ValueError("No se encontraron keypoints en el dataset.")

        self.n_descriptors = max(1, int(0.8 * self.bneck_value))
        selected_descriptors = []
        for label in self.labels_list:
            kp = self.labels_dict[label]['kp']
            des = self.labels_dict[label]['des']
            if len(kp) == 0 or len(des) == 0:
                continue
            idx_sorted = sorted(range(len(kp)), key=lambda i: kp[i].response, reverse=True)
            n_sel = min(self.n_descriptors, len(des))
            idx_top = idx_sorted[:n_sel]
            selected_descriptors.append(des[idx_top])

        des_vector = np.vstack(selected_descriptors).astype(np.float64)
        self.bag = KMeans(n_clusters=self.n_clusters, random_state=0).fit(des_vector)
        return self

    def transform(self, X):
        N = len(X)
        K = self.bag.n_clusters
        feature_vector = np.zeros((N, K), dtype=np.float32)
        for i in range(N):
            _, curr_des = self.detector.detectAndCompute(X[i], None)
            if curr_des is None or len(curr_des) == 0:
                continue
            word_vector = self.bag.predict(curr_des.astype(float))
            for w in np.unique(word_vector):
                feature_vector[i, w] = np.sum(word_vector == w)
            cv2.normalize(feature_vector[i], feature_vector[i], norm_type=cv2.NORM_L2)
        return feature_vector


def sift_and_kaze(img, nfeatures_sift=500, kaze_threshold=0.001):
    """
    Detector híbrido que combina SIFT + KAZE.
    Devuelve keypoints y descriptores concatenados.
    """
    # Preprocesamiento ligero
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img_proc = clahe.apply(img)

    # Detectores
    sift = cv2.SIFT_create(nfeatures=nfeatures_sift)
    kaze = cv2.KAZE_create(threshold=kaze_threshold)

    kp_sift, des_sift = sift.detectAndCompute(img_proc, None)
    kp_kaze, des_kaze = kaze.detectAndCompute(img_proc, None)

    # Normalizar salidas vacías
    if des_sift is None: des_sift, kp_sift = np.empty((0, 128), np.float32), []
    if des_kaze is None: des_kaze, kp_kaze = np.empty((0, 64), np.float32), []

    if des_sift.shape[0] > max_des:
        des_sift = des_sift[np.random.choice(des_sift.shape[0], max_des, replace=False)]
    if des_kaze.shape[0] > max_des:
        des_kaze = des_kaze[np.random.choice(des_kaze.shape[0], max_des, replace=False)]

    # Padding de KAZE a 128 dims
    if des_kaze.shape[0] > 0:
        des_kaze_padded = np.zeros((des_kaze.shape[0], 128), dtype=np.float32)
        des_kaze_padded[:, :64] = des_kaze
    else:
        des_kaze_padded = np.empty((0, 128), dtype=np.float32)

    # Concatenar resultados
    kp = kp_sift + kp_kaze
    des = np.vstack([des_sift, des_kaze_padded]) if len(kp) > 0 else None

    return kp, des
