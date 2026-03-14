# Python
import joblib

from bvw import BagOfVisualWords, get_detector

bovw_state = joblib.load("models/bovw_20251002-220954_bovw350_KAZE_GradientBoostingClassifier.joblib")
clf = joblib.load("models/clf_20251002-220954_bovw350_KAZE_GradientBoostingClassifier.joblib")

bovw = BagOfVisualWords(
    n_clusters=bovw_state["n_clusters"],
    detector_type=bovw_state["detector_type"],
    nfeatures=bovw_state["nfeatures"],
)
bovw.bag = bovw_state["bag"]
bovw.detector = get_detector(bovw.detector_type, bovw.nfeatures)
print(bovw_state.keys())  # n_clusters, detector_type, nfeatures, bag
print(clf)  # imprime el estimador con sus hiperparámetros
print(bovw_state["bag"].cluster_centers_.shape)  # centros KMeans

"""
img = cv2.imread("ruta/imagen.tiff", cv2.IMREAD_GRAYSCALE)
X_bovw = bovw.transform([img])
pred = clf.predict(X_bovw)
print(pred[0])
"""
# X_new_bovw = bovw.transform(X_new); pred = clf.predict(X_new_bovw)
