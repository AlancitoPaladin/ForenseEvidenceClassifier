import cv2
import matplotlib.pyplot as plt
import os
import numpy as np

# --- Configuración de Rutas y Número de Keypoints ---
# Rutas específicas a los archivos .tif (¡Asegúrate de que estas rutas son correctas en tu sistema!)
pathNike = "/Volumes/AlanDisk/Shoes/Results/Nike(results)/cleaned_roi_005LBFT11-21-22AGGV12-02-22MM.tif"
pathAdidas = "/Volumes/AlanDisk/Shoes/Results/Adidas(results)/cleaned_roi_009RBPT11-22-22AGGV12-19-22AT.tif"
MAX_KEYPOINTS = 50


# --- Funciones Auxiliares ---
# ... (load_and_process_image es la misma)

def load_and_process_image(image_path, name):
    """Carga la imagen, extrae keypoints (SIFT) y selecciona los N mejores."""
    if not os.path.isfile(image_path):
        print(f"Error: No se encontró la imagen en la ruta especificada para {name}: {image_path}")
        return None, None
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error al cargar la imagen. El archivo puede estar corrupto o el formato no es compatible: {image_path}")
        return None, None
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create()
    kp, _ = sift.detectAndCompute(img_gray, None)
    kp_sorted = sorted(kp, key=lambda x: x.response, reverse=True)
    selected_kp = kp_sorted[:MAX_KEYPOINTS]
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    print(f"Keypoints detectados para {name}: {len(kp)}. Seleccionados: {len(selected_kp)}.")
    return img_rgb, selected_kp


def plot_keypoints(img, keypoints, title, ax):
    """Grafica la imagen y sus keypoints en un subplot específico."""
    ax.imshow(img)
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.axis('off')

    points = cv2.KeyPoint.convert(keypoints)

    if points.size > 0:
        x_coords = points[:, 0]
        y_coords = points[:, 1]

        ax.scatter(
            x_coords,
            y_coords,
            s=50,
            c='red',
            edgecolor='yellow',
            linewidth=1.5,
            marker='o',
            label=f'{len(keypoints)} Keypoints SIFT' # Corregido para incluir el número de KPs
        )
        ax.legend(loc='lower right', fontsize=10, framealpha=0.7)


# ----------------------------------------------------
## 🚀 Proceso Principal: Cargar, Procesar y Graficar (Vertical Ajustado)
# ----------------------------------------------------

# 1. Cargar y procesar ambas imágenes
img_nike, kp_nike = load_and_process_image(pathNike, "Nike")
img_adidas, kp_adidas = load_and_process_image(pathAdidas, "Adidas")

# 2. Verificar carga exitosa y graficar si es posible
if img_nike is not None and kp_nike is not None and img_adidas is not None and kp_adidas is not None:

    # 3. Crear la figura profesional para el cártel (2 FILAS, 1 COLUMNA)
    # Cambiado a (2, 1) y tamaño vertical
    fig, axes = plt.subplots(2, 1, figsize=(8, 10))

    # Título general de la figura
    fig.suptitle(
        'Análisis de Keypoints (SIFT) - Comparación de suelas',
        fontsize=16, # Fuente reducida para acercar el título
        fontweight='bold',
        y=1.0 # Pega el título al borde superior
    )

    # Ajuste de espacio entre subplots y bordes
    plt.subplots_adjust(
        left=0.05,
        right=0.95,
        top=0.98, # Margen superior mínimo
        bottom=0.05,
        wspace=0.0, # Sin espacio horizontal
        hspace=0.1 # Espacio vertical mínimo entre las imágenes
    )

    # 4. Graficar los resultados (Usamos axes[0] y axes[1])
    plot_keypoints(img_nike, kp_nike, f'Nike: Keypoints SIFT ({MAX_KEYPOINTS})', axes[0])
    plot_keypoints(img_adidas, kp_adidas, f'Adidas: Keypoints SIFT ({MAX_KEYPOINTS})', axes[1])

    # 5. Guardar la gráfica para el cártel (alta resolución)
    output_filename = 'analisis_keypoints_cartel_vertical_final.png'
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')

    print(f"\n✨ Gráfica generada y guardada como: {output_filename}")

    # Mostrar la gráfica
    plt.show()

else:
    print("\nEl proceso de graficado no se pudo completar. Por favor, revisa las rutas de los archivos y los mensajes de error.")