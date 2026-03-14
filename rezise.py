import os
import shutil
import tempfile
from pathlib import Path
from typing import Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


class ForensicRulerRemover:

    def __init__(self,
                 conservative_crop: bool = True,
                 debug_mode: bool = False,
                 top_crop_percentage: float = 0.15,
                 left_crop_percentage: float = 0.12,  # CAMBIADO: ahora es izquierda
                 use_only_percentages: bool = False):
        self.conservative_crop = conservative_crop
        self.debug_mode = debug_mode
        self.top_crop_percentage = top_crop_percentage
        self.left_crop_percentage = left_crop_percentage  # CAMBIADO
        self.use_only_percentages = use_only_percentages

    @staticmethod
    def load_image(image_path: str) -> np.ndarray:
        try:
            with Image.open(image_path) as img:
                if img.mode != 'L':
                    img = img.convert('L')
                return np.array(img)
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            return None

    def detect_rulers_comprehensive(self, image: np.ndarray) -> Tuple[int, int]:
        height, width = image.shape

        if self.debug_mode:
            print(f"Image dimensions: {width} x {height}")

        if self.use_only_percentages:
            top_crop = int(height * self.top_crop_percentage)
            left_crop = int(width * self.left_crop_percentage)  # CAMBIADO

            return top_crop, left_crop

        top_uniform, left_uniform = self._detect_uniform_ruler_background(image)

        top_histogram, left_histogram = self._detect_rulers_by_histogram(image)

        top_text_density, left_text_density = self._detect_text_density_boundaries(image)

        top_edge_pattern, left_edge_pattern = self._detect_ruler_edge_patterns(image)

        top_conservative = int(height * self.top_crop_percentage)
        left_conservative = int(width * self.left_crop_percentage)  # CAMBIADO

        top_candidates = [top_uniform, top_histogram, top_text_density, top_edge_pattern]
        left_candidates = [left_uniform, left_histogram, left_text_density, left_edge_pattern]  # CAMBIADO

        valid_top = [t for t in top_candidates if 50 <= t <= height // 2]
        valid_left = [l for l in left_candidates if 30 <= l <= width // 2]  # CAMBIADO

        if self.conservative_crop or not valid_top:
            final_top = max(top_conservative, max(valid_top) if valid_top else top_conservative)
        else:
            final_top = int(np.median(valid_top)) if valid_top else top_conservative

        if self.conservative_crop or not valid_left:
            final_left = max(left_conservative, max(valid_left) if valid_left else left_conservative)
        else:
            final_left = int(np.percentile(valid_left, 75)) if valid_left else left_conservative

        return final_top, final_left

    @staticmethod
    def _detect_uniform_ruler_background(image: np.ndarray) -> Tuple[int, int]:
        height, width = image.shape

        # Detección TOP (sin cambios)
        top_boundary = 80
        search_height = min(200, height // 3)

        strip_height = 10
        for y in range(50, search_height - strip_height, strip_height):
            strip = image[y:y + strip_height, :]
            strip_std = np.std(strip)

            next_strip = image[y + strip_height:min(y + 2 * strip_height, height), :]
            if next_strip.size > 0:
                next_std = np.std(next_strip)

                if strip_std < 15 and next_std > strip_std * 2 and next_std > 30:
                    top_boundary = y + strip_height + 20
                    break

        # Detección IZQUIERDA (CAMBIADO: busca desde la izquierda)
        left_boundary = int(width * 0.12)  # Default 12% desde la izquierda
        search_width = min(250, width // 3)

        strip_width = 8
        for x in range(50, search_width, strip_width):  # CAMBIADO: avanza hacia la derecha
            strip = image[:, x:x + strip_width]
            strip_std = np.std(strip)
            strip_mean = np.mean(strip)

            next_strip = image[:, x + strip_width:min(x + 2 * strip_width, width)]
            if next_strip.size > 0:
                next_mean = np.mean(next_strip)

                has_text_elements = np.any(strip < next_mean - 30)
                has_uniform_bg = strip_std < 20
                intensity_difference = abs(strip_mean - next_mean) > 15

                if has_uniform_bg and has_text_elements and intensity_difference:
                    left_boundary = x + strip_width + 15
                    break

        return top_boundary, left_boundary

    @staticmethod
    def _detect_rulers_by_histogram(image: np.ndarray) -> Tuple[int, int]:
        height, width = image.shape

        # Detección TOP (sin cambios)
        top_boundary = 100
        search_height = min(250, height // 2)

        ruler_area = image[:search_height, :]
        ruler_hist, _ = np.histogram(ruler_area.flatten(), bins=50, range=(0, 255))

        evidence_area = image[search_height:, :]
        if evidence_area.size > 0:
            evidence_hist, _ = np.histogram(evidence_area.flatten(), bins=50, range=(0, 255))

            hist_diff = np.sum(np.abs(ruler_hist - evidence_hist))

            if hist_diff > np.mean(ruler_hist) * 10:
                for y in range(80, search_height - 20, 10):
                    current_area = image[y:y + 20, :]
                    below_area = image[y + 20:min(y + 60, height), :]

                    if below_area.size > 0:
                        current_hist, _ = np.histogram(current_area.flatten(), bins=20)
                        below_hist, _ = np.histogram(below_area.flatten(), bins=20)

                        row_diff = np.sum(np.abs(current_hist - below_hist))
                        if row_diff > np.mean(current_hist) * 3:
                            top_boundary = y + 30
                            break

        # Detección IZQUIERDA (CAMBIADO)
        left_boundary = int(width * 0.12)
        search_width = min(200, width // 4)

        ruler_area_left = image[:, :search_width]  # CAMBIADO: área izquierda
        ruler_hist_left, _ = np.histogram(ruler_area_left.flatten(), bins=50, range=(0, 255))

        evidence_start = search_width
        evidence_end = width - (width // 4)
        evidence_area_left = image[:, evidence_start:evidence_end]

        if evidence_area_left.size > 0:
            evidence_hist_left, _ = np.histogram(evidence_area_left.flatten(), bins=50, range=(0, 255))

            hist_diff_left = np.sum(np.abs(ruler_hist_left - evidence_hist_left))

            if hist_diff_left > np.mean(ruler_hist_left) * 12:
                for x in range(80, search_width, 15):  # CAMBIADO: avanza hacia la derecha
                    current_area = image[:, x:x + 15]
                    right_area = image[:, x + 15:min(x + 45, width)]

                    if right_area.size > 0 and current_area.size > 0:
                        current_hist, _ = np.histogram(current_area.flatten(), bins=20)
                        right_hist, _ = np.histogram(right_area.flatten(), bins=20)

                        col_diff = np.sum(np.abs(current_hist - right_hist))
                        if col_diff > np.mean(current_hist) * 4:
                            left_boundary = x + 25
                            break

        return top_boundary, left_boundary

    @staticmethod
    def _detect_text_density_boundaries(image: np.ndarray) -> Tuple[int, int]:
        height, width = image.shape

        edges = cv2.Canny(image, 50, 150, apertureSize=3, L2gradient=True)

        kernel_horizontal = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 3))
        kernel_vertical = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 9))

        # Detección TOP (sin cambios)
        top_boundary = 120
        search_height = min(300, height // 2)

        text_enhanced = cv2.morphologyEx(edges[:search_height, :], cv2.MORPH_CLOSE, kernel_horizontal)

        row_text_density = np.sum(text_enhanced, axis=1)

        if len(row_text_density) > 10:
            smoothed = cv2.GaussianBlur(row_text_density.reshape(-1, 1), (11, 1), 0).flatten()

            mean_density = np.mean(smoothed[:len(smoothed) // 2])

            for i in range(60, len(smoothed) - 20):
                window = smoothed[i:i + 20]
                if len(window) > 0 and np.mean(window) < mean_density * 0.3:
                    top_boundary = i + 20
                    break

        # Detección IZQUIERDA (CAMBIADO)
        left_boundary = int(width * 0.12)
        search_width = min(300, width // 3)

        text_enhanced_left = cv2.morphologyEx(edges[:, :search_width], cv2.MORPH_CLOSE, kernel_vertical)

        col_text_density = np.sum(text_enhanced_left, axis=0)

        if len(col_text_density) > 10:
            smoothed_left = cv2.GaussianBlur(col_text_density.reshape(1, -1), (1, 11), 0).flatten()

            mean_density_left = np.mean(smoothed_left[:len(smoothed_left) // 2])

            if mean_density_left > 50:
                for i in range(60, len(smoothed_left) - 30):  # CAMBIADO: avanza hacia la derecha
                    window = smoothed_left[i:i + 25]
                    if len(window) > 0 and np.mean(window) < mean_density_left * 0.5:
                        left_boundary = i + 25
                        break

        return top_boundary, left_boundary

    @staticmethod
    def _detect_ruler_edge_patterns(image: np.ndarray) -> Tuple[int, int]:
        height, width = image.shape

        edges = cv2.Canny(image, 30, 100)

        # Detección TOP (sin cambios)
        top_boundary = 100
        search_height = min(200, height // 3)

        for y in range(60, search_height - 20, 10):
            strip = edges[y:y + 20, :]

            vertical_projection = np.sum(strip, axis=0)

            if len(vertical_projection) > 100:
                autocorr = np.correlate(vertical_projection, vertical_projection, mode='full')
                autocorr = autocorr[len(autocorr) // 2:]

                if len(autocorr) > 50:
                    peaks = []
                    for i in range(10, min(50, len(autocorr) - 10)):
                        if (autocorr[i] > autocorr[i - 5:i].max() and
                                autocorr[i] > autocorr[i + 1:i + 6].max()):
                            peaks.append(i)

                    if len(peaks) >= 3:
                        below_strip = edges[y + 20:min(y + 40, height), :]
                        below_projection = np.sum(below_strip, axis=0)

                        if len(below_projection) > 0:
                            below_autocorr = np.correlate(below_projection, below_projection, mode='full')
                            below_autocorr = below_autocorr[len(below_autocorr) // 2:]

                            below_peaks = []
                            for i in range(10, min(50, len(below_autocorr) - 10)):
                                if (below_autocorr[i] > below_autocorr[i - 5:i].max() and
                                        below_autocorr[i] > below_autocorr[i + 1:i + 6].max()):
                                    below_peaks.append(i)

                            if len(below_peaks) < len(peaks):
                                top_boundary = y + 30
                                break

        # Detección IZQUIERDA (CAMBIADO)
        left_boundary = int(width * 0.12)
        search_width = min(250, width // 4)

        for x in range(70, search_width, 15):  # CAMBIADO: avanza hacia la derecha
            strip = edges[:, x:x + 15]

            if strip.size > 0:
                horizontal_projection = np.sum(strip, axis=1)
                if len(horizontal_projection) > 100:
                    autocorr = np.correlate(horizontal_projection, horizontal_projection, mode='full')
                    autocorr = autocorr[len(autocorr) // 2:]

                    if len(autocorr) > 50:
                        peaks = []
                        for i in range(15, min(40, len(autocorr) - 10)):
                            if (autocorr[i] > autocorr[i - 7:i].max() and
                                    autocorr[i] > autocorr[i + 1:i + 8].max()):
                                peaks.append(i)

                        if len(peaks) >= 3 and max(autocorr[peaks]) > np.mean(autocorr) * 2:
                            ruler_section = image[:, x:x + 30]
                            has_text = np.std(ruler_section) > 15 and np.any(
                                ruler_section < np.mean(ruler_section) - 25)

                            if has_text:
                                right_strip = edges[:, x + 15:min(x + 50, width)]

                                if right_strip.size > 0:
                                    right_projection = np.sum(right_strip, axis=1)

                                    if len(right_projection) > 0:
                                        right_autocorr = np.correlate(right_projection, right_projection, mode='full')
                                        right_autocorr = right_autocorr[len(right_autocorr) // 2:]

                                        right_peaks = []
                                        for i in range(15, min(40, len(right_autocorr) - 10)):
                                            if (right_autocorr[i] > right_autocorr[i - 7:i].max() and
                                                    right_autocorr[i] > right_autocorr[i + 1:i + 8].max()):
                                                right_peaks.append(i)

                                        if len(right_peaks) < len(peaks):
                                            left_boundary = x + 35
                                            break

        return top_boundary, left_boundary

    def extract_roi(self, image: np.ndarray) -> Tuple[np.ndarray, dict]:
        if image is None:
            return None, {}

        height, width = image.shape
        top_crop, left_crop = self.detect_rulers_comprehensive(image)

        margin_top = 170
        margin_left = 230  # CAMBIADO: margen izquierdo
        margin_bottom = 150

        # CAMBIADO: Aplicar recorte desde la izquierda
        roi_image = image[
            top_crop + margin_top:max(0, height - margin_bottom),
            left_crop + margin_left:  # CAMBIADO: corta desde la izquierda
        ]

        if self.debug_mode:
            print(f"  - Resulting size: {roi_image.shape[1]} x {roi_image.shape[0]}")

        crop_info = {
            'original_shape': (height, width),
            'top_crop': top_crop,
            'left_crop': left_crop,  # CAMBIADO
            'roi_shape': roi_image.shape,
            'crop_percentage': (roi_image.size / image.size) * 100,
            'removed_top_percentage': (top_crop / height) * 100,
            'removed_left_percentage': (left_crop / width) * 100,  # CAMBIADO
            'removed_top_pixels': top_crop,
            'removed_left_pixels': left_crop  # CAMBIADO
        }

        return roi_image, crop_info

    def process_single_image(self, input_path: str, output_path: str = None,
                             show_result: bool = False) -> dict:
        image = self.load_image(input_path)
        if image is None:
            return {'success': False, 'error': 'Failed to load image'}

        roi_image, crop_info = self.extract_roi(image)

        if roi_image is None:
            return {'success': False, 'error': 'Failed to extract ROI'}

        if output_path:
            try:
                cv2.imwrite(output_path, roi_image)
                crop_info['output_path'] = output_path
            except Exception as e:
                crop_info['save_error'] = str(e)

        if show_result:
            self.visualize_result(image, roi_image, crop_info)

        crop_info['success'] = True
        crop_info['input_path'] = input_path

        return crop_info

    def process_batch(self, input_folder: str, output_folder: str,
                      file_pattern: str = "*.tif") -> list:
        input_path = Path(input_folder)
        output_path = Path(output_folder)
        output_path.mkdir(parents=True, exist_ok=True)

        results = []
        image_files = list(input_path.glob(file_pattern))

        print(f"Processing {len(image_files)} forensic images...")

        for i, img_file in enumerate(image_files):
            print(f"Processing {i + 1}/{len(image_files)}: {img_file.name}")

            output_file = output_path / f"cleaned_roi_{img_file.stem}.tif"
            result = self.process_single_image(str(img_file), str(output_file))
            results.append(result)

        return results

    @staticmethod
    def visualize_result(original: np.ndarray, roi: np.ndarray, crop_info: dict):
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))

        # Original con líneas de corte
        axes[0].imshow(original, cmap='gray')
        axes[0].axhline(y=crop_info['top_crop'], color='red', linestyle='--', linewidth=2)
        axes[0].axvline(x=crop_info['left_crop'], color='blue', linestyle='--', linewidth=2)  # CAMBIADO
        axes[0].set_title(f'Original Image with Crop Lines\n{crop_info["original_shape"]}\n'
                          f'Removing {crop_info["removed_top_percentage"]:.1f}% top, '
                          f'{crop_info["removed_left_percentage"]:.1f}% left')  # CAMBIADO

        # ROI resultado
        axes[1].imshow(roi, cmap='gray')
        axes[1].set_title(f'Cleaned Evidence Image\n{crop_info["roi_shape"]}\n'
                          f'Retained {crop_info["crop_percentage"]:.1f}% of original area')

        for ax in axes:
            ax.axis('on')
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()


def main():
    print("Removido de reglas laterales - CORTE IZQUIERDO")
    print("=" * 60)

    remover = ForensicRulerRemover(
        conservative_crop=True,
        debug_mode=True,
        top_crop_percentage=0.097,
        left_crop_percentage=0.045,  # CAMBIADO
        use_only_percentages=True
    )
    print("-" * 40)

    input_folder = "/Volumes/AlanDisk/Shoes/NikeGel"
    output_folder = "/Volumes/AlanDisk/Shoes/ResultsGel/NikeGelR"

    """

    input_folder = "./proof"
    output_folder = "./predict"
    """

    os.makedirs(output_folder, exist_ok=True)

    if os.path.exists(input_folder):
        with tempfile.TemporaryDirectory() as temp_folder:
            tif_files = 0
            for filename in os.listdir(input_folder):
                if filename.lower().endswith((".tif", ".tiff")):
                    shutil.copy2(
                        os.path.join(input_folder, filename),
                        os.path.join(temp_folder, filename)
                    )
                    tif_files += 1

            print(f"Se copiaron {tif_files} archivos .tif a la carpeta temporal")

            if tif_files == 0:
                print("No hay archivos .tif en la carpeta de entrada. Revisa la ruta.")
            else:
                results = remover.process_batch(temp_folder, output_folder)
                print(f"Procesadas {len(results)} imágenes, guardadas en {output_folder}")
    else:
        print(f"Input folder not found: {input_folder}")


if __name__ == "__main__":
    main()
