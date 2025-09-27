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
    """
    Specialized ROI extractor for forensic evidence images with complex ruler removal.
    Optimized for images with measurement rulers containing text, numbers, and tick marks.
    """

    def __init__(self,
                 conservative_crop: bool = True,
                 debug_mode: bool = False,
                 top_crop_percentage: float = 0.15,  # Remove the top 15%
                 right_crop_percentage: float = 0.12,  # Remove right 12% (LESS AGGRESSIVE)
                 use_only_percentages: bool = False):  # NEW: Force use only percentages
        """
        Initialize the forensic ruler remover.

        Args:
            conservative_crop: If True, uses safer but more aggressive cropping
            debug_mode: If True, shows detection process details
            top_crop_percentage: Percentage to remove from top (conservative fallback)
            right_crop_percentage: Percentage to remove from right (conservative fallback)
            use_only_percentages: If True, ignores automatic detection and uses only percentages
        """
        self.conservative_crop = conservative_crop
        self.debug_mode = debug_mode
        self.top_crop_percentage = top_crop_percentage
        self.right_crop_percentage = right_crop_percentage
        self.use_only_percentages = use_only_percentages

    @staticmethod
    def load_image(image_path: str) -> np.ndarray:
        """Load .tif image using PIL to handle various formats."""
        try:
            with Image.open(image_path) as img:
                if img.mode != 'L':
                    img = img.convert('L')
                return np.array(img)
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            return None

    def detect_rulers_comprehensive(self, image: np.ndarray) -> Tuple[int, int]:
        """
        Comprehensive ruler detection using multiple aggressive methods.

        Returns:
            Tuple of (top_crop_line, right_crop_line)
        """
        height, width = image.shape

        if self.debug_mode:
            print(f"Image dimensions: {width} x {height}")

        # If you use_only_percentages is True, skip all detection and use only percentages
        if self.use_only_percentages:
            top_crop = int(height * self.top_crop_percentage)
            right_crop = int(width * (1 - self.right_crop_percentage))

            return top_crop, right_crop

        # Method 1: Find rulers by detecting uniform background patterns
        top_uniform, right_uniform = self._detect_uniform_ruler_background(image)

        # Method 2: Color/intensity histogram analysis
        top_histogram, right_histogram = self._detect_rulers_by_histogram(image)

        # Method 3: Text density detection (most reliable for forensic rulers)
        top_text_density, right_text_density = self._detect_text_density_boundaries(image)

        # Method 4: Edge pattern analysis for ruler tick marks
        top_edge_pattern, right_edge_pattern = self._detect_ruler_edge_patterns(image)

        # Method 5: Conservative percentage-based approach (FIXED - CORRECT LOGIC)
        top_conservative = int(height * self.top_crop_percentage)  # Remove from TOP
        right_conservative = int(width * (1 - self.right_crop_percentage))  # KEEP from LEFT (crop from right)

        # Combine methods with priority weighting
        top_candidates = [top_uniform, top_histogram, top_text_density, top_edge_pattern]
        right_candidates = [right_uniform, right_histogram, right_text_density, right_edge_pattern]

        # Filter out invalid candidates and use a conservative approach if needed
        valid_top = [t for t in top_candidates if 50 <= t <= height // 2]
        valid_right = [r for r in right_candidates if width // 2 <= r <= width - 30]  # Less restrictive

        # TOP BOUNDARY: Remove from top, so we want the MAXIMUM (most conservative)
        if self.conservative_crop or not valid_top:
            final_top = max(top_conservative, max(valid_top) if valid_top else top_conservative)
        else:
            final_top = int(np.median(valid_top)) if valid_top else top_conservative

        # RIGHT BOUNDARY: Keep from left, so we want the MAXIMUM (keeps more evidence)
        if self.conservative_crop or not valid_right:
            # Use the MAXIMUM of detected boundaries (keeps more evidence)
            final_right = max(right_conservative, max(valid_right) if valid_right else right_conservative)
        else:
            # Use the 75th percentile instead of median to be more conservative
            final_right = int(np.percentile(valid_right, 75)) if valid_right else right_conservative

        return final_top, final_right

    @staticmethod
    def _detect_uniform_ruler_background(image: np.ndarray) -> Tuple[int, int]:
        """Detect rulers by finding uniform background regions."""
        height, width = image.shape

        # Top ruler detection
        top_boundary = 80  # Default fallback
        search_height = min(200, height // 3)

        # Analyze horizontal strips
        strip_height = 10
        for y in range(50, search_height - strip_height, strip_height):
            strip = image[y:y + strip_height, :]
            strip_std = np.std(strip)

            # Look for low variation (uniform ruler background) followed by high variation
            next_strip = image[y + strip_height:min(y + 2 * strip_height, height), :]
            if next_strip.size > 0:
                next_std = np.std(next_strip)

                # If the current strip is uniform but the next has more variation
                if strip_std < 15 and next_std > strip_std * 2 and next_std > 30:
                    top_boundary = y + strip_height + 20  # Add safety margin
                    break

        # Right ruler detection - PRECISION MODE (find actual ruler, don't cut evidence)
        right_boundary = int(width * 0.88)  # Less aggressive default fallback
        search_width = min(250, width // 3)  # Search narrower area

        # Analyze vertical strips from the right side-look for actual ruler characteristics
        strip_width = 8
        for x in range(width - 50, width - search_width, -strip_width):
            strip = image[:, x - strip_width:x]
            strip_std = np.std(strip)
            strip_mean = np.mean(strip)

            # Look for ruler characteristics: uniform background and high contrast elements (text/numbers)
            prev_strip = image[:, max(0, x - 2 * strip_width):x - strip_width]
            if prev_strip.size > 0:
                prev_mean = np.mean(prev_strip)

                # Check for a ruler pattern: low background variation and high contrast text
                has_text_elements = np.any(strip < prev_mean - 30)  # Dark text elements
                has_uniform_bg = strip_std < 20  # Uniform background
                intensity_difference = abs(strip_mean - prev_mean) > 15  # Different from evidence

                if has_uniform_bg and has_text_elements and intensity_difference:
                    right_boundary = x - strip_width - 15  # Smaller safety margin
                    break

        return top_boundary, right_boundary

    @staticmethod
    def _detect_rulers_by_histogram(image: np.ndarray) -> Tuple[int, int]:
        """Detect rulers using intensity histogram analysis."""
        height, width = image.shape

        # Top ruler detection
        top_boundary = 100
        search_height = min(250, height // 2)

        # Calculate histogram for potential ruler area
        ruler_area = image[:search_height, :]
        ruler_hist, _ = np.histogram(ruler_area.flatten(), bins=50, range=(0, 255))

        # Calculate histogram for rest of image
        evidence_area = image[search_height:, :]
        if evidence_area.size > 0:
            evidence_hist, _ = np.histogram(evidence_area.flatten(), bins=50, range=(0, 255))

            # Compare histogram distributions
            hist_diff = np.sum(np.abs(ruler_hist - evidence_hist))

            # If significant difference, ruler likely present
            if hist_diff > np.mean(ruler_hist) * 10:
                # Find a transition point by analyzing row-wise histograms
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

        # Right ruler detection - PRECISION MODE
        right_boundary = int(width * 0.88)  # Less aggressive default
        search_width = min(200, width // 4)  # Narrower search

        # Calculate histogram for potential ruler area from right
        ruler_area_right = image[:, width - search_width:]
        ruler_hist_right, _ = np.histogram(ruler_area_right.flatten(), bins=50, range=(0, 255))

        # Calculate histogram for evidence area (middle portion to avoid left artifacts)
        evidence_start = width // 4
        evidence_end = width - search_width
        evidence_area_right = image[:, evidence_start:evidence_end]

        if evidence_area_right.size > 0:
            evidence_hist_right, _ = np.histogram(evidence_area_right.flatten(), bins=50, range=(0, 255))

            # Compare histogram distributions - be more selective
            hist_diff_right = np.sum(np.abs(ruler_hist_right - evidence_hist_right))

            if hist_diff_right > np.mean(ruler_hist_right) * 12:  # Higher threshold
                # Find a transition point by analyzing column-wise histograms
                for x in range(width - 80, width - search_width, -15):  # Larger steps, less aggressive start
                    current_area = image[:, x - 15:x]
                    left_area = image[:, max(0, x - 45):x - 15]

                    if left_area.size > 0 and current_area.size > 0:
                        current_hist, _ = np.histogram(current_area.flatten(), bins=20)
                        left_hist, _ = np.histogram(left_area.flatten(), bins=20)

                        col_diff = np.sum(np.abs(current_hist - left_hist))
                        if col_diff > np.mean(current_hist) * 4:  # Higher threshold
                            right_boundary = x - 25  # Moderate safety margin
                            break

        return top_boundary, right_boundary

    @staticmethod
    def _detect_text_density_boundaries(image: np.ndarray) -> Tuple[int, int]:
        """Detect ruler boundaries by analyzing text density."""
        height, width = image.shape

        # Enhanced edge detection for text
        edges = cv2.Canny(image, 50, 150, apertureSize=3, L2gradient=True)

        # Morphological operations to connect text components
        kernel_horizontal = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 3))
        kernel_vertical = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 9))

        # Top ruler detection
        top_boundary = 120
        search_height = min(300, height // 2)

        # Apply horizontal morphology for horizontal text detection
        text_enhanced = cv2.morphologyEx(edges[:search_height, :], cv2.MORPH_CLOSE, kernel_horizontal)

        # Calculate text density for each row
        row_text_density = np.sum(text_enhanced, axis=1)

        # Smooth the density signal
        if len(row_text_density) > 10:
            smoothed = cv2.GaussianBlur(row_text_density.reshape(-1, 1), (11, 1), 0).flatten()

            # Find where text density drops significantly
            mean_density = np.mean(smoothed[:len(smoothed) // 2])

            for i in range(60, len(smoothed) - 20):
                window = smoothed[i:i + 20]
                if len(window) > 0 and np.mean(window) < mean_density * 0.3:
                    top_boundary = i + 20
                    break

        # Right ruler detection - PRECISION MODE
        right_boundary = int(width * 0.88)  # Less aggressive default
        search_width = min(300, width // 3)  # Moderate search area

        # Apply vertical morphology for vertical text detection
        text_enhanced_right = cv2.morphologyEx(edges[:, width - search_width:], cv2.MORPH_CLOSE, kernel_vertical)

        # Calculate text density for each column
        col_text_density = np.sum(text_enhanced_right, axis=0)

        # Smooth the density signal
        if len(col_text_density) > 10:
            smoothed_right = cv2.GaussianBlur(col_text_density.reshape(1, -1), (1, 11), 0).flatten()

            # Find where text density drops significantly from right - LESS SENSITIVE
            mean_density_right = np.mean(smoothed_right[len(smoothed_right) // 2:])

            # Only proceed if there's actually significant text density in the ruler area
            if mean_density_right > 50:  # Minimum text density threshold
                for i in range(len(smoothed_right) - 60, 30, -1):  # Less aggressive start
                    window = smoothed_right[max(0, i - 25):i]
                    if len(window) > 0 and np.mean(window) < mean_density_right * 0.5:  # Less sensitive
                        right_boundary = width - search_width + i - 25  # Moderate safety margin
                        break

        return top_boundary, right_boundary

    @staticmethod
    def _detect_ruler_edge_patterns(image: np.ndarray) -> Tuple[int, int]:
        """Detect rulers by finding periodic tick mark patterns."""
        height, width = image.shape

        # Apply edge detection
        edges = cv2.Canny(image, 30, 100)

        # Top ruler - look for vertical tick marks
        top_boundary = 100
        search_height = min(200, height // 3)

        # Look for periodic vertical patterns in horizontal strips
        for y in range(60, search_height - 20, 10):
            strip = edges[y:y + 20, :]

            # Find vertical lines (ruler ticks)
            vertical_projection = np.sum(strip, axis=0)

            # Look for periodicity in the projection
            if len(vertical_projection) > 100:
                # Simple periodicity detection
                autocorr = np.correlate(vertical_projection, vertical_projection, mode='full')
                autocorr = autocorr[len(autocorr) // 2:]

                # If we find strong periodic patterns, this is likely a ruler
                if len(autocorr) > 50:
                    peaks = []
                    for i in range(10, min(50, len(autocorr) - 10)):
                        if (autocorr[i] > autocorr[i - 5:i].max() and
                                autocorr[i] > autocorr[i + 1:i + 6].max()):
                            peaks.append(i)

                    if len(peaks) >= 3:  # Found a periodic pattern
                        # Check if a pattern stops below this line
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

                            if len(below_peaks) < len(peaks):  # Pattern weakens
                                top_boundary = y + 30
                                break

        # Right ruler - PRECISION SEARCH (preserve evidence)
        right_boundary = int(width * 0.88)  # Less aggressive default
        search_width = min(250, width // 4)  # Narrower search

        # Similar process for right ruler - look for actual ruler patterns, not just edges
        for x in range(width - 70, width - search_width, -15):  # Less aggressive start, bigger steps
            strip = edges[:, x - 15:x]

            if strip.size > 0:
                # Find horizontal lines (ruler ticks)
                horizontal_projection = np.sum(strip, axis=1)

                # Look for periodicity - but be more selective
                if len(horizontal_projection) > 100:
                    autocorr = np.correlate(horizontal_projection, horizontal_projection, mode='full')
                    autocorr = autocorr[len(autocorr) // 2:]

                    if len(autocorr) > 50:
                        peaks = []
                        for i in range(15, min(40, len(autocorr) - 10)):  # Narrower search range
                            if (autocorr[i] > autocorr[i - 7:i].max() and
                                    autocorr[i] > autocorr[i + 1:i + 8].max()):
                                peaks.append(i)

                        # Require stronger evidence of ruler pattern
                        if len(peaks) >= 3 and max(autocorr[peaks]) > np.mean(autocorr) * 2:
                            # Verify this is actually a ruler by checking text/number presence
                            ruler_section = image[:, x - 30:x]
                            has_text = np.std(ruler_section) > 15 and np.any(
                                ruler_section < np.mean(ruler_section) - 25)

                            if has_text:
                                # Check if a pattern stops to the left
                                left_strip = edges[:, max(0, x - 50):x - 15]

                                if left_strip.size > 0:
                                    left_projection = np.sum(left_strip, axis=1)

                                    if len(left_projection) > 0:
                                        left_autocorr = np.correlate(left_projection, left_projection, mode='full')
                                        left_autocorr = left_autocorr[len(left_autocorr) // 2:]

                                        left_peaks = []
                                        for i in range(15, min(40, len(left_autocorr) - 10)):
                                            if (left_autocorr[i] > left_autocorr[i - 7:i].max() and
                                                    left_autocorr[i] > left_autocorr[i + 1:i + 8].max()):
                                                left_peaks.append(i)

                                        if len(left_peaks) < len(peaks):  # Pattern weakens
                                            right_boundary = x - 35  # Moderate safety margin
                                            break

        return top_boundary, right_boundary

    def extract_roi(self, image: np.ndarray) -> Tuple[np.ndarray, dict]:
        """
        Extract ROI by removing ruler marks.

        Args:
            image: Input grayscale image

        Returns:
            Tuple of (cropped_image, crop_info)
        """
        if image is None:
            return None, {}

        height, width = image.shape

        # Detect rulers using a comprehensive method
        top_crop, right_crop = self.detect_rulers_comprehensive(image)

        margin_top = 170  # píxeles a recortar arriba
        margin_right = 230  # píxeles a recortar a la derecha
        margin_bottom = 150

        # Aplicar recorte seguro
        roi_image = image[
            top_crop + margin_top: max(0, image.shape[0] - margin_bottom), :max(0, right_crop - margin_right)]

        if self.debug_mode:
            print(f"  - Resulting size: {roi_image.shape[1]} x {roi_image.shape[0]}")

        crop_info = {
            'original_shape': (height, width),
            'top_crop': top_crop,
            'right_crop': right_crop,
            'roi_shape': roi_image.shape,
            'crop_percentage': (roi_image.size / image.size) * 100,
            'removed_top_percentage': (top_crop / height) * 100,
            'removed_right_percentage': ((width - right_crop) / width) * 100,
            'removed_top_pixels': top_crop,
            'removed_right_pixels': width - right_crop
        }

        return roi_image, crop_info

    def process_single_image(self, input_path: str, output_path: str = None,
                             show_result: bool = False) -> dict:
        """Process a single forensic evidence image."""
        # Load image
        image = self.load_image(input_path)
        if image is None:
            return {'success': False, 'error': 'Failed to load image'}

        # Extract ROI
        roi_image, crop_info = self.extract_roi(image)

        if roi_image is None:
            return {'success': False, 'error': 'Failed to extract ROI'}

        # Save the result if an output path provided
        if output_path:
            try:
                cv2.imwrite(output_path, roi_image)
                crop_info['output_path'] = output_path
            except Exception as e:
                crop_info['save_error'] = str(e)

        # Display result if requested
        if show_result:
            self.visualize_result(image, roi_image, crop_info)

        crop_info['success'] = True
        crop_info['input_path'] = input_path

        return crop_info

    def process_batch(self, input_folder: str, output_folder: str,
                      file_pattern: str = "*.tif") -> list:
        """Process multiple forensic evidence images in a folder."""
        input_path = Path(input_folder)
        output_path = Path(output_folder)
        output_path.mkdir(parents=True, exist_ok=True)

        results = []
        image_files = list(input_path.glob(file_pattern))

        print(f"Processing {len(image_files)} forensic images...")

        for i, img_file in enumerate(image_files):
            print(f"Processing {i + 1}/{len(image_files)}: {img_file.name}")

            output_file = output_path / f"cleaned_roi_{img_file.stem}.tif"  # FIXED: Generate the correct filename
            result = self.process_single_image(str(img_file), str(output_file))
            results.append(result)

        return results

    @staticmethod
    def visualize_result(original: np.ndarray, roi: np.ndarray, crop_info: dict):
        """Visualize the ROI extraction result."""
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))

        # Original image with crop lines
        axes[0].imshow(original, cmap='gray')
        axes[0].axhline(y=crop_info['top_crop'], color='red', linestyle='--', linewidth=2)
        axes[0].axvline(x=crop_info['right_crop'], color='red', linestyle='--', linewidth=2)
        axes[0].set_title(f'Original Image with Crop Lines\n{crop_info["original_shape"]}\n'
                          f'Removing {crop_info["removed_top_percentage"]:.1f}% top, '
                          f'{crop_info["removed_right_percentage"]:.1f}% right')

        # ROI result
        axes[1].imshow(roi, cmap='gray')
        axes[1].set_title(f'Cleaned Evidence Image\n{crop_info["roi_shape"]}\n'
                          f'Retained {crop_info["crop_percentage"]:.1f}% of original area')

        for ax in axes:
            ax.axis('on')
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()


def main():
    print("Removido de reglas laterales")
    print("=" * 60)

    remover = ForensicRulerRemover(
        conservative_crop=True,
        debug_mode=True,
        top_crop_percentage=0.08,
        right_crop_percentage=0.12,
        use_only_percentages=False
    )
    print("-" * 40)

    input_folder = "/Volumes/AlanDisk/Shoes/Nike(blood)"
    output_folder = "/Volumes/AlanDisk/Shoes/Nike(results)"

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
