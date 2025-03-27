

import cv2
import numpy as np
import pandas as pd

class SaliencyPlotter():
    def draw_point(
        image: np.ndarray, points: np.ndarray, line_color: (int, int, int), output_name: str
    ):
        c = 0
        color_step = 255 // points.shape[0]
        color = [0, 0, 255]
        radius = 25
        image = image.copy()
        for x, y in points:
            color[0] += color_step
            color[2] -= color_step
            cv2.circle(image, (x, y), radius, color, -1)
            if c != 0:
                cv2.line(image, (points[c - 1][0], points[c - 1][1]), (x, y), color, 2)
            c += 1
        cv2.imwrite(f"visualization/{output_name}.png", image)
        print(f'points={points}')
        print("#################################")


    def generate_saliency_map(self, image_path: str, fixations: np.ndarray, scale_fixations: bool = False, sigma: int = 60, alpha: float = 0.6, weight_factor: float = 3.0) -> np.ndarray:
        """
        Generates and overlays a saliency map based on multiple fixation points.

        Args:
            image (np.ndarray): Input image (H, W, 3).
            fixations (np.ndarray): Array of (x, y) fixation points.
            sigma (int): Standard deviation for Gaussian blobs (higher = more spread).
            alpha (float): Opacity of the saliency map overlay (0-1).
            weight_factor (float): Multiplier to enhance fixation visibility.

        Returns:
            np.ndarray: Image with the accumulated saliency map overlay.
        """
        image_ = cv2.imread(image_path)
        height, width, _ = image_.shape
        
        if isinstance(fixations, pd.DataFrame):
                if scale_fixations:
                    fixations = self._scale_fixations(fixations, width, height)
                fixations = self._convert_fixations_to_numpy(fixations)
        elif scale_fixations:
            raise ValueError("You cant scale fixations if is not a pandas DataFrame")
                
        saliency_map = np.zeros((height, width), dtype=np.float32)

        # Accumulate all fixation points together
        for (x, y) in fixations:
            if 0 <= x < width and 0 <= y < height:
                gaussian = np.zeros((height, width), dtype=np.float32)
                cv2.circle(gaussian, (x, y), sigma, 255 * weight_factor, -1)  # Draw circle at fixation
                gaussian = cv2.GaussianBlur(gaussian, (0, 0), sigma)  # Apply Gaussian blur
                saliency_map += gaussian  # ✅ Accumulate (instead of overwriting)

        # Normalize the saliency map to range [0, 255]
        saliency_map = cv2.normalize(saliency_map, None, 0, 255, cv2.NORM_MINMAX)

        # Enhance contrast to make multiple fixations clearer
        saliency_map = np.power(saliency_map, 1.5)
        saliency_map = cv2.normalize(saliency_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        # Apply colormap for visualization
        heatmap = cv2.applyColorMap(saliency_map, cv2.COLORMAP_JET)

        # Ensure heatmap matches image size
        heatmap = cv2.resize(heatmap, (width, height))

        # Convert grayscale to RGB if needed
        if len(image_.shape) == 2 or image_.shape[2] == 1:
            image_ = cv2.cvtColor(image_, cv2.COLOR_GRAY2BGR)

        # Blend saliency map with the original image
        overlay = cv2.addWeighted(image_, 1 - alpha, heatmap, alpha, 0)
       
        return overlay
    
    def _convert_fixations_to_numpy(self, fixations):
        fixations['x'] = fixations['x'].astype(int)
        fixations['y'] = fixations['y'].astype(int)
        return fixations[['x', 'y']].to_numpy()

    def save_saliency_map(self, overlay, figure_name, folder):
         cv2.imwrite(f"{folder}/{figure_name}", overlay)

    def _scale_fixations(self, fixations, width, height):
        fixations['x'] = fixations['x'] * width
        fixations['y'] = fixations['y'] * height
        return fixations