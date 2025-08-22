

import cv2
import numpy as np
import pandas as pd
from skimage.filters import threshold_otsu

class SaliencyGenerator():
 
    def generate_saliency_map(self, image_path: str, fixations: np.ndarray, scale_fixations: bool = False, 
                             sigma: int = 60, alpha: float = 0.6, weight_factor: float = 3.0, 
                             return_overlay: bool = False) -> np.ndarray:
        """
        Generates a saliency map based on multiple fixation points.

        Args:
            image_path (str): Path to the input image.
            fixations (np.ndarray): Array of (x, y) fixation points.
            scale_fixations (bool): Whether to scale fixations to image dimensions.
            sigma (int): Standard deviation for Gaussian blobs (higher = more spread).
            alpha (float): Opacity of the saliency map overlay (0-1).
            weight_factor (float): Multiplier to enhance fixation visibility.
            return_overlay (bool): If True, return overlay image; if False, return saliency map.

        Returns:
            np.ndarray: Saliency map overlay or raw saliency map based on return_overlay parameter.
        """
        # Load and validate image
        image_ = cv2.imread(image_path)
        if image_ is None:
            raise ValueError(f"Could not load image from path: {image_path}")
        
        height, width = image_.shape[:2]
        
        # Validate and process fixations
        if len(fixations) == 0:
            raise ValueError("No fixation points provided")
            
        if isinstance(fixations, pd.DataFrame):
            if scale_fixations:
                fixations = self._scale_fixations(fixations, width, height)
            fixations = self._convert_fixations_to_numpy(fixations)
        elif scale_fixations:
            raise ValueError("Cannot scale fixations if not a pandas DataFrame")
        
        # Validate fixation coordinates
        valid_fixations = []
        for (x, y) in fixations:
            if 0 <= x < width and 0 <= y < height:
                valid_fixations.append((x, y))
        
        if len(valid_fixations) == 0:
            raise ValueError("No valid fixation points found within image boundaries")
        
        # Initialize saliency map
        saliency_map = np.zeros((height, width), dtype=np.float32)
        
        # Calculate kernel size for Gaussian blur (should be odd and related to sigma)
        kernel_size = max(3, int(sigma * 2 + 1))
        if kernel_size % 2 == 0:
            kernel_size += 1  # Ensure odd kernel size
        
        # Generate saliency map from fixations
        for (x, y) in valid_fixations:
            # Create Gaussian blob at fixation point
            gaussian = np.zeros((height, width), dtype=np.float32)
            cv2.circle(gaussian, (int(x), int(y)), sigma, 255 * weight_factor, -1)
            
            # Apply proper Gaussian blur
            gaussian = cv2.GaussianBlur(gaussian, (kernel_size, kernel_size), sigma)
            
            # Accumulate to saliency map
            saliency_map += gaussian
        
        # Normalize saliency map to [0, 255] range
        if saliency_map.max() > 0:
            saliency_map = cv2.normalize(saliency_map, None, 0, 255, cv2.NORM_MINMAX)
        
        # Enhance contrast for better visualization
        saliency_map = np.power(saliency_map, 1.5)
        saliency_map = cv2.normalize(saliency_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        if not return_overlay:
            return saliency_map
        
        # Create overlay visualization
        heatmap = cv2.applyColorMap(saliency_map, cv2.COLORMAP_JET)
        
        # Ensure heatmap matches image size
        if heatmap.shape[:2] != (height, width):
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
    
    @staticmethod
    def compute_shannon_entropy(saliency_map: np.ndarray) -> float:
        """
        Compute the Shannon Entropy of the saliency map.

        Args:
            saliency_map (np.ndarray): Grayscale saliency map.

        Returns:
            float: Shannon Entropy value.
        """
        total = np.sum(saliency_map)
        if total > 0:
            saliency_norm = saliency_map / total
        else:
            return 0.0
        p = saliency_norm[saliency_norm > 0]
        entropy = -np.sum(p * np.log2(p))
        return entropy

    @staticmethod
    def compute_saliency_coverage(saliency_map: np.ndarray, threshold: float = None) -> tuple[float, float]:
        """
        Compute the Saliency Coverage (percentage of the image above threshold).

        Args:
            saliency_map (np.ndarray): Grayscale saliency map.
            threshold (float, optional): Threshold value. If not provided, Otsu's method is used.

        Returns:
            tuple: (coverage (float), used threshold (float))
        """
        if threshold is None:
            threshold = threshold_otsu(saliency_map)
        binary_map = saliency_map > threshold
        coverage = np.sum(binary_map) / binary_map.size
        return coverage, threshold


