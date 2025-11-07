

import cv2
import numpy as np
import pandas as pd
from skimage.filters import threshold_otsu
from PIL import Image

class SaliencyGenerator():
 
    def generate_saliency_map(self, image_path: str, fixations: np.ndarray, scale_fixations: bool = True, 
                          sigma: int = 60, alpha: float = 0.6, weight_factor: float = 3.0, 
                          return_overlay: bool = False, normalize: bool = True) -> np.ndarray:
        """
        Generates a saliency map based on multiple fixation points.
        
        Args:
            image_path: Path to image or image array
            fixations: Array of fixation points (x, y coordinates)
            scale_fixations: Whether to scale fixations to image dimensions
            sigma: Gaussian blur sigma (higher = more spread)
            alpha: Overlay transparency (0-1)
            weight_factor: Weight per fixation point (higher = stronger saliency)
            return_overlay: Whether to return overlay visualization
            normalize: Whether to normalize output to [0, 1] range
        """
        # Load and validate image
        if isinstance(image_path, np.ndarray):
            image_ = image_path
        elif isinstance(image_path, Image.Image):
            # Convert PIL image to numpy array (RGB to BGR for OpenCV)
            image_ = cv2.cvtColor(np.array(image_path), cv2.COLOR_RGB2BGR)
        else:
            image_ = cv2.imread(image_path)
        if image_ is None:
            raise ValueError(f"Could not load image from path: {image_path}")
        
        height, width = image_.shape[:2]
        # More flexible sigma constraint - allow larger values for better spread
        sigma = min(sigma, min(width, height) // 5)  # Allow up to 20% of smaller dimension
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
        
        # Build an impulse map (delta peaks) instead of drawing disks and blurring each one
        impulse = np.zeros((height, width), dtype=np.float32)
        for (x, y) in valid_fixations:
            # accumulate weight per fixation (use weight_factor as a simple multiplier)
            impulse[y, x] += float(weight_factor)

        # Single Gaussian blur over the whole impulse map
        # kernel size ~ cover ±3σ
        ksize = int(2 * np.ceil(3 * float(sigma)) + 1)
        saliency_map = cv2.GaussianBlur(
            impulse, (ksize, ksize),
            sigmaX=float(sigma), sigmaY=float(sigma),
            borderType=cv2.BORDER_REPLICATE
        )
        
        # Normalize to [0, 1] range if requested (matches generative model scales)
        if normalize:
            max_val = np.max(saliency_map)
            if max_val > 0:
                saliency_map = saliency_map / max_val
        
        if not return_overlay:
            return saliency_map
        
        overlay = self.create_overlay(image_path, saliency_map, alpha)
        return saliency_map, overlay

    @staticmethod
    def create_overlay(image: str, saliency_map: np.ndarray, alpha: float = 0.6) -> np.ndarray:
        # Load and validate image
        if isinstance(image, np.ndarray):
            image_ = image
        elif isinstance(image, Image.Image):
            # Convert PIL image to numpy array (RGB to BGR for OpenCV)
            image_ = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        else:
            image_ = cv2.imread(image)
        
        height, width = image_.shape[:2]
        
        # Enhance contrast for better visualization
        saliency_map_visualization = np.power(saliency_map, 1.5)
        saliency_map_visualization = cv2.normalize(saliency_map_visualization, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        # Create overlay visualization
        heatmap = cv2.applyColorMap(saliency_map_visualization, cv2.COLORMAP_JET)
        
        # Ensure heatmap matches image size
        if heatmap.shape[:2] != (height, width):
            heatmap = cv2.resize(heatmap, (width, height))
        
        # Convert grayscale to RGB if needed
        if len(image_.shape) == 2 or image_.shape[2] == 1:
            image_ = cv2.cvtColor(image_, cv2.COLOR_GRAY2BGR)
        
        # Blend saliency map with the original image
        overlay = cv2.addWeighted(image_, 1 - alpha, heatmap, alpha, 0)
        return overlay

    def create_overlay_and_save_saliency_map(self, image: str, saliency_map: np.ndarray, alpha: float = 0.6, folder: str = None, figure_name: str = None) -> np.ndarray:
        """
        Generates a saliency map based on multiple fixation points.
        Args:
        Returns:
            np.ndarray: Saliency map overlay or raw saliency map based on return_overlay parameter.
        """
        overlay = self.create_overlay(image, saliency_map, alpha)
        self.save_saliency_map(overlay, figure_name, folder)
        
        return True

    def _convert_fixations_to_numpy(self, fixations):
        fixations['x'] = fixations['x'].astype(int)
        fixations['y'] = fixations['y'].astype(int)
        return fixations[['x', 'y']].to_numpy()

    def save_saliency_map(self, overlay, figure_name, folder):
        # Ensure folder path ends with '/' for proper path concatenation
        if not folder.endswith('/'):
            folder = folder + '/'
        
        cv2.imwrite(f"{folder}{figure_name}", overlay)

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
        if np.max(saliency_map) <= 1:
            saliency_map_coverage = saliency_map * 255
        else:
            saliency_map_coverage = saliency_map
        if threshold is None:
            threshold = threshold_otsu(saliency_map_coverage)
        binary_map = saliency_map_coverage > threshold
        coverage = np.sum(binary_map) / binary_map.size
        return coverage, threshold


