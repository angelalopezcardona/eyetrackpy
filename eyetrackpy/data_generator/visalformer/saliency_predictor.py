import numpy as np
import cv2
import os
import pathlib
import sys
# Get the current working directory
cwd = os.getcwd()
sys.path.append(cwd)
from torchvision.utils import save_image
from pathlib import Path
import torch
import matplotlib.pyplot as plt
from eyetrackpy.data_generator.visalformer.model_swin import SalFormer
import torchvision.transforms.functional as TF
from eyetrackpy.data_generator.fixations_predictor.models.model_manager import download_model


# Function to preprocess a single image
 # Add batch dimension



class VisalformerSaliencyPredictor:
    def __init__(self, device=None, model_name = "visalformer"):
        self.model_name = model_name

        cwd = os.path.dirname(os.getcwd())
        if self.model_name == "visalformer":
            trained_weights_path = os.path.join(
                cwd,
                "eyetrackpy",
                "eyetrackpy",
                "data_generator",
                "visalformer",
                "VisSalFormer_weights.tar",
            )
        else:
            trained_weights_path = os.path.join(
                cwd,
                "eyetrackpy",
                "eyetrackpy",
                "data_generator",
                "visalformer",
                "best_model.pth",
            )
        if not os.path.isfile(trained_weights_path):
            download_model(self.model_name)
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        if device == 'cuda':
            if not torch.cuda.is_available():
                print("CUDA is not available. Falling back to CPU.")
                device = 'cpu'
            else:
                print(f"Using GPU: {torch.cuda.get_device_name(0)}")
                print(f"Current device: {device}")
        else:
            print(f"Using CPU: {device}")
        self.device = device
        torch.cuda.empty_cache()  # Clear GPU memory cache
        model = SalFormer.from_pretrained().to(device)
        model.load_ckpt(trained_weights_path, device)
        self.model = model

    
    def predict(self, dataloader, save_path, sharpen_saliency=False):
     
        results_path_saliency = save_path + '/saliency'
        results_path_all = save_path + '/all'
        Path(results_path_saliency).mkdir(parents=True, exist_ok=True)
        Path(results_path_all).mkdir(parents=True, exist_ok=True)

        for batch, (img, input_ids, trial_numbers, original_images) in enumerate(dataloader):
            img = img.to(self.device)
            input_ids = input_ids.to(self.device)
            predictions = self.model(img, input_ids)
            if sharpen_saliency:
                predictions= [ self.sharpen_saliency(p.detach().cpu().squeeze().numpy(), gamma=1.0) for p in predictions]
            predictions = torch.tensor(predictions).to(self.device)
            self.postprocess_predictions(original_images, predictions, results_path_saliency, results_path_all, trial_numbers)
        return predictions
    

   

    def sharpen_saliency(self,saliency_map, gamma=3):
        """
        Sharpen a normalized saliency map using Power Transformation.
        
        Args:
            saliency_map (np.array): 2D array normalized between [0, 1]
            gamma (float): Values > 1 sharpen/concentrate the map. 
                        Values < 1 spread it out.
        Returns:
            np.array: Sharpened and re-normalized saliency map.
        """
        # 1. Apply Power Transformation
        # High values stay high, low-to-mid values drop significantly
        sharpened = np.power(saliency_map, gamma)
        
        # 2. Re-normalize to ensure the peak is exactly 1.0 again
        if sharpened.max() > 0:
            sharpened = sharpened / sharpened.max()
            
        return sharpened
    def postprocess_predictions(self, original_images, predictions, results_path_saliency, results_path_all, trial_numbers):
        for i in range(0, predictions.shape[0]):
            trial_num = trial_numbers[i] if trial_numbers else i
            save_image(predictions[i], f"{results_path_saliency}/saliency_trial_{trial_num}.png")

            # Use the original PIL image directly (no need to convert from tensor)
            img_pil = original_images[i]
            saliency_map = predictions[i].detach().cpu().squeeze().numpy()
            
            # Handle saliency map resizing for visualization only
            from PIL import Image
            import numpy as np
            
            # Get original image dimensions
            img_width, img_height = img_pil.size
            
            # Save the original saliency map (130x130) for metrics
            np.save(f"{results_path_saliency}/saliency_trial_{trial_num}.npy", saliency_map)
            
            # For VISUALIZATION: Resize to match original image (only for display)
            # Check if saliency map is in [0,1] or [0,255] range
            if saliency_map.max() <= 1.0:
                # Values are in [0,1] range, multiply by 255
                saliency_for_pil = (saliency_map * 255).astype(np.uint8)
            else:
                # Values are already in [0,255] range
                saliency_for_pil = saliency_map.astype(np.uint8)
            
            saliency_pil = Image.fromarray(saliency_for_pil)
            saliency_resized = saliency_pil.resize((img_width, img_height), Image.LANCZOS)
            saliency_for_visualization = np.array(saliency_resized) / 255.0
            
            # Plot
            plt.figure(figsize=(16, 4))

            # Panel 1: Original image
            plt.subplot(1, 4, 1)
            plt.imshow(img_pil)
            plt.title("Original Image")
            plt.axis("off")

            # Panel 2: Predicted saliency overlaid
            plt.subplot(1, 4, 2)
            plt.imshow(img_pil)
            plt.imshow(saliency_for_visualization, cmap='hot', alpha=0.6)
            plt.title("Predicted Saliency")
            plt.axis("off")

            plt.tight_layout()
            plt.savefig(f"{results_path_all}/comparison_trial_{trial_num}.png", dpi=300, bbox_inches='tight')
            plt.close()
    
    

