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
    def __init__(self, device=None):
        model_name = "visalformer"

        cwd = os.path.dirname(os.getcwd())
        trained_weights_path = os.path.join(
            cwd,
            "eyetrackpy",
            "eyetrackpy",
            "data_generator",
            "visalformer",
            "VisSalFormer_weights.tar",
        )
        if not os.path.isfile(trained_weights_path):
            download_model(model_name)
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

    
    def predict(self, dataloader, save_path):

     
        results_path_saliency = save_path + '/saliency'
        results_path_all = save_path + '/all'
        Path(results_path_saliency).mkdir(parents=True, exist_ok=True)
        Path(results_path_all).mkdir(parents=True, exist_ok=True)

        for batch, (img, input_ids, trial_numbers, original_images) in enumerate(dataloader):
            img = img.to(self.device)
            input_ids = input_ids.to(self.device)
            predictions = self.model(img, input_ids)
            self.postprocess_predictions(original_images, predictions, results_path_saliency, results_path_all, trial_numbers)
        return predictions
    

    
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
            saliency_pil = Image.fromarray((saliency_map * 255).astype(np.uint8))
            saliency_resized = saliency_pil.resize((img_width, img_height), Image.BILINEAR)
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
    
    

