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
from eyetrackpy.data_generator.visalformer.Code.model_swin import SalFormer
import torchvision.transforms.functional as TF
from eyetrackpy.data_generator.fixations_predictor.models.model_manager import download_model


# Function to preprocess a single image
 # Add batch dimension



class FixationsPredictor:
    def __init__(self, device=None):
        model_name = "visalformer"

        cwd = os.getcwd()
        trained_weights_path = os.path.join(
            cwd,
            "eyetrackpy",
            "data_generator",
            "visalformer",
            "VisSalFormer_weights.tar",
        )
        if not os.path.isfile(trained_weights_path):
            download_model(model_name)
        if device is None:
            if device == 'cuda':
                if not torch.cuda.is_available():
                    print("CUDA is not available. Falling back to CPU.")
                    device = 'cpu'
            else:
                print(f"Using GPU: {torch.cuda.get_device_name(0)}")
                print(f"Current device: {device}")
        self.device = device
        torch.cuda.empty_cache()  # Clear GPU memory cache
        model = SalFormer.from_pretrained().to(device)
        model.load_ckpt(trained_weights_path, device)
        self.model = model

    
    def predict(self, image_path, save_path):

        # Create the dataset loader
        test_dataloader = DatasetLoader(dataset_name=dataset_name, split='test', model='bert').create_dataloader(batch_size=batch_size)
        
        results_path_saliency = save_path + '/saliency'
        results_path_all = save_path + '/all'
        Path(results_path_saliency).mkdir(parents=True, exist_ok=True)
        Path(results_path_all).mkdir(parents=True, exist_ok=True)

        for batch, (img, input_ids) in enumerate(test_dataloader):
            img = img.to(self.device)
            input_ids = input_ids.to(self.device)


            predictions = self.model(img, input_ids)
            self.postprocess_predictions(img, predictions, results_path_saliency, results_path_all)
        return predictions
    

    
    def postprocess_predictions(self, img, predictions, results_path_saliency, results_path_all):
        for i in range(0, predictions.shape[0]):
            save_image(predictions[i], f"{results_path_saliency}")

            img_np = TF.to_pil_image(img[i].cpu()).convert("RGB")
            saliency_map = predictions[i].detach().cpu().squeeze().numpy()

            # Plot
            plt.figure(figsize=(16, 4))

            # Panel 1: Original image
            plt.subplot(1, 4, 1)
            plt.imshow(img_np)
            plt.title("Original Image")
            plt.axis("off")

            # Panel 2: Predicted saliency overlaid
            plt.subplot(1, 4, 2)
            plt.imshow(img_np)
            plt.imshow(saliency_map, cmap='hot', alpha=0.6)
            plt.title("Predicted Saliency")
            plt.axis("off")

            plt.tight_layout()
            plt.savefig(f"{results_path_all}/_comparison.png", dpi=300, bbox_inches='tight')
            plt.close()
    
    

