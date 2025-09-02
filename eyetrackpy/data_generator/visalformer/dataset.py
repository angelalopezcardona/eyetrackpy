import torch
from torch.utils.data import Dataset
import numpy as np

class ImagesWithSaliency(Dataset):
    def __init__(self, npy_path, dtype=None):
        self.dtype = dtype
        self.datas = np.load(npy_path, allow_pickle = True)

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, idx):
        if self.dtype:
            self.datas[idx][0] = self.datas[idx][0].type(self.dtype)
            self.datas[idx][3] = self.datas[idx][3].type(self.dtype)

        return self.datas[idx]

import torch
from torch.utils.data import Dataset
import numpy as np
from transformers import AutoTokenizer, AutoImageProcessor
from torch.utils.data import DataLoader
import os
import sys

# for SalChartQA dataset
class ImagesWithSaliency(Dataset):
    def __init__(self, npy_path, dtype=None):
        self.dtype = dtype
        self.datas = self._find_and_load_npy(npy_path)

    def _find_and_load_npy(self, npy_path, max_depth=3):
        """Search for the npy file in current directory and parent directories up to max_depth levels."""
        current_dir = os.getcwd()
        
        for depth in range(max_depth + 1):
            # Try current directory
            file_path = os.path.join(current_dir, npy_path)
            if os.path.exists(file_path):
                return np.load(file_path, allow_pickle=True)
            
            # Move up one directory level
            current_dir = os.path.dirname(current_dir)
            
            # Stop if we've reached the root directory
            if current_dir == os.path.dirname(current_dir):
                break
        
        # If we get here, the file wasn't found
        raise FileNotFoundError(f"Could not find {npy_path} in current directory or {max_depth} parent directories")

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, idx):
        if self.dtype:
            self.datas[idx][0] = self.datas[idx][0].type(self.dtype)
            self.datas[idx][3] = self.datas[idx][3].type(self.dtype)

        return self.datas[idx]


class DatasetLoader:
    
    def __init__(self, model='bert', vision_model='microsoft/swin-tiny-patch4-window7-224'):
        self.model = model.lower()
        self.vision_model = vision_model
        if self.model == 'bert':
            self.model_name="bert-base-uncased"
        
        # Initialize the image processor for the vision model
        self.image_processor = AutoImageProcessor.from_pretrained(vision_model)


    def create_dataloader(self, data, batch_size=1):
        self.batch_size = batch_size
        padding_fn = self._load_padding_fn()
            
        return DataLoader(data, batch_size=batch_size, shuffle=False, collate_fn=padding_fn, num_workers=8)
        
    def _load_padding_fn(self):
        def padding_fn(data):
            tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            
            # Extract trial_numbers, images and questions from the data structure
            # New data structure is [trial_number, image, question, ...]
            trial_list = []
            img_list = []
            original_img_list = []  # Store original PIL images for plotting
            q_list = []
            
            for item in data:
                # Extract trial_number (index 0), image (index 1), question (index 2)
                trial_number = item[0]
                img_data = item[1]
                question = item[2]
                
                # Convert to PIL Image if needed
                from PIL import Image
                
                if isinstance(img_data, str):
                    # Image path provided, load the image
                    img_pil = Image.open(img_data).convert('RGB')
                elif isinstance(img_data, Image.Image):
                    # Already a PIL Image, use it directly
                    img_pil = img_data
                elif isinstance(img_data, np.ndarray):
                    # Ensure the image is in the correct format (H, W, C) and uint8
                    img_array = img_data
                    if img_array.dtype != np.uint8:
                        # Convert to uint8 if not already
                        if img_array.max() <= 1.0:
                            img_array = (img_array * 255).astype(np.uint8)
                        else:
                            img_array = img_array.astype(np.uint8)
                    
                    # Convert numpy array to PIL Image
                    img_pil = Image.fromarray(img_array)
                else:
                    # If it's a tensor, convert to PIL Image
                    from torchvision.transforms.functional import to_pil_image
                    if img_data.dtype == torch.uint8:
                        img_pil = to_pil_image(img_data)
                    else:
                        # Convert float tensor back to uint8 for PIL
                        img_tensor_uint8 = (img_data * 255).clamp(0, 255).byte()
                        img_pil = to_pil_image(img_tensor_uint8)
                
                # Use the image processor to preprocess the image
                img_tensor = self.image_processor(img_pil, return_tensors="pt")["pixel_values"].squeeze(0)
                
                trial_list.append(trial_number)
                img_list.append(img_tensor)
                original_img_list.append(img_pil)  # Store original PIL image
                q_list.append(question)

            input_ids = tokenizer(q_list, return_tensors="pt", padding=True)

            return torch.stack(img_list), input_ids, trial_list, original_img_list

        return padding_fn

