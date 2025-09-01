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
from transformers import AutoTokenizer
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
    
    def __init__(self, dataset_name='SalChartQA', split='test', model='bert'):
        self.model = model.lower()


    def create_dataloader(self, batch_size=1):
        self.batch_size = batch_size
        if self.dataset_name == 'salchartqa':
            test_set, padding_fn = self._load_salchartqa_dataset(batch_size)
            
            return DataLoader(test_set, batch_size=batch_size, shuffle=False, collate_fn=padding_fn, num_workers=8)
        
    def _load_padding_fnt(self):
        # if self.model == 'bert':
        def padding_fn(data):
            tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
            img, q = zip(*data)

            input_ids = tokenizer(q, return_tensors="pt", padding=True)

            return torch.stack(img), input_ids

        return padding_fn

