import os
import requests

MODEL_FILES = {
    "model_1": {
        "url": "https://drive.google.com/uc?id=1_skRPxLzlY3d68bKk1cf656pYFc--URt&export=download",
        "local_path": "data_generator/fixations_predictor_trained_1/T5-tokenizer-BiLSTM-TRT-12-concat-3",
    },
    "model_2": {
        "url": "https://drive.google.com/uc?id=1CTiali54Q7zsT25ciY0y0sIIf2jZVbZG&export=download",
        "local_path": "data_generator/fixations_predictor_trained_2/model.pth",
    }
}

def download_model(model_name):
    if model_name not in MODEL_FILES:
        raise ValueError(f"Unknown model: {model_name}")
    
    model_info = MODEL_FILES[model_name]
    package_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    # package_dir = os.path.dirname(os.path.abspath(__file__))
    full_path = os.path.join(package_dir, model_info["local_path"])
    
    # Check if model already exists
    if os.path.exists(full_path):
        return full_path
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(full_path), exist_ok=True)
    
    # Download the file
    print(f"Downloading {model_name}...")
    response = requests.get(model_info["url"])
    if response.status_code == 200:
        with open(full_path, "wb") as f:
            f.write(response.content)
        print(f"Model saved to: {full_path}")
        return full_path
    else:
        raise RuntimeError(f"Failed to download model. Status code: {response.status_code}")

def get_model_path(model_name):
    """Get the path to a model, downloading it if necessary."""
    return download_model(model_name)