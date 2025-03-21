import os
import gdown

MODEL_FILES = {
    "model_1": {
        "url": "https://drive.google.com/uc?id=1_skRPxLzlY3d68bKk1cf656pYFc--URt",
        "local_path": "data_generator/fixations_predictor_trained_1/T5-tokenizer-BiLSTM-TRT-12-concat-3",
    },
    "model_2": {
        "url": "https://drive.google.com/uc?id=1CTiali54Q7zsT25ciY0y0sIIf2jZVbZG",
        "local_path": "data_generator/fixations_predictor_trained_2/model.pth",
    },
    "mdsem": {
        "url": "https://drive.google.com/uc?id=1piMiLmRveWacGkE48JbfA7nPZfZw2zNQ",
        "local_path": "data_generator/fixations_predictor_mdsem/mdsem_codecharts0_cameraready_weights.hdf5",
    }
}

def download_model(model_name):
    if model_name not in MODEL_FILES:
        raise ValueError(f"Unknown model: {model_name}")
    
    model_info = MODEL_FILES[model_name]
    package_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    full_path = os.path.join(package_dir, model_info["local_path"])
    
    # Check if model already exists
    if os.path.exists(full_path):
        print(f"{model_name} already exists at {full_path}")
        return full_path
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(full_path), exist_ok=True)
    
    # Download the file using gdown
    print(f"Downloading {model_name} from {model_info['url']}...")
    try:
        gdown.download(model_info["url"], full_path, quiet=False)
        print(f"Model saved to: {full_path}")
        return full_path
    except Exception as e:
        raise RuntimeError(f"Failed to download model {model_name}. Error: {e}")

def get_model_path(model_name):
    """Get the path to a model, downloading it if necessary."""
    return download_model(model_name)
