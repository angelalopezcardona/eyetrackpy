import pathlib

folder_name = str(pathlib.Path(__file__).parent.resolve().parent.resolve())
import sys

sys.path.append("../")
sys.path.append(folder_name)
from eyetrackpy.data_generator.fixations_predictor_trained_2.fixations_predictor_model_2 import (
    FixationsPredictor2,
)

# Create a new instance of the model
# Ensure the architecture matches the saved state
model_name = "roberta-base"
model_path = folder_name + "/model/model.pth"
modelloader = FixationsPredictor2(model_name="roberta-base", model_path=model_path)
