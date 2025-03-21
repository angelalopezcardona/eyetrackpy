import os
import sys
import pathlib

cwd = os.getcwd()
sys.path.append(cwd)
from eyetrackpy.data_generator.fixations_predictor_mdsem.fixations_predictor import FixationsPredictor

if __name__ == '__main__':
    #----------------------------------------


    #----------------------------------------
    #load model
    # Path to the saved weights or trained model
    fixations_predictor = FixationsPredictor()
    #----------------------------------------
    image_path = cwd + '/examples/data/example_image.png'
    save_path = cwd + '/examples/data_generator/results/'
    print(image_path)
    predictions = fixations_predictor.predict(image_path, save_path)
    # print(predictions)
    # Convert to numpy arrays and save as a list

