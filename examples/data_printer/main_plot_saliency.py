import os
import sys

# Get the current working directory and add it to path
cwd = os.getcwd()
sys.path.append(cwd)

from eyetrackpy.data_printer.models.saliency_plotter import SaliencyPlotter
import pandas as pd
from eyetrackpy.data_processor.models.gazepoint import Gazepoint
if __name__ == "__main__":
    plotter = SaliencyPlotter()
    fixations = pd.read_csv(cwd + '/examples/data/fixations.csv')
    image_path = cwd + '/examples/data/example_image.png'
    save_directory = cwd + '/examples/data_printer/results/'
    fixations = Gazepoint().preprocess_fixations_trial(fixations)
    saliency_map = plotter.generate_saliency_map(image_path, fixations, scale_fixations=True, sigma= 20, alpha= 0.6, weight_factor = 3.0) 
    plotter.save_saliency_map(saliency_map, 'saliency_map.png', save_directory)


