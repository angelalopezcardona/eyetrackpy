import os
import sys

# Get the current working directory and add it to path
cwd = os.getcwd()
sys.path.append(cwd)

from eyetrackpy.data_printer.models.scanpath_plotter import ScanpathPlotter
import pandas as pd
from eyetrackpy.data_processor.models.gazepoint import Gazepoint

plotter = ScanpathPlotter()
fixations = pd.read_csv(cwd + '/examples/data/fixations.csv')
image_path = cwd + '/examples/data/example_image.png'
save_directory = cwd + '/examples/data_printer/results/'
fixations = Gazepoint().preprocess_fixations_trial(fixations)
plotter.plot_image(image_path=image_path, fixations=fixations, save_directory=save_directory)





