import numpy as np
import cv2
import os
import pathlib
import sys
# Get the current working directory
cwd = os.getcwd()
sys.path.append(cwd)


import matplotlib.pyplot as plt
from eyetrackpy.data_generator.fixations_predictor_mdsem.multiduration_models import md_sem
from eyetrackpy.data_generator.fixations_predictor_mdsem.sal_imp_utilities import preprocess_images
from PIL import Image

from eyetrackpy.data_generator.fixations_predictor_mdsem.losses_keras2 import loss_wrapper, kl_time, cc_time, nss_time, cc_match
from eyetrackpy.data_generator.fixations_predictor.models.model_manager import download_model
MODELS = {
    'md_sem': (md_sem, 'singlestream'),
}

LOSSES = {
    'kl': (kl_time, 'heatmap'),
    'cc': (cc_time, 'heatmap'),
    'nss': (nss_time, 'fixmap'),
    'ccmatch': (cc_match, 'heatmap')
}

def get_model_by_name(name): 
    """ Returns a model and a string indicating its mode of use."""
    if name not in MODELS: 
        allowed_models = list(MODELS.keys())
        raise RuntimeError("Model %s is not recognized. Please choose one of: %s" % (name, ",".join(allowed_models)))
    else: 
        return MODELS[name]

# Function to preprocess a single image
 # Add batch dimension

class FixationsPredictor:
    def __init__(self):
        model_name = "md_sem"
        # Input image size and model parameters
        self.model_inp_size = (240, 320)  # Input size for the model
        self.model_out_size = (480, 640)  # Output size for predictions
        self.times = [500, 3000, 5000]
        n_timesteps = len(self.times)
        losses = {
            'kl': 10,
            'cc': -5,
            'nss': -1,
            'ccmatch': 3
        }

        model_params = {
            'input_shape': self.model_inp_size + (3,),
            'n_outs': len(losses),
            'nb_timestep': n_timesteps
        }
        # Load the trained model
        model_func, mode = get_model_by_name(model_name)
        model = model_func(**model_params)
        cwd = os.getcwd()
        trained_weights_path = os.path.join(
            cwd,
            "eyetrackpy",
            "data_generator",
            "fixations_predictor_mdsem",
            "mdsem_codecharts0_cameraready_weights.hdf5",
        )
        if not os.path.isfile(trained_weights_path):
            download_model('mdsem')

        model.load_weights(trained_weights_path)
        self.model = model

    def preprocess_image(self, image_path):
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # plt.imshow(image)
        # plt.axis('off')  # Hide axes
        # plt.show()
        # image = cv2.resize(image, model_inp_size)
        image = preprocess_images([image_path], self.model_inp_size[0], self.model_inp_size[1])  # Apply preprocessing defined in sal_imp_utilities
        return image 
    
    def predict(self, image_path, save_path):
            # Inference on a new image
        input_image = self.preprocess_image(image_path)
        predictions = self.model.predict(input_image)
        self.postprocess_predictions(image_path, predictions, save_path)
        return predictions
    
    def transparent_cmap(self, cmap, N=255):
        "Copy colormap and set alpha values"
        mycmap = cmap
        mycmap._init()
        mycmap._lut[:,-1] = np.linspace(0.5, 0.8, N+4)
        #mycmap._lut[:,-1] = np.linspace(0, 0.8, N+4) # full transparent
        return mycmap
    
    def postprocess_predictions(self, image_path, predictions, save_path):
        shape_r,shape_c= self.model_inp_size
        ntimes = len(self.times)
        #----------------------------------------
        # postprocess
        path_sal = save_path + '/salmaps/'
        path_sal_vis = save_path + '/salmaps_vis/'
        #check if path_sal exists
        if not os.path.exists(path_sal):
            os.makedirs(path_sal)
        if not os.path.exists(path_sal_vis):
            os.makedirs(path_sal_vis)

        mycmap = self.transparent_cmap(plt.cm.jet)

        
        input_image = self.preprocess_image(image_path)

        preds_mdsem = predictions

        im = Image.open(image_path)
        width, height = im.size
        i = 0

        for t in range(ntimes):
            plt.imshow(im)
            a = preds_mdsem[0][i,t,:,:,0]

            if width*shape_r > height*shape_c:
                crop_pred = preds_mdsem[0][i,t,:,:,0][int(shape_r - height/width*shape_c):int(shape_r + height/width*shape_c),:]
            else:
                crop_pred = preds_mdsem[0][i,t,:,:,0][:,int(shape_c - width/height*shape_r):int(shape_c + width/height*shape_r)]

            pred = Image.fromarray(np.uint8(crop_pred * 255)).resize(im.size)
            plt.imshow(pred, alpha=0.5, cmap=mycmap)
            plt.axis('off')
            file_name = image_path.split('/')[-1].replace('.',f"_{self.times[t]}.")
            pred.save(path_sal + file_name)
            plt.savefig(path_sal_vis + file_name, bbox_inches='tight', pad_inches=0, dpi=300)
            plt.close()

        

