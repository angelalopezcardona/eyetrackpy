
import os
import argparse
import json
import sys
cwd = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(cwd)
from eyetrackpy.data_generator.visalformer.saliency_predictor import VisalformerSaliencyPredictor
from eyetrackpy.data_generator.visalformer.dataset import DatasetLoader


def evaluation(batch_size=1):
    #load model
    image_path = cwd + '/examples/data/example_image.png'
    save_path = cwd + '/examples/data_generator/results/'
    data_for_loader = []
    data_for_loader.append(('example', image_path, 'What do you see in the image?'))
    dataset_loader = DatasetLoader().create_dataloader(data_for_loader, batch_size=batch_size)
    
    model = VisalformerSaliencyPredictor()
    model.predict(dataset_loader, save_path=save_path + 'visalformer_saliency')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=1)
    args = vars(parser.parse_args())

    evaluation(batch_size = args['batch_size'])
