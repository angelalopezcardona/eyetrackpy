

import cv2
import numpy as np
import pandas as pd
from eyetrackpy.data_processor.models.saliency_generator import SaliencyGenerator


class SaliencyPlotter(SaliencyGenerator):
    def __init__(self):
        pass


    def draw_point(
        image: np.ndarray, points: np.ndarray, line_color: (int, int, int), output_name: str
    ):
        c = 0
        color_step = 255 // points.shape[0]
        color = [0, 0, 255]
        radius = 25
        image = image.copy()
        for x, y in points:
            color[0] += color_step
            color[2] -= color_step
            cv2.circle(image, (x, y), radius, color, -1)
            if c != 0:
                cv2.line(image, (points[c - 1][0], points[c - 1][1]), (x, y), color, 2)
            c += 1
        cv2.imwrite(f"visualization/{output_name}.png", image)
        print(f'points={points}')
        print("#################################")


   