


import cv2
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt



class ScanpathPlotter():
    def __init__(
        self,
        x_screen=1280,
        y_screen=1024,
    ):
        self.x_screen = x_screen
        self.y_screen = y_screen
        

    def _plot_image_fixations_v1(self, fixations: pd.DataFrame, img, save=True, image_fixations_path=None):
        color=(0, 0, 255)
        for _, row in fixations.iterrows():
            x = int(round(row["x"], 0))
            y = int(round(row["y"], 0))
            cv2.circle(img, (x, y), 5, color, -1)
        if save:
            #check if the directory exists
            cv2.imwrite(
                image_fixations_path,
                img,
            )
        else:
            cv2.imshow("img", img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    def _rescale_fixations(self,  fixations: pd.DataFrame, width, height):
        fixations["x"] = fixations["x"] * width
        fixations["y"] = fixations["y"] * height
        return fixations
    
    def _plot_image_fixations_v2(self, fixations, img, save=True, image_fixations_path=None):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.imshow(img)

        # Generate colors for points
        num_fixations = len(fixations)
        colors = plt.cm.plasma(np.linspace(0, 1, num_fixations))  # Using 'plasma' colormap

        # Draw fixations and connections
        for i in range(num_fixations):
            x, y = fixations.iloc[i]["x"], fixations.iloc[i]["y"]
            ax.scatter(x, y, color=colors[i], s=100, edgecolors='black', zorder=3)  # Fixation point
            if i > 0:
                x_prev, y_prev = fixations.iloc[i - 1]["x"], fixations.iloc[i - 1]["y"]
                ax.plot([x_prev, x], [y_prev, y], color=colors[i], linewidth=2, zorder=2) 

        ax.axis("off")
        if save:
            plt.savefig(image_fixations_path, bbox_inches="tight", dpi=300)
            plt.close(fig) 
        else:
            plt.show()

    def plot_image(
        self, image_path, fixations:pd.DataFrame=None, save=True, save_directory=None
    ):
        """
        Plot image trial with fixations"""

        if save:
            if not os.path.exists(save_directory):
                os.makedirs(save_directory)
            image_name_save = image_path.split('/')[-1]
            image_fixations_path = save_directory + '/' + 'scanpath_' + image_name_save
        else:
            image_fixations_path = None
        img = cv2.imread(image_path)
        height, width, _ = img.shape
        fixations = self._rescale_fixations(fixations, width, height)
        # img = cv2.resize(img, (self.x_screen, self.y_screen))
        # self._plot_image_fixations_v1(fixations_sample, img, save=save, image_fixations_path=image_fixations_path)
        self._plot_image_fixations_v2(fixations, img, save=save, image_fixations_path=image_fixations_path)

        return True


    def _plot_image_fixations(self, fixations_sample, img, color=(0, 0, 255)):
        """
        Plot image trial with fixations of trial"""
        for _, row in fixations_sample.iterrows():
            x = int(round(row["x"], 0))
            y = int(round(row["y"], 0))
            cv2.circle(img, (x, y), 5, color, -1)

        return True


    
