# import OS module
import argparse

# Get the list of all files and directories
import pathlib
import sys

sys.path.append("../..")
path = str(
    pathlib.Path(__file__)
    .parent.resolve()
    .parent.resolve()
    .parent.resolve()
    .parent.resolve()
    .parent.resolve()
)
sys.path.append(path)
path = str(
    pathlib.Path(__file__)
    .parent.resolve()
    .parent.resolve()
    .parent.resolve()
    .parent.resolve()
)
sys.path.append(path)
path = str(pathlib.Path(__file__).parent.resolve().parent.resolve().parent.resolve())
sys.path.append(path)
path = str(pathlib.Path(__file__).parent.resolve().parent.resolve())
sys.path.append(path)
path = str(pathlib.Path(__file__).parent.resolve())
sys.path.append(path)

from eyetrackpy.data_processor.models.deprecated_eye_tracking_analyzer import EyeTrackingAnalyser
# create class to analyse eye tracking load_dataset

import os


os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--user", default=1, help="user")
    parser.add_argument("--session", default=1, help="session")
    parser.add_argument("--trial", default=None)

    args = parser.parse_args()
    user = args.user
    session = args.session
    trial = args.trial
    show = True
    # -------------------- read data from text and join with eye tracking tokens --------------------
    # trials = [4.1, 4.3, 6.1, 6.3, 7.1, 7.3, 8.1, 8.2, 8888.1, 8888.2, 9999.1, 9999.2]
    # analyzer = EyeTrackingDataText(user, session, trial, show)
    # analyzer.join_words_vertices()
    # ---------------------------------------------------------------

    # -------------------- to plot image trials --------------------
    # analyzer = EyeTrackingDataImage(user, session, trials, show)
    # trials = [4.1, 4.3]
    # trials = [4.1, 4.3, 6.1, 6.3, 7.1, 7.3, 8.1, 8.2, 8888.1, 8888.2, 9999.1, 9999.2]
    # for trial in trials:
    # analyzer.plot_image_trial(trial, fixations=True, coordinates=True, calibrate=False)
    # analyzer.plot_image_trial(trial, fixations=True, coordinates=True, calibrate=True)
    # ---------------------------------------------------------------

    # -------------------- asign fixations to words,  print them on the image--------------------
    users = [1]
    sessions = [2]
    # EyeTrackingAnalyser.compute_images_and_show(users, sessions)
    # ----------------------------------------------------------------

    # -----------------------------compute_images_and_save----------------------------------
    users = [1, 2, 3, 4, 5, 6, 7, 8]
    sessions = [2]
    # EyeTrackingAnalyser.compute_images_and_save(users, sessions)
    # ----------------------------------------------------------------

    # ------------------------------compute_fixations_and_save----------------------------------
    # users= [1]
    # sessions= [2]
    # EyeTrackingAnalyser.compute_fixations_and_save(users, sessions)
    # ----------------------------------------------------------------

    # ------------------------------compute_entropy----------------------------------
    users = [1, 2, 3, 4, 5, 6, 7, 8]
    sessions = [1, 2]
    fixations_entropy = EyeTrackingAnalyser.compute_entropy(users, sessions)
    fixations_entropy.to_csv("fixations_entropy.csv", index=False)
    # ----------------------------------------------------------------

    # -------------------- compute trial coordenates ans save them --------------------
    # analyzer = EyeTrackingDataImage(user=user, session=session, show=show)
    # we obtain all trials of this user and session
    # coordinates = analyzer.read_image_trials(analyzer.trials)
    # analyzer.save_coordinates(analyzer.trials, coordinates)
    # ---------------------------------------------------------------

    # -------------------- asign fixations to words, compute general fetures, save then on a CSV--------------------
    users = [1]
    sessions = [2]
    # EyeTrackingAnalyser.compute_general_features(users, sessions)

    # ---------------------------------------------------------------
