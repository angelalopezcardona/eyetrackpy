from setuptools import setup, find_packages
from setuptools.command.install import install
import os
import requests
# Custom install command to download weights
class CustomInstallCommand(install):
    def run(self):
        # Run the standard install process
        install.run(self)
        
        # Download weights and save to the desired location
        url = "https://drive.google.com/uc?id=1CTiali54Q7zsT25ciY0y0sIIf2jZVbZG&export=download"
        local_path = "eyetrackpy/data_generator/fixations_predictor_trained_2/FPmodels/model.pth"

        # Ensure the directory exists
        directory = os.path.dirname(local_path)
        if not os.path.exists(directory):
            print(f"Creating directory: {directory}")
            os.makedirs(directory)

        # Download and save the file
        print("Downloading weights...")
        response = requests.get(url)
        if response.status_code == 200:
            with open(local_path, "wb") as f:
                f.write(response.content)
            print(f"File saved to: {local_path}")
        else:
            print(f"Failed to download file. Status code: {response.status_code}")


setup(
    name="eyetrackpy",
    version="0.1",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        # List your dependencies here
        'numpy',
        'pandas',
        "requests",
        # Add other dependencies as needed
    ],
     cmdclass={
        'install': CustomInstallCommand,  # Use the custom install command
    },
    # other metadata like author, description, license, etc.
    author="Angela Lopez",
    description="A Python package for eye tracking analysis",
)

