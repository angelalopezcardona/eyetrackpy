from setuptools import setup, find_packages

setup(
    name="eyetrackpy",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        # List your dependencies here
        'numpy',
        'pandas',
        # Add other dependencies as needed
    ],
    # other metadata like author, description, license, etc.
    author="Angela Lopez",
    description="A Python package for eye tracking analysis",
)
