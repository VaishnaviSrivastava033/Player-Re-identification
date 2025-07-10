from setuptools import setup, find_packages

setup(
    name="player_reidentification",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        'numpy',
        'opencv-python',
        'ultralytics',
        'scikit-learn'
    ],
)