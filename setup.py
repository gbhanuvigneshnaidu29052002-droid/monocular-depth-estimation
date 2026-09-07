from setuptools import setup, find_packages

setup(
    name="monocular_depth_estimation",
    version="1.0.0",
    description="Monocular Depth Map Prediction and Spatial 4-Quadrant Risk Analysis System",
    author="Bhanu Vignesh Naidu Ganeshna",
    url="https://github.com/gbhanuvigneshnaidu29052002-droid/monocular-depth-estimation",
    packages=find_packages(),
    install_requires=[
        "torch>=1.10.0",
        "torchvision>=0.11.0",
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "scikit-learn>=1.0.0",
        "Pillow>=8.0.0",
        "opencv-python>=4.5.0",
        "matplotlib>=3.4.0",
        "seaborn>=0.11.0"
    ],
    entry_points={
        'console_scripts': [
            'depth-estimate=main:main',
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
    ],
    python_requires='>=3.8',
)
