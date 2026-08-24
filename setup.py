from setuptools import setup, find_packages

setup(
    name="monocular_depth_estimation",
    version="1.0.0",
    description="Monocular Depth Map Prediction and Spatial 4-Quadrant Risk Analysis System",
    author="Bhanu Vignesh Naidu Ganeshna",
    packages=find_packages(),
    install_requires=[
        "torch>=1.10.0",
        "torchvision>=0.11.0",
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "matplotlib>=3.4.0"
    ],
    entry_points={
        'console_scripts': [
            'depth-estimate=main:main',
        ],
    },
    python_requires='>=3.8',
)
