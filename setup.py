from setuptools import setup, find_packages

setup(
    name="beacon",
    version="0.1.0",
    description="Medical Image Analysis Framework with Interpretability",
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_packages(),
    install_requires=[
        "torch>=1.9.0",
        "torchvision>=0.10.0",
        "numpy>=1.19.0",
        "pandas>=1.3.0",
        "scikit-learn>=0.24.0",
        "matplotlib>=3.4.0",
        "seaborn>=0.11.0",
        "pillow>=8.3.0",
        "opencv-python>=4.5.0",
        "captum>=0.4.0",
        "click>=8.0.0",
        "tqdm>=4.62.0",
        "pyyaml>=5.4.0",
        "jsonschema>=3.2.0"
    ],
    entry_points={
        'console_scripts': [
            'beacon=beacon.cli.main:cli',
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    include_package_data=True,
    zip_safe=False
) 