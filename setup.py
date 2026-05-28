from setuptools import find_packages, setup

CONVERT_REQUIRES = [
    "MNN>=3.5.0",
    "av>=14",
    "librosa>=0.10.2",
    "numpy>=1.26",
    "onnx>=1.16",
    "onnxruntime>=1.18",
    "pyyaml>=6.0.1",
    "torch>=2.7.1",
]

setup(
    name="pymss",
    version="2.0.2",
    packages=find_packages(),
    description="MNN conversion support package for music source separation models.",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/pymss-project/pymss-mnn",
    author="KitsuneX07",
    maintainer="baicai1145",
    author_email="ghast1085654218@163.com",
    maintainer_email="3423714059@qq.com",
    license="MIT",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Topic :: Multimedia :: Sound/Audio",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Operating System :: OS Independent",
    ],
    keywords="music source separation, audio separation, music processing, machine learning, audio",
    python_requires=">=3.10",
    package_data={
        "pymss": ["resources/model_catalog.json", "resources/vr_modelparams/*.json"],
    },
    install_requires=[
        "tqdm>=4.60",
    ],
    extras_require={
        "convert": CONVERT_REQUIRES,
    },
    project_urls={
        "Bug Tracker": "https://github.com/pymss-project/pymss-mnn/issues",
        "Source Code": "https://github.com/pymss-project/pymss-mnn",
        "Documentation": "https://github.com/pymss-project/pymss-mnn/blob/main/README.md",
    },
    entry_points={
        "console_scripts": [
            "pymss=pymss.cli:main",
        ],
    },
)
