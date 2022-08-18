import pathlib
from setuptools import setup

# The directory containing this file
HERE = pathlib.Path(__file__).parent

# The text of the README file
README = (HERE / "README.md").read_text()

setup(
    name="wenda_gpu",
    version="0.7.5",
    description="Fast domain adaptation for genomic data",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/greenelab/wenda_gpu",
    author="Ariel Hippen",
    author_email="ariel.hippen@gmail.com",
    license="BSD 3-Clause",
    packages=["wenda_gpu"],
    python_requires=">=3.7",
    install_requires=[
        "pandas",
        "numpy",
        "torchvision",
        "scipy",
        "matplotlib",
        "gpytorch >= 1.5.0",
        "glmnet >= 2.0"
    ]
)
