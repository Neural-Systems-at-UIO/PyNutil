import os
from setuptools import setup, find_packages
from pathlib import Path

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

# Get version from environment variable or fallback to default
version = os.environ.get("PACKAGE_VERSION")

setup(
    name="PyNutil",
    version=version,
    packages=find_packages(),
    license="MIT",
    description="a package to quantify atlas registered brain data",
    long_description=long_description,
    url="https://github.com/Neural-Systems-at-UIO/PyNutil",
    long_description_content_type="text/markdown",
    install_requires=[
        "numpy",
        "brainglobe-atlasapi",
        "pandas",
        "requests",
        "pynrrd",
        "xmltodict",
        "opencv-python",
        "scikit-image",
    ],
)
