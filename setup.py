import setuptools
import dosed

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="dosed",
    version=dosed.__version__,
    author="Dreem",
    author_email=" ",
    description="Implementation of DOSED algorithm for event detection",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Dreem-Organization/dosed/",
    packages=["dosed"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)