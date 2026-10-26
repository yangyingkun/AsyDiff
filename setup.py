from setuptools import find_packages, setup

setup(
    name="AsyDiff",
    version="0.1.0",
    description="Multi-dataset satellite-prompt diffusion for pansharpening",
    packages=find_packages(include=["ldm", "ldm.*"]),
    python_requires=">=3.8",
)
