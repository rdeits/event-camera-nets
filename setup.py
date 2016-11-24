from setuptools import setup, find_packages

import os

setup(
      name="eventcnn",
      packages=find_packages(exclude=["data"]),
)

