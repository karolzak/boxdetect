from setuptools import setup
from setuptools import find_packages

with open("README.md") as f:
    long_description = f.read()

setup(
    name="boxdetect",
    version="0.1.0",
    description="boxdetect is a Python package based on OpenCV which allows you to easily detect rectangular shapes like characters boxes on scanned forms.",
    long_description=long_description,
    long_description_content_type="text/markdown",  # This is important!
    url="http://github.com/karolzak/boxdetect",
    author="Karol Zak",
    author_email="karol.zak@hotmail.com",
    license="MIT",
    packages=find_packages(),
    zip_safe=False,
    install_requires=['opencv-python', 'numpy', 'imutils'],
)
