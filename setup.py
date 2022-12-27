from setuptools import setup
from setuptools import find_packages

with open("README.md") as f:
    long_description = f.read()

setup(
    name="boxdetect",
    version="1.0.1",
    description="boxdetect is a Python package based on OpenCV which allows you to easily detect rectangular shapes like characters boxes on scanned forms.",
    long_description=long_description,
    long_description_content_type="text/markdown",  # This is important!
    url="http://github.com/karolzak/boxdetect",
    author="Karol Zak",
    author_email="karol.zak@hotmail.com",
    license="MIT",
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Topic :: Software Development :: Libraries',
        'Topic :: Software Development :: Libraries :: Python Modules'
    ],
    packages=find_packages(),
    zip_safe=False,
    install_requires=[
        'opencv-python', 'numpy', 'imutils', 'pyyaml', 'sklearn'],
)
