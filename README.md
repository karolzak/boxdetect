**BoxDetect** is a Python package based on OpenCV which allows you to easily detect rectangular shapes like character or checkbox boxes on scanned forms.

Main purpose of this library is to provide helpful functions for processing document images like bank forms, applications, etc. and extract regions where character boxes or tick/check boxes are present.

![](https://raw.githubusercontent.com/karolzak/boxdetect/master/images/example1.png)


## Getting Started

Checkout the [examples below](#Usage-examples) and 
[get-started.ipynb](https://github.com/karolzak/boxdetect/blob/master/notebooks/get-started.ipynb) notebook which holds end to end examples for using **BoxDetect**.

## Installation

**BoxDetect** can be installed directly from this repo using `pip`:

```
pip install git+https://github.com/karolzak/boxdetect
```

or through [PyPI](https://pypi.org/project/boxdetect/)

```
pip install boxdetect
```

## Usage examples

You can use `BoxDetect` either by leveraging one of the pre-made pipelines or by treating it as a toolbox to compose your own pipelines that fits your needs perfectly.

#### Using existing pipelines:

Start with getting the default config and modifying it for your requirements and data:
```python
from boxdetect import config

file_name = 'form_example1.png'
# important to adjust these values to match the size of boxes on your image
config.min_w, config.max_w = (35,48)
config.min_h, config.max_h = (30,39)
# the more scaling factors the more accurate the results but also it takes more time to processing
# too small scaling factor may cause false positives
# too big scaling factor will take a lot of processing time
config.scaling_factors = [0.7, 1.0]
# w/h ratio range for boxes/rectangles filtering
config.wh_ratio_range = (0.5, 1.5)
# num of iterations when running dilation tranformation (to engance the image)
config.dilation_iterations = 1
```

As a second step simply run:
```python
from boxdetect.pipelines import get_boxes

rects, grouping_rects, image, output_image = get_boxes(
    file_name, config=config, plot=False)
```

Each of the returned elements are rectangular bounding boxes representing grouped character boxes (x, y, w, h)
```python
print(grouping_rects)

OUT:
# (x, y, w, h)
[(276, 276, 1221, 33),
 (324, 466, 430, 33),
 (384, 884, 442, 33),
 (985, 952, 410, 32),
 (779, 1052, 156, 33),
 (253, 1256, 445, 33)]
```

```python
plt.figure(figsize=(20,20))
plt.imshow(output_image)
plt.show()
```

![](https://raw.githubusercontent.com/karolzak/boxdetect/master/images/example1.png)
