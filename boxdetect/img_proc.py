import cv2
import numpy as np


def apply_merge_transformations(
        image, kernels, transformations=(cv2.MORPH_OPEN, 1), plot=False):
    """
    Process image by applying morphological transformations using OpenCV.
    It takes in a list of kernels as an input and itterates through that list applying each transormation to input image and merges all the results back together later.

    Args:
        image (numpy.ndarray):
            Input image.
        kernels (list of numpy.ndarray):
            List of kernels to be used for morphological transformations.
        transformations (tuple or list of tuples):
            Tuple or list of tuples (cv2.MORPH_*, num_iterations) that describes, number, order and type of OpenCV Morphological transofrmations to be performed on input image.
            Defaults to `(cv2.MORPH_OPEN, 1)`
        plot (bool, optional): 
            Display intermediate results of transformations. 
            Defaults to False.

    Returns:
        numpy.ndarray:
            Merged results of all the transformations
    """ # NOQA E501
    if type(transformations) is not list:
        transformations = [transformations]

    new_image = np.zeros_like(image)
    for kernel in kernels:
        morphs = image
        for transform, iterations in transformations:
            morphs = cv2.morphologyEx(
                morphs, transform, kernel, iterations=iterations)
        new_image += morphs

    image = new_image
    image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY)[1]

    if plot:
        cv2.imshow("rectangular shape enhanced image", image)
        cv2.waitKey(0)

    return image


def apply_thresholding(image, plot=False):
    """
    Applies thresholding to the image. Sets pixel values to 0 or 255 using OTSU thresholding.

    Args:
        image (numpy.ndarray):
            Input image.
        plot (bool, optional):
            Displays image after thresholding.
            Defaults to False.

    Returns:
        numpy.ndarray:
            Resulting image
    """ # NOQA E501

    otsu = cv2.threshold(
        image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    binary = cv2.threshold(
        image, np.mean(image), 255, cv2.THRESH_BINARY_INV)[1]

    image = otsu + binary

    if plot:
        cv2.imshow("thresholded image", image)
        cv2.waitKey(0)

    return image


def get_rect_kernels(
        width_range, height_range,
        wh_ratio_range=(0.5, 1.1),
        border_thickness=1):
    """
    Returns a list of rectangular kernels for OpenCV morphological transformations.
    It's using `width_range`, `height_range` params to create all the possible combinations of rectangular kernels and performs filtering based on `wh_ratio_range`.

    Args:
        width_range (tuple):
            Min/max width range for rectangular kernels.
            Should be adjusted to the pixel width of boxes to be detected on the image.
        height_range (tuple):
            Min/max height range for rectangular kernels.
            Should be adjusted to the pixel height of boxes to be detected on the image.
        wh_ratio_range (tuple, optional):
            Width / Height ratio range. 
            Defaults to (0.5, 1.1).
        border_thickness (int, optional):
            Rectangles border thickness.
            Defaults to 1.

    Returns:
        list of numpy.ndarray:
            List of rectangular `numpy.ndarray` kernels
    """ # NOQA E501

    kernels = [
        np.pad(
            np.zeros(
                (h - (2 * border_thickness), w - (2 * border_thickness)),
                dtype=np.uint8),
            border_thickness, mode='constant', constant_values=1)
        for w in range(*width_range)
        for h in range(*height_range)
        if w / h >= wh_ratio_range[0] and w / h <= wh_ratio_range[1]
    ]
    return kernels


def get_line_kernels(length, thickness=1):
    """
    Line kernels generator. Creates a list of two `numpy.ndarray` based on length and thickness.
    First kernel represents a vertical line and the second one represents a horizontal line.

    Args:
        length (int):
            Length of the lines to be created.
        thickness (int, optional):
            Thickness of the lines.
            Defaults to 1.

    Returns:
        list of numpy.ndarray:
            List of two kernels representing vertical and horizontal lines.
    """ # NOQA E501
    kernels = [
        np.ones((length, thickness), dtype=np.uint8),
        np.ones((thickness, length), dtype=np.uint8),
    ]
    return kernels


def draw_rects(image, rects, color=(0, 255, 0), thickness=1):
    """
    Draws rectangles (x, y, width, height) on top of the input image.

    Args:
        image (numpy.ndarray):
            Input image.
        rects (list of tuples):
            List of rectangles to be drawn represented as coordinates (x, y, width, height).
        color (tuple, optional):
            Color definition in RGB.
            Defaults to (0, 255, 0).
        thickness (int, optional):
            Thickness of the bounding rectangle to be drawn.
            Defaults to 1.

    Returns:
        numpy.ndarray:
            Output image.
    """ # NOQA E501
    # loop over the contours
    for r in rects:
        x, y, w, h = r
        cv2.rectangle(image, (x, y), (x + w, y + h), color, thickness)
    return image
