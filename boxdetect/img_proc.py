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

    if plot:  # pragma: no cover
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

    if plot:  # pragma: no cover
        cv2.imshow("thresholded image", image)
        cv2.waitKey(0)

    return image


def get_rect_kernels(
        width_range, height_range,
        wh_ratio_range=None,
        border_thickness=1,
        tolerance=0.02):
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
            Width / Height ratio range. If None it will create ratio range based on width and height.
            Defaults to None.
        border_thickness (int, optional):
            Rectangles border thickness.
            Defaults to 1.
        tolerance (float):
            Expands `wh_ratio_range` upper and lower boundries by tolerance value.
            Defaults to `0.05`.

    Returns:
        list of numpy.ndarray:
            List of rectangular `numpy.ndarray` kernels
    """ # NOQA E501

    kernels = [
        np.pad(
            np.zeros(
                (h, w),
                # (h - (2 * border_thickness), w - (2 * border_thickness)),
                dtype=np.uint8),
            border_thickness, mode='constant', constant_values=1)
        for w in range(
            int((1 - tolerance) * width_range[0]),
            int((1 + tolerance) * width_range[1]))
        for h in range(
            int((1 - tolerance) * height_range[0]),
            int((1 + tolerance) * height_range[1]))
        if w / h >= wh_ratio_range[0] and w / h <= wh_ratio_range[1]
    ]
    return kernels


def get_line_kernels(horizontal_length, vertical_length, thickness=1):
    """
    Line kernels generator. Creates a list of two `numpy.ndarray` based on length and thickness.
    First kernel represents a vertical line and the second one represents a horizontal line.

    Args:
        horizontal_length (int):
            Length of the horizontal line kernel.
        vertical_length (int):
            Length of the vertical line kernel.
        thickness (int, optional):
            Thickness of the lines.
            Defaults to 1.

    Returns:
        list of numpy.ndarray:
            List of two kernels representing vertical and horizontal lines.
    """ # NOQA E501
    kernels = [
        np.ones((vertical_length, thickness), dtype=np.uint8),
        np.ones((thickness, horizontal_length), dtype=np.uint8),
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


def get_checkbox_crop(img, rect, border_crop_factor=0.15):
    """
    Takes in image as `numpy.ndarray` and rectangle to be cropped out and returns the cropped out image.

    Args:
        img (numpy.ndarray):
            Image as numpy array.
        rect (list, tuple or array):
            Rectangle from OpenCV with following values: `(x, y, width, height)`
        border_crop_factor (float, optional):
            Defines how much from the image border should be removed during cropping.
            This is used to remove any leftovers of checkbox frame.
            Defaults to 0.15.

    Returns:
        numpy.ndarray:
            Image crop as numpy array.
    """ # NOQA E501
    # collect base parameters of the crop
    width = rect[2]
    height = rect[3]
    x1 = rect[0]
    y1 = rect[1]
    x2 = x1 + width
    y2 = y1 + height

    # calculate horizontal and vertical border to be cropped
    w_pad = int(width * border_crop_factor)
    h_pad = int(height * border_crop_factor)

    # crop checkbox area from original image
    im_crop = img[y1 + h_pad:y2 - h_pad, x1 + w_pad:x2 - w_pad]

    return im_crop


def contains_pixels(img, px_threshold=0.1, verbose=False):
    """
    Counts white pixels inside the input image and based on the `px_threshold` it estimates if there's enough white pixels present in the image. 
    As this function sums pixel values you need to make sure that what you're passing as an input image is well preprocessed for that.

    Args:
        img (numpy.ndarray):
            Image as numpy array.
        px_threshold (float, optional):
            This is the threshold used when estimating if pixels are present inside the checkbox.
            Defaults to 0.1.
        verbose (bool, optional):
            Defines if messages should be printed or not.
            Defaults to False.

    Returns:
        bool:
            True - if input image contains enough white pixels
            False - if input image does not contain enough white pixels
    """ # NOQA E501
    # calculate maximum pixel values capacity
    all_px_count = img.shape[0] * img.shape[1]
    # all_px = img.shape[0] * img.shape[1] * img.max()
    # divide sum of pixel values for the image by maximum pixel values capacity
    # return True if calculated value is above the px_threshold,
    # which means that enough pixels where detected in the image
    # else it returns False
    nonzero_px_count = np.count_nonzero(img)
    if verbose:  # pragma: no cover
        print("----------------------------------")
        print("nonzero_px_count: ", nonzero_px_count)
        print("all_px_count: ", all_px_count)
        print(
            "nonzero_px_count / all_px_count = ",
            nonzero_px_count / all_px_count)
        print("----------------------------------")
    return True if nonzero_px_count / all_px_count >= px_threshold else False
    # return True if np.sum(img) / all_px >= px_threshold and all_px != 0 else False  # NOQA E501


def get_image(img):
    """
    Helper function to take image either as `numpy.ndarray` or `str` and always return image as `numpy.ndarray`.

    Args:
        img (numpy.ndarray or str):
            Image as numpy array or string file path.

    Returns:
        numpy.ndarray:
            Image as `numpy.ndarray` with `dtype=np.uint8`
    """ # NOQA E501
    assert(type(img) in [np.ndarray, str])
    if type(img) is np.ndarray:
        image_org = img.copy()
        image_org = image_org.astype(np.uint8)
    elif type(img) is str:
        print("Processing file: ", img)
        image_org = cv2.imread(img)
    return image_org
