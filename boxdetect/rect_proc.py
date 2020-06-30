import cv2
import numpy as np
import imutils


def group_countours(cnts, epsilon=0.1):
    """
    Merge multiple contours into a single bounding rectangle using `epsilon`

    Args:
        cnts (list of contours):
            List of countours to be merged.
        epsilon (float, optional):
            Value deciding if two contours are close enough to each other to merge them together.
            Defaults to 0.1.

    Returns:
        list of rectangles:
            List of bounding rectangles after merging the overlapping contours.
    """ # NOQA E501
    rects = [get_bounding_rect(c)[:4] for c in cnts]
    # we need to duplicate all the rects for grouping below to work
    rects += rects
    rects, weights = cv2.groupRectangles(rects, 1, epsilon)
    return rects


def get_bounding_rect(c):
    """
    Takes in a single contour coordinates and returns bounding rectangle coordinates.

    Args:
        c (numpy.ndarray):
            `numpy.ndarray` representing contour object.

    Returns:
        x, y, w, h, is_rect:
            Returns bounding rectangle over given contour and `is_rect` bool flag.
    """ # NOQA E501
    # epsilon = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 2, True)
    (x, y, w, h) = cv2.boundingRect(approx)
    if len(approx) == 4:
        return x, y, w, h, True
    return x, y, w, h, False


def filter_contours_by_size_range(cnts, width_range=None, height_range=None):
    """  # NOQA E501
    Filters the input list of contours and removes all the contours for which size is outside of provided `width_range` and/or `height_range`.

    Args:
        cnts (list of numpy.ndarray):
            List of contours (numpy.ndarray)
        width_range (tuple of ints, optional):
            Tuple of integers like: (min_width, max_width).
            If set to `None` - width won't be taken into account.
            Defaults to None.
        height_range (tuple of ints, optional):
            Tuple of integers like: (min_height, max_height).
            If set to `None` - height won't be taken into account.
            Defaults to None.

    Returns:
        list of numpy.ndarray:
            List of filtered contours (numpy.ndarray)
    """
    return [
        c for c in cnts
        if size_in_range(c, width_range, height_range)
    ]


def size_in_range(c, width_range=None, height_range=None):
    """  # NOQA E501
    Returns `bool` indicating if given contour object is within a specific range of height and/or width.

    Args:
        c (numpy.ndarray):
            Contour object.
        width_range (tuple of ints, optional):
            Tuple of integers like: (min_width, max_width).
            If set to `None` - width won't be taken into account.
            Defaults to None.
        height_range (tuple of ints, optional):
            Tuple of integers like: (min_height, max_height).
            If set to `None` - height won't be taken into account.
            Defaults to None.

    Returns:
        bool:
            `True` if contour size is in provided ranges of width and height.
            `False` if it's not.
    """
    (x, y, w, h, is_rect) = get_bounding_rect(c)
    if width_range:
        if w >= width_range[0] and w <= width_range[1]:
            pass
        else:
            return False
    if height_range:
        if h >= height_range[0] and h <= height_range[1]:
            pass
        else:
            return False
    return True


def wh_ratio_in_range(c, wh_ratio_range, tolerance=0.05):
    """
    Performs 2 checks:
    1. Using `get_bounding_box` function it retrieves bounding box coords and `is_rect` bool flag which defines if given contour object is in fact a rectangle.
    2. Calculates rectangles `width / height` ratio and checks if it's inside given range (`wh_ratio_range`).
    Returns `True` if input contour object is a rectangle and it's w/h ratio is inside the given range otherwise returns `False`.

    Args:
        c (numpy.ndarray):
            Contour object. More about contours in OpenCV: https://docs.opencv.org/3.4/d4/d73/tutorial_py_contours_begin.html
        wh_ratio_range (tuple):
            Tuple of ints like `(0.5, 1.5)` representing minimum and maximum value of rectangles width / height ratio.
            If given rectangle won't make it inside the range, function will return `False`.
        tolerance (float):
            Expands `wh_ratio_range` upper and lower boundries by tolerance value.
            Defaults to `0.05`.

    Returns:
        bool:
            Returns `True` if input contour object is a rectangle and it's w/h ratio is inside the given range otherwise returns `False`.
    """ # NOQA E501
    (x, y, w, h, is_rect) = get_bounding_rect(c)
    ar = w / float(h)
    if is_rect and ar >= wh_ratio_range[0] * 1 - tolerance and ar <= wh_ratio_range[1] * 1 + tolerance:  # NOQA E501
        return True
    return False


def filter_contours_by_wh_ratio(cnts, wh_ratio_range):
    """
    Performs filtering based on width / height ratio of contours bounding rectangles.

    Args:
        cnts (list of contour objects):
            List of `numpy.ndarray` objects representing contours from OpenCV.
        wh_ratio_range (tuple):
            Tuple of ints like `(0.5, 1.5)` representing minimum and maximum value of rectangles width / height ratio.

    Returns:
        list of contour objects:
            List of contour objects after filtering.
    """ # NOQA E501
    return [
        c for c in cnts
        if wh_ratio_in_range(c, wh_ratio_range)
    ]


def get_contours(image):
    """
    For input image use `cv2.findContours` function to get a list of contour objects.

    Args:
        image (numpy.ndarray):
            Input image.

    Returns:
        list of contour objects:
            List of contour objects extracted from the image.
    """ # NOQA E501
    # find contours in the thresholded image
    cnts = cv2.findContours(
        image.copy(),
        cv2.RETR_LIST,
        cv2.CHAIN_APPROX_SIMPLE
    )
    cnts = imutils.grab_contours(cnts)
    return cnts


def filter_contours_by_area_size(cnts, area_range):
    """
    Performs filtering based on contours area size (width * height).    

    Args:
        cnts (list of contour objects):
            List of `numpy.ndarray` objects representing contours from OpenCV.
        area_range (tuple):
            Tuple of ints like `(1000, 2000)` which stands for minimum and maximum range of values for area size.

    Returns:
        list of contour objects:
            List of contour objects after filtering.
    """ # NOQA E501
    cnts_filtered = []
    for c in cnts:
        area = cv2.contourArea(c)
        if area >= area_range[0] and area <= area_range[1]:
            cnts_filtered.append(c)
    return cnts_filtered


def rescale_contours(cnts, ratio):
    """
    Rescales contours based on scalling `ratio`.

    Args:
        cnts (list of contour objects):
            List of `numpy.ndarray` objects representing contours from OpenCV.
        ratio (float):
            Float value used to rescale all the points in contours list.
    Returns:
        list of contour objects:
            List of contour objects after rescaling.
    """
    cnts_rescaled = []
    for c in cnts:
        c = c.astype("float")
        c *= ratio
        c = c.astype("int")
        cnts_rescaled.append(c)
    return cnts_rescaled


def get_grouping_rectangles(rect_groups):
    """
    For a list of groups of boxes it returns a list of bounding rectangles calculated for each each group of boxes.

    Args:
        rect_groups (list of lists of rectangles):
            Example structure: `[ [[x,y,w,h],[x,y,w,h]], [[x,y,w,h],[x,y,w,h]] ]`

    Returns:
        list of bounding rectangles:
            Returns a list o bounding rectangles generated based on groups of boxes (rectangles)
    """ # NOQA E501
    return [
        cv2.boundingRect(np.asarray(
            [
                get_point(rect)
                for rect in group
                for get_point in (
                    lambda rect: (rect[0], rect[1]),
                    lambda rect: (rect[0] + rect[2], rect[1] + rect[3]))
            ]))
        for group in rect_groups
    ]


def get_groups_from_groups(
        list_of_groups,
        max_distance,
        group_size_range,
        grouping_mode='horizontal'):
    """
    Helper function that iterates through a list of groups of rectangles
    and within those groups it will try to find new groups based on input parameters

    Args:
        list_of_groups (list of rectangles lists):
            Example structure: `[ [[x,y,w,h],[x,y,w,h]], [[x,y,w,h],[x,y,w,h]] ]`
        max_distance (int):
            Max distance between two boxes (rectangles) to be considered the same group
        group_size_range (tuple):
            Tuple of ints like `(1, 100)` defining the minimum and maximum size of the groups to be created.
        grouping_mode (str, optional):
            Either 'horizontal' or 'vertical'.
            Defaults to 'horizontal'.

    Returns:
        list of rectangles lists:
            Example output structure: `[ [[x,y,w,h],[x,y,w,h]], [[x,y,w,h],[x,y,w,h]] ]`
    """ # NOQA E501
    rect_groups = [
        rect_group
        for rect_group in(
            group_rects(
                np.asarray(group), max_distance=max_distance,
                group_size_range=group_size_range, grouping_mode=grouping_mode)
            for group in list_of_groups)
        if rect_group != []
    ]
    rect_groups = [
        horizontal_group
        for vertical_group in rect_groups
        for horizontal_group in vertical_group
    ]
    return rect_groups


def group_rects(
        rects,
        max_distance,
        group_size_range=(1, 1000),
        grouping_mode='vertical'):
    """
    Grouping function for rectangles in a list.

    Args:
        rects (list of rectangles):
            List of rectangles (x,y,width,height).
        max_distance (int):
            Max distance between two boxes (rectangles) to be considered the same group
        group_size_range (tuple, optional):
            Tuple of ints like `(1, 100)` defining the minimum and maximum size of the groups to be created.
            Defaults to (1, 1000).
        grouping_mode (str, optional):
            Either 'horizontal' or 'vertical'.
            Defaults to 'vertical'.

    Returns:
        list of rectangles lists:
            Example output structure: `[ [[x,y,w,h],[x,y,w,h]], [[x,y,w,h],[x,y,w,h]] ]`
    """ # NOQA E501
    grouping_mode = grouping_mode.lower()
    assert(grouping_mode in ['vertical', 'horizontal'])
    if grouping_mode == 'vertical':
        m, n = (1, 3)
    elif grouping_mode == 'horizontal':
        m, n = (0, 2)

    if len(rects) == 0:
        return []

    rects_sorted = rects[np.argsort(rects[:, m])]
    new_groups = []
    temp_group = []
    temp_group.append(rects_sorted[0])
    for i in range(1, len(rects_sorted)):
        rect1 = rects_sorted[i-1]
        x1 = rect1[m] + int(rect1[n] / 2)

        rect2 = rects_sorted[i]
        x2 = rect2[m] + int(rect1[n] / 2)

        distance = abs(x2 - x1)
        if distance <= max_distance:
            temp_group.append(rect2)
        else:
            new_groups.append(temp_group)
            temp_group = []
            temp_group.append(rect2)
    new_groups.append(temp_group)

    group_range = list(range(group_size_range[0], group_size_range[1]+1))

    new_groups = [
        group
        for group in new_groups
        if group and len(group) in group_range]
    return new_groups
