import cv2
import imutils
import numpy as np

from . import rect_proc, img_proc, config


def get_checkboxes(
        img, cfg: config.PipelinesCfg,
        px_threshold=0.1, plot=False, verbose=False):
    """    
    Pipeline function to extract checkboxes locations from input image along with an estimation if pixels are present inside that checkbox.
    Short description of pipeline steps:
    - read image from a path or from `numpy.ndarray`
    - run an image processing iteration for every value provided in `cfg.scaling_factors` list:
        - resize image based on `scaling_factor`
        - try convert to grayscale
        - apply otsu thresholding
        - run dilation based on `cfg` params. If `cfg.dilation_iterations=0` this step will be skipped
        - process image with morphological transformations to extract rectangular shapes based on `cfg` params
        - get contours from transformed images and filter them based on area size and `cfg.wh_ratio_range`
    - aggregate contours from all the iterations and merge overlapping countours
    - convert contours to rectangles `(x, y, width, height)`
    - group rectangles first vertically and then horizontally based on `cfg.vertical_max_distance` and `cfg.horizontal_max_distance_multiplier`
    - extract groups of boxes which have only a single box inside (checkboxes)
    - run estimation function to determine if checkbox contains pixels (naive approach for checking if it's checked)
    - return an array of arrays with following information for each detected checkbox: `[checkbox_coords, contains_pixels, cropped_checkbox]`                    
        - checkbox_coords - `numpy.ndarray` rectangle (x,y,width,height)
        - contains_pixels - `bool`, True/False
        - cropped_checkbox - `numpy.ndarray` of cropped checkbox image

    Args:
        img (str or numpy.ndarray):
            Input image. Can be passed in either as
            `string` (filepath) or as `numpy.ndarray` represting an image
        cfg (boxdetect.config object):
            Object holding all the necessary settings to run this pipeline
        px_threshold (float, optional):
            This is the threshold used when estimating if pixels are present inside the checkbox.
            Defaults to 0.1.
        plot (bool, optional):
            Display different stages of image being processed.
            Defaults to False.
        verbose (bool, optional):
            Defines if messages should be printed or not.
            Defaults to False.

    Returns:
        numpy.ndarray:
            Return an array of arrays with following values: `[checkbox_coords, contains_pixels, cropped_checkbox]`                    
            - checkbox_coords - `numpy.ndarray` rectangle (x,y,width,height)
            - contains_pixels - `bool`, True/False
            - cropped_checkbox - `numpy.ndarray` of cropped checkbox image
    """ # NOQA E501
    # st group_size_range to (1, 1) to focus on single box groups only (checkboxes) # NOQA E501
    cfg.group_size_range = (1, 1)

    # get the image from str or numpy.ndarray
    img = img_proc.get_image(img)

    # try converting to grayscale
    try:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    except Exception:
        pass
        # print("WARNING: Failed to convert to grayscale... skipping")

    # run get_boxes function
    rects, grouping_rects, image, output_image = get_boxes(
        img, cfg=cfg, plot=plot)

    # sets all the pixel values to either 0 or 255
    # this function also inverts colors:
    # - black pixels will become the background
    # - white pixels will become the foreground
    img = img_proc.apply_thresholding(img, plot)

    # return an array of arrays containing a set of values for each checkbox:
    # [checkbox_coords, contains_pixels, cropped_checkbox]
    # - checkbox_coords - rectangle (x,y,width,height)
    # - contains_pixels - bool, True/False
    # - cropped_checkbox - numpy.ndarray of cropped checkbox image
    return np.asarray([
        [
            rect,
            img_proc.contains_pixels(
                img_proc.get_checkbox_crop(
                    img, rect), px_threshold, verbose=verbose),
            img_proc.get_checkbox_crop(img, rect)
        ]
        for rect in grouping_rects])


def get_boxes(img, cfg: config.PipelinesCfg, plot=False):
    """
    Single function to run a complicated pipeline to extract rectangular boxes locations from input image.
    Short description of pipeline steps:
    - read image from a path or from `numpy.ndarray`
    - run an image processing iteration for every value provided in `cfg.scaling_factors` list:
        - resize image based on `scaling_factor`
        - try convert to grayscale
        - apply otsu thresholding
        - run dilation based on `cfg` params. If `cfg.dilation_iterations=0` this step will be skipped
        - process image with morphological transformations to extract rectangular shapes based on `cfg` params
        - get contours from transformed images and filter them based on area size and `cfg.wh_ratio_range`
    - aggregate contours from all the iterations and merge overlapping countours
    - convert contours to rectangles `(x, y, width, height)`
    - group rectangles first vertically and then horizontally based on `cfg.vertical_max_distance` and `cfg.horizontal_max_distance_multiplier`
    - draw rectangles and grouped rectangles on original image
    - return rects, grouped_rects, input, output_image

    Args:
        img (str or numpy.ndarray):
            Input image. Can be passed in either as
            `string` (filepath) or as `numpy.ndarray` represting an image
        cfg (boxdetect.config object):
            Object holding all the necessary settings to run this pipeline
        plot (bool, optional):
            Display different stages of image being processed.
            Defaults to False.

    Returns:
        (rects, grouped_rects, src_image, output_image):
            rects - list of rectangles (x, y, width, height) representing all the detected boxes
            grouped_rects - list of rectangles (x, y, width, height) representing grouped boxes
            src_image - same object that was passed in as `img` parameter
            output_image - `numpy.ndarray` representing source image with plotted boxes and grouped boxes
    """ # NOQA E501
    image_org = img_proc.get_image(img)

    ch = None
    if image_org.ndim == 3:
        ch = image_org.shape[-1]
    elif image_org.ndim == 2:
        ch = 1

    # parameters
    width_range = cfg.width_range
    height_range = cfg.height_range
    wh_ratio_range = cfg.wh_ratio_range
    border_thickness = cfg.border_thickness
    thickness = cfg.thickness
    scaling_factors = cfg.scaling_factors

    dilation_kernel = cfg.dilation_kernel
    dilation_iterations = cfg.dilation_iterations

    group_size_range = cfg.group_size_range
    vertical_max_distance = cfg.vertical_max_distance
    horizontal_max_distance_multiplier = cfg.horizontal_max_distance_multiplier  # NOQA E501

    # process image using range of scaling factors
    cnts_list = []
    for scaling_factor in scaling_factors:
        # resize the image for processing time
        image = image_org.copy()
        image = imutils.resize(
            image, width=int(image.shape[0] * scaling_factor))

        resize_ratio = image_org.shape[0] / image.shape[0]
        resize_ratio_inv = image.shape[0] / image_org.shape[0]

        min_w = int(width_range[0] * resize_ratio_inv)
        max_w = int(width_range[1] * resize_ratio_inv)
        min_h = int(height_range[0] * resize_ratio_inv)
        max_h = int(height_range[1] * resize_ratio_inv)

        area_range = (
            round(min_w * min_h * 1.00),
            round(max_w * max_h * 1.00)
        )
        # convert the resized image to grayscale
        try:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        except Exception:
            pass
            # print("WARNING: Failed to convert to grayscale... skipping")

        # apply tresholding to get all the pixel values to either 0 or 255
        # this function also inverts colors
        # (black pixels will become the background)
        image = img_proc.apply_thresholding(image, plot)

        # basic pixel inflation
        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, dilation_kernel)
        image = cv2.dilate(
            image, kernel, iterations=dilation_iterations)
        if plot:
            cv2.imshow("dilated", image)
            cv2.waitKey(0)

        # creating line-shape kernels to be used for image enhancing step
        # try it out only in case of very poor results with previous setup
        # kernels = get_line_kernels(length=4)
        # image = img_proc.apply_merge_transformations(
        #     image, kernels, plot=plot,
        #     transformations=[
        #         (cv2.MORPH_CLOSE, 2),
        #         (cv2.MORPH_OPEN, 2)
        #     ])

        # creating rectangular-shape kernels to be used for
        # extracting rectangular shapes
        kernels = img_proc.get_rect_kernels(
            width_range=(min_w, max_w), height_range=(min_h, max_h),
            wh_ratio_range=wh_ratio_range,
            border_thickness=border_thickness)
        image = img_proc.apply_merge_transformations(
            image, kernels, plot=plot)

        # find contours in the thresholded image
        cnts = rect_proc.get_contours(image)
        # filter countours based on area size
        cnts = rect_proc.filter_contours_by_area_size(cnts, area_range)
        # rescale countours to original image size
        cnts = rect_proc.rescale_contours(cnts, resize_ratio)
        # add countours detected with current scaling factor run
        # to the global collection
        cnts_list += cnts

    # filter global countours by rectangle WxH ratio
    cnts_list = rect_proc.filter_contours_by_wh_ratio(
        cnts_list, wh_ratio_range)

    # merge rectangles into group if overlapping
    rects = rect_proc.group_countours(cnts_list)

    mean_width = np.mean(rects[:, 2])
    # mean_height = np.mean(rects[:, 3])

    # group rectangles vertically (line by line)
    vertical_rect_groups = rect_proc.group_rects(
        rects, max_distance=vertical_max_distance,
        grouping_mode='vertical')

    # group rectangles horizontally (horizontally cluster nearby rects)
    rect_groups = rect_proc.get_groups_from_groups(
        vertical_rect_groups,
        max_distance=mean_width * horizontal_max_distance_multiplier,
        group_size_range=group_size_range, grouping_mode='horizontal')

    # get grouping rectangles
    grouping_rectangles = rect_proc.get_grouping_rectangles(rect_groups)

    if ch == 1:
        image_org = cv2.cvtColor(image_org, cv2.COLOR_GRAY2BGR)

    # draw character rectangles on original image
    image_org = img_proc.draw_rects(
        image_org, rects, color=(0, 255, 0), thickness=thickness)
    # draw grouping rectangles on original image
    image_org = img_proc.draw_rects(
        image_org, grouping_rectangles, color=(255, 0, 0), thickness=thickness)

    if plot:
        cv2.imshow("Org image with boxes", image_org)
        cv2.waitKey(0)

    return rects, grouping_rectangles, img, image_org
