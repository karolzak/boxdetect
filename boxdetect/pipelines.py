import cv2
import imutils
import numpy as np

from . import rect_proc, img_proc, config


def get_checkboxes(
        img, cfg: config.PipelinesConfig,
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
        ::

            [
                [checkbox1_coords, contains_pixels1, cropped_checkbox1],
                [checkbox2_coords, contains_pixels2, cropped_checkbox2],
                ...
            ]

    Args:
        img (str or numpy.ndarray): Input image.
            Can be passed in either as `string` (filepath) or as `numpy.ndarray` representing an image
        cfg (boxdetect.config object): Object holding all the necessary settings to run this pipeline
        px_threshold (float, optional): This is the threshold used when estimating if pixels are present inside the checkbox.
            Defaults to 0.1.
        plot (bool, optional): Display different stages of image being processed.
            Defaults to False.
        verbose (bool, optional): Defines if messages should be printed or not.
            Defaults to False.

    Returns:
        numpy.ndarray: Return an array of arrays with following values:
            checkbox_coords - `numpy.ndarray` rectangle (x,y,width,height)
            contains_pixels - `bool`, True/False
            cropped_checkbox - `numpy.ndarray` of cropped checkbox image
            ::

                [
                    [checkbox1_coords, contains_pixels1, cropped_checkbox1],
                    [checkbox2_coords, contains_pixels2, cropped_checkbox2],
                    ...
                ]
            
            
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


def get_boxes(img, cfg: config.PipelinesConfig, plot=False):
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

    cfg.update_num_iterations()

    # parameters
    thickness = cfg.thickness
    scaling_factors = cfg.scaling_factors

    group_size_range = cfg.group_size_range
    # vertical_max_distance = cfg.vertical_max_distance
    # horizontal_max_distance = cfg.horizontal_max_distance  # NOQA E501

    # process image using range of scaling factors
    cnts_list = []
    for scaling_factor in scaling_factors:
        # resize the image for processing time
        image_scaled = image_org.copy()
        image_scaled = imutils.resize(
            image_scaled, width=int(image_scaled.shape[1] * scaling_factor))

        resize_ratio = image_org.shape[0] / image_scaled.shape[0]
        resize_ratio_inv = image_scaled.shape[0] / image_org.shape[0]

        # convert the resized image to grayscale
        try:
            image_scaled = cv2.cvtColor(image_scaled, cv2.COLOR_BGR2GRAY)
        except Exception:
            pass
            # print("WARNING: Failed to convert to grayscale... skipping")

        # apply tresholding to get all the pixel values to either 0 or 255
        # this function also inverts colors
        # (black pixels will become the background)
        image_scaled = img_proc.apply_thresholding(image_scaled, plot)

        for (
            width_range, height_range, wh_ratio_range,
            dilation_iterations,
            dilation_kernel, vertical_max_distance,
            horizontal_max_distance,
            morph_kernels_type,
            morph_kernels_thickness
        ) in cfg.variables_as_iterators():
            image = image_scaled.copy()

            min_w = int(width_range[0] * resize_ratio_inv)
            max_w = int(width_range[1] * resize_ratio_inv)
            min_h = int(height_range[0] * resize_ratio_inv)
            max_h = int(height_range[1] * resize_ratio_inv)

            area_range = (
                round(min_w * min_h * 0.90),
                round(max_w * max_h * 1.10)
            )

            # basic pixel inflation
            kernel = cv2.getStructuringElement(
                cv2.MORPH_RECT, dilation_kernel)
            image = cv2.dilate(
                image, kernel, iterations=dilation_iterations)
            if plot:  # pragma: no cover
                cv2.imshow("dilated", image)
                cv2.waitKey(0)

            if morph_kernels_type == 'rectangles':
                # creating rectangular-shape kernels to be used for
                # extracting rectangular shapes
                kernels = img_proc.get_rect_kernels(
                    width_range=(min_w, max_w), height_range=(min_h, max_h),
                    wh_ratio_range=wh_ratio_range,
                    border_thickness=morph_kernels_thickness)
            elif morph_kernels_type == 'lines':
                # creating line-shape kernels to be used for image enhancing
                kernels = img_proc.get_line_kernels(
                    horizontal_length=int(min_w * 0.95),
                    vertical_length=int(min_h * 0.95),
                    thickness=morph_kernels_thickness)

            image = img_proc.apply_merge_transformations(
                image, kernels, plot=plot)

            # find contours in the thresholded image
            cnts = rect_proc.get_contours(image)

            # filter countours based on area size
            cnts = rect_proc.filter_contours_by_area_size(cnts, area_range)

            # rescale countours to original image size
            cnts = rect_proc.rescale_contours(cnts, resize_ratio)

            cnts = rect_proc.filter_contours_by_size_range(
                cnts, width_range, height_range)

            # filter global countours by rectangle WxH ratio
            cnts = rect_proc.filter_contours_by_wh_ratio(
                cnts, wh_ratio_range)
            # add countours detected with current scaling factor run
            # to the global collection
            cnts_list += cnts

    # rects = [rect_proc.get_bounding_rect(c)[:4] for c in cnts]
    # merge rectangles into group if overlapping
    rects = rect_proc.group_countours(cnts_list)

    # mean_width = np.mean(rects[:, 2])
    # mean_height = np.mean(rects[:, 3])

    # group rectangles vertically (line by line)
    vertical_rect_groups = rect_proc.group_rects(
        rects, max_distance=vertical_max_distance,
        grouping_mode='vertical')

    # group rectangles horizontally (horizontally cluster nearby rects)
    rect_groups = rect_proc.get_groups_from_groups(
        vertical_rect_groups,
        max_distance=horizontal_max_distance,
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

    if plot:  # pragma: no cover
        cv2.imshow("Org image with boxes", image_org)
        cv2.waitKey(0)

    if len(rects) == 0:
        print("WARNING: No rectangles were found in the input image.")

    return rects, grouping_rectangles, img, image_org
