import cv2
import imutils
import numpy as np

from . import rect_proc, img_proc


def get_boxes(img, config, plot=False):
    """
    Single function to run a complicated pipeline to extract rectangular boxes locations from input image. 
    Short description of pipeline steps:
    - read image from a path or from `numpy.ndarray`
    - run an image processing iteration for every value provided in `config.scaling_factors` list:
        - resize image based on `scaling_factor`
        - try convert to grayscale
        - apply otsu thresholding
        - run dilation based on `config` params. If `config.dilation_iterations=0` this step will be skipped
        - process image with morphological transformations to extract rectangular shapes based on `config` params
        - get contours from transformed images and filter them based on area size and `config.wh_ratio_range`
    - aggregate contours from all the iterations and merge overlapping countours
    - convert contours to rectangles `(x, y, width, height)`
    - group rectangles first vertically and then horizontally based on `config.vertical_max_distance` and `config.horizontal_max_distance_multiplier`
    - draw rectangles and grouped rectangles on original image
    - return rects, grouped_rects, input, output_image

    Args:
        img (str or numpy.ndarray):
            Input image. Can be passed in either as
            `string` (filepath) or as `numpy.ndarray` represting an image
        config (boxdetect.config object):
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
    assert(type(img) in [np.ndarray, str])
    if type(img) is np.ndarray:
        image_org = img.copy()
        image_org = image_org.astype(np.uint8)
    elif type(img) is str:
        print("Processing file: ", img)
        image_org = cv2.imread(img)

    ch = None
    if image_org.ndim == 3:
        ch = image_org.shape[-1]
    elif image_org.ndim == 2:
        ch = 1

    # parameters
    min_w, max_w = (config.min_w, config.max_w)
    min_h, max_h = (config.min_h, config.max_h)
    wh_ratio_range = config.wh_ratio_range
    padding = config.padding
    thickness = config.thickness
    scaling_factors = config.scaling_factors

    dilation_kernel = config.dilation_kernel
    dilation_iterations = config.dilation_iterations

    group_size_range = config.group_size_range
    vertical_max_distance = config.vertical_max_distance
    horizontal_max_distance_multiplier = config.horizontal_max_distance_multiplier  # NOQA E501

    # process image using range of scaling factors
    cnts_list = []
    for scaling_factor in scaling_factors:
        # resize the image for processing time
        image = image_org.copy()
        image = imutils.resize(
            image, width=int(image.shape[0] * scaling_factor))

        resize_ratio = image_org.shape[0] / image.shape[0]
        resize_ratio_inv = image.shape[0] / image_org.shape[0]

        min_w_res = int(min_w * resize_ratio_inv)
        max_w_res = int(max_w * resize_ratio_inv)
        min_h_res = int(min_h * resize_ratio_inv)
        max_h_res = int(max_h * resize_ratio_inv)

        area_range = (
            round(min_w_res * min_h_res * 0.90),
            round(max_w_res * max_h_res * 1.00)
        )
        # convert the resized image to grayscale
        try:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        except Exception as e:
            print("Warning: failed to convert to grayscale...")
            print(e)

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
        # image = enhance_image(image, kernels, plot)

        # creating rectangular-shape kernels to be used for
        # extracting rectangular shapes
        kernels = img_proc.get_rect_kernels(
            wh_ratio_range=wh_ratio_range,
            min_w=min_w_res,	max_w=max_w_res,
            min_h=min_h_res,	max_h=max_h_res,
            pad=padding)
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

    # filter gloal countours by rectangle WxH ratio
    cnts_list = rect_proc.filter_contours_by_rect_ratio(
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
        # image_org = np.repeat(np.expand_dims(image_org, axis=-1), 3, axis=-1)
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
