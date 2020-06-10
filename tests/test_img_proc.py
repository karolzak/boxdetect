import cv2
import numpy as np
import sys
sys.path.append(".")
sys.path.append("../.")
from boxdetect import config, img_proc


def DefaultConfig():
    cfg = config.PipelinesCfg()
    cfg.width_range = (25, 50)
    cfg.height_range = (25, 50)
    cfg.scaling_factors = [2.0]
    cfg.wh_ratio_range = (0.5, 1.5)
    cfg.group_size_range = (1, 100)
    cfg.dilation_iterations = 0
    return cfg


IMG1 = cv2.imread("tests/data/tests_color_enhance1.png")
IMG2 = cv2.imread("tests/data/tests_color_enhance2.png")


def test_apply_merge_transformations():
    cfg = DefaultConfig()

    resize_ratio_inv = 0.4166666666666667

    min_w_res = int(cfg.width_range[0] * resize_ratio_inv)
    max_w_res = int(cfg.width_range[1] * resize_ratio_inv)
    min_h_res = int(cfg.height_range[0] * resize_ratio_inv)
    max_h_res = int(cfg.height_range[1] * resize_ratio_inv)

    kernels = img_proc.get_rect_kernels(
        width_range=(min_w_res, max_w_res),
        height_range=(min_h_res, max_h_res),
        wh_ratio_range=cfg.wh_ratio_range,
        border_thickness=cfg.border_thickness)

    image = img_proc.apply_merge_transformations(IMG1.copy(), kernels)

    threshold = 0.02 * image.shape[0] * image.shape[1] * image.shape[2] * image.max()  # NOQA E501
    assert(abs(np.sum(image - IMG2.copy())) < threshold)


def test_get_image():
    result1 = img_proc.get_image("tests/data/tests_color_enhance1.png")
    result2 = img_proc.get_image(IMG1)

    assert((result1 == result2).all())
    assert((result1 == IMG1).all())
    assert((result2 == IMG1).all())


def test_contains_pixels():
    input_img_true = np.ones((40, 40))
    input_img_false = np.zeros((40, 40))
    result_true = img_proc.contains_pixels(input_img_true, px_threshold=0.1)
    result_false = img_proc.contains_pixels(input_img_false, px_threshold=0.1)
    assert(result_true)
    assert(not result_false)


def test_get_checkbox_crop():
    ones = np.ones((40, 40))
    input_img = np.pad(
        ones, ((4, 100), (4, 100)), 'constant', constant_values=0)
    print(input_img.shape)
    cropped = img_proc.get_checkbox_crop(
        input_img, (0, 0, 48, 48), border_crop_factor=0.10)
    assert(cropped.shape == (40, 40))
    assert((cropped == ones).all())
