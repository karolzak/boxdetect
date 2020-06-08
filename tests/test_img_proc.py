import cv2
import numpy as np
import sys
sys.path.append(".")
sys.path.append("../.")
from boxdetect import config, img_proc


def DefaultConfig():
    config.width_range = (25, 50)
    config.height_range = (25, 50)
    config.scaling_factors = [2.0]
    config.wh_ratio_range = (0.5, 1.5)
    config.group_size_range = (1, 100)
    config.dilation_iterations = 0
    return config


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
