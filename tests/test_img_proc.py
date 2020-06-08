import cv2
import sys
sys.path.append(".")
sys.path.append("../.")
from boxdetect import config, img_proc


def DefaultConfig():
    config.min_w, config.max_w = (25, 45)
    config.min_h, config.max_h = (25, 45)
    config.scaling_factors = [1.0, 2.0]
    config.wh_ratio_range = (0.5, 1.5)
    config.group_size_range = (1, 100)
    config.dilation_iterations = 0
    return config


IMG1 = cv2.imread("tests/data/tests_color_enhance1.png")
IMG2 = cv2.imread("tests/data/tests_color_enhance2.png")


def test_apply_merge_transformations():
    cfg = DefaultConfig()

    resize_ratio_inv = 0.4166666666666667

    min_w_res = int(cfg.min_w * resize_ratio_inv)
    max_w_res = int(cfg.max_w * resize_ratio_inv)
    min_h_res = int(cfg.min_h * resize_ratio_inv)
    max_h_res = int(cfg.max_h * resize_ratio_inv)
    kernels = img_proc.get_rect_kernels(
        wh_ratio_range=cfg.wh_ratio_range,
        min_w=min_w_res, max_w=max_w_res,
        min_h=min_h_res, max_h=max_h_res,
        pad=cfg.padding)
    image = img_proc.apply_merge_transformations(IMG1.copy(), kernels)
    assert((image == IMG2.copy()).all())
