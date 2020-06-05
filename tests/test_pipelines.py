import pytest
import numpy as np
import cv2
import sys
sys.path.append(".")
sys.path.append("../.")
from boxdetect import config
from boxdetect import pipelines


def DefaultConfig():
    config.min_w, config.max_w = (25, 45)
    config.min_h, config.max_h = (25, 45)
    config.scaling_factors = [1.0, 2.0]
    config.wh_ratio_range = (0.5, 1.5)
    config.group_size_range = (1, 100)
    config.dilation_iterations = 0
    return config


ALL_BOXES = np.array([
    [13, 8, 44, 34],
    [412, 76, 33, 28],
    [374, 74, 32, 32],
    [530, 74, 35, 32],
    [492, 74, 32, 32],
    [451, 72, 34, 34],
    [336, 74, 32, 32],
    [295, 72, 34, 34],
    [252, 74, 37, 32]], dtype=np.int32)

OUT1 = [(13, 8, 45, 35), (252, 72, 314, 35)]
OUT2 = [(13, 8, 45, 35)]
OUT3 = [(252, 72, 314, 35)]
OUT4 = []

GET_BOXES_TEST_DATA = [
    ("tests/data/tests_color.png", (1, 100),
        ALL_BOXES, OUT1, "tests/data/tests_color_out1.png"),
    ("tests/data/tests_color.png", (1, 1),
        ALL_BOXES, OUT2, "tests/data/tests_color_out2.png"),
    ("tests/data/tests_color.png", (2, 100),
        ALL_BOXES, OUT3, "tests/data/tests_color_out3.png"),
    ("tests/data/tests_color.png", (10, 100),
        ALL_BOXES, OUT4, "tests/data/tests_color_out4.png"),
]


@pytest.mark.parametrize(
    "img, group_size_range, exp_rects, exp_groups, exp_output_image",
    GET_BOXES_TEST_DATA)
def test_get_boxes(
        img, group_size_range, exp_rects, exp_groups, exp_output_image):
    # get default config
    cfg = DefaultConfig()

    cfg.group_size_range = group_size_range

    rects, grouping_rects, image, output_image = pipelines.get_boxes(
        img, config=cfg, plot=False)
    assert((rects == exp_rects).all())
    assert(grouping_rects == exp_groups)
    exp_output_image = cv2.imread(exp_output_image)
    assert((output_image == exp_output_image).all())
