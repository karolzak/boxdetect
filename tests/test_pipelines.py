import pytest
import cv2
from boxdetect import config, pipelines


def DefaultConfig():
    config.width_range = (25, 50)
    config.height_range = (25, 50)
    config.scaling_factors = [2.0]
    config.wh_ratio_range = (0.5, 1.5)
    config.group_size_range = (1, 100)
    config.dilation_iterations = 0
    return config


def OpenTestImage(file_path):
    return cv2.imread(file_path)


def GetGrayscaleFromPath(file_path):
    im = cv2.imread(file_path)
    return cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)


OUT1 = [(13, 8, 45, 35), (252, 72, 314, 35)]
OUT2 = [(13, 8, 45, 35)]
OUT3 = [(252, 72, 314, 35)]
OUT4 = []

GET_BOXES_TEST_DATA = [
    ("tests/data/tests_color.png", (1, 100), 9, 2),
    ("tests/data/tests_color.png", (1, 1), 9, 1),
    ("tests/data/tests_color.png", (2, 100), 9, 1),
    ("tests/data/tests_color.png", (10, 100), 9, 0),
    (OpenTestImage("tests/data/tests_color.png"), (1, 100), 9, 2),
    (GetGrayscaleFromPath("tests/data/tests_color.png"), (1, 100), 9, 2),
]

GET_BOXES_FAILS_TEST_DATA = [
    ("", (1, 100),
        [], [], AttributeError)
]


@pytest.mark.parametrize(
    "img, group_size_range, exp_rects_count, exp_groups_count,",
    GET_BOXES_TEST_DATA)
def test_get_boxes(
        img, group_size_range, exp_rects_count, exp_groups_count):
    # get default config
    cfg = DefaultConfig()

    cfg.group_size_range = group_size_range

    rects, grouping_rects, image, output_image = pipelines.get_boxes(
        img, config=cfg, plot=False)
    assert(len(rects) == exp_rects_count)
    assert(len(grouping_rects) == exp_groups_count)


@pytest.mark.parametrize(
    "inputs, group_size_range, exp_rects, exp_groups, exp_exception",
    GET_BOXES_FAILS_TEST_DATA)
def test_get_boxes_fails(
        inputs, group_size_range, exp_rects, exp_groups, exp_exception):
    # get default config
    cfg = DefaultConfig()

    cfg.group_size_range = group_size_range
    with pytest.raises(exp_exception):
        rects, grouping_rects, image, output_image = pipelines.get_boxes(
            inputs, config=cfg, plot=False)
