import pytest
import numpy as np
import cv2
from boxdetect import config, pipelines


def DefaultConfig():
    cfg = config.PipelinesCfg()
    cfg.width_range = (25, 50)
    cfg.height_range = (25, 50)
    cfg.scaling_factors = [2.0]
    cfg.wh_ratio_range = (0.5, 1.5)
    cfg.group_size_range = (1, 100)
    cfg.dilation_iterations = 0
    return cfg


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
        img, cfg=cfg, plot=False)
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
            inputs, cfg=cfg, plot=False)


def test_get_checkboxes():
    # get default config
    cfg = DefaultConfig()

    cfg.width_range = (40, 55)
    cfg.height_range = (40, 55)
    cfg.scaling_factors = [1.0]
    cfg.wh_ratio_range = (0.8, 1.2)
    cfg.dilation_iterations = 0

    input_image = "tests/data/dummy_example.png"

    checkboxes = pipelines.get_checkboxes(
        input_image, cfg=cfg, plot=False)
    # check if it recognized correct number of checkboxes as checked
    assert(np.sum(checkboxes[:, 1]) == 7)
    # check if specific checkboxes where recognized as checked/non checked
    assert((checkboxes[:, 1][-3:] == [False, False, False]).all())
