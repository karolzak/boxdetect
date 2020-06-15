import pytest
import numpy as np
import sys
sys.path.append(".")
sys.path.append("../.")
from boxdetect import rect_proc


CNTS1 = np.load("tests/data/cnts1.npy", allow_pickle=True)
IMG_4_CNTS = np.load("tests/data/image_for_cnts.npy", allow_pickle=True)

TEST_CNTS = [
    np.array([
        [[156,  32]], [[157,  31]], [[168,  31]], [[169,  32]],
        [[169,  43]], [[168,  44]], [[157,  44]], [[156,  43]]],
        dtype=np.int32),
    np.array([
        [[221,  31]], [[222,  30]], [[234,  30]], [[235,  31]],
        [[235,  43]], [[234,  44]], [[222,  44]], [[221,  43]]],
        dtype=np.int32)
]

TEST_CNTS_SCALED = [
    np.array([
        [[374,  76]],
        [[376,  74]],
        [[403,  74]],
        [[405,  76]],
        [[405, 103]],
        [[403, 105]],
        [[376, 105]],
        [[374, 103]]]),
    np.array([
        [[530,  74]],
        [[532,  72]],
        [[561,  72]],
        [[564,  74]],
        [[564, 103]],
        [[561, 105]],
        [[532, 105]],
        [[530, 103]]]),
]

RECTS = np.array([
    [412,  76,  33,  28],
    [374,  74,  32,  32],
    [530,  74,  35,  32],
    [492,  74,  32,  32],
    [451,  72,  34,  34],
    [336,  74,  32,  32],
    [295,  72,  34,  34],
    [252,  74,  37,  32],
    [12,   8,  44,  33]], dtype=np.int32)


def test_get_bounding_rect():
    rect = rect_proc.get_bounding_rect(TEST_CNTS_SCALED[0])
    assert(rect == (374,  74,  32,  32, True))
    rect = rect_proc.get_bounding_rect(
        np.array([
            [[564, 103]],
            [[561, 105]],
            [[561,  72]],
            [[405,  76]],
            [[405, 103]],
            [[403, 105]],
            [[376, 105]],
            [[374, 103]]])
    )
    assert(rect[4] is False)


def test_group_rects():
    vertical_max_distance = 15.0
    rects = np.vstack((RECTS, RECTS[-1] + [200, 0, 0 ,0]))

    vertical_rect_groups = rect_proc.group_rects(
        rects, max_distance=vertical_max_distance,
        grouping_mode='vertical')

    assert((vertical_rect_groups[0] == rects[-2]).any())
    assert((vertical_rect_groups[1] == rects[:-2]).any())
    assert(
        len(vertical_rect_groups[0]) == 2
        and len(vertical_rect_groups[1]) == 8)

    mean_width = np.mean(rects[:, 2])

    rect_groups = rect_proc.get_groups_from_groups(
        vertical_rect_groups,
        max_distance=mean_width * 4, group_size_range=(1, 100),
        grouping_mode='horizontal')

    assert(
        len(rect_groups[0]) == 1 and len(rect_groups[1]) == 1
        and len(rect_groups[2]) == 8)

    grouping_rectangles = rect_proc.get_grouping_rectangles(rect_groups)
    assert(
        grouping_rectangles == [
            tuple(rects[-2]+[0, 0, 1, 1]),
            tuple(rects[-1]+[0, 0, 1, 1]),
            (252, 72, 314, 35)])


def test_filter_contours_by_area_size():
    area_range = (90, 170)
    cnts = rect_proc.filter_contours_by_area_size(TEST_CNTS, area_range)
    assert(len(cnts) == 1)
    assert((cnts == TEST_CNTS[0]).all())


def test_filter_contours_by_wh_ratio():
    wh_ratio_range = (0.5, 1.0)
    cnts = rect_proc.filter_contours_by_wh_ratio(
        TEST_CNTS_SCALED, wh_ratio_range)
    assert(len(cnts) == 1)
    assert((cnts == TEST_CNTS_SCALED[0]).all())


def test_wh_ratio_in_range():
    wh_ratio_range = (0.5, 1.0)
    is_rect = rect_proc.wh_ratio_in_range(TEST_CNTS_SCALED[0], wh_ratio_range)
    assert(is_rect is True)


def test_group_countours():
    test_cnts = [TEST_CNTS_SCALED[0], TEST_CNTS_SCALED[0]]
    rects = rect_proc.group_countours(test_cnts, epsilon=0.1)
    assert((rects == [[374,  74,  32,  32]]).all())


def test_get_contours():
    image = IMG_4_CNTS
    exp_cnts = CNTS1
    cnts = rect_proc.get_contours(image)
    for x, y in zip(cnts, exp_cnts):
        assert((x == y).all())


def test_rescale_contours():
    resize_ratio = 2.4
    cnts = rect_proc.rescale_contours(TEST_CNTS, resize_ratio)
    for x, y in zip(cnts, TEST_CNTS_SCALED):
        assert((y == x).all())


def test_filter_contours_by_size_range():
    results = rect_proc.filter_contours_by_size_range(
        TEST_CNTS_SCALED, width_range=None, height_range=None)
    assert(results == TEST_CNTS_SCALED)
    results = rect_proc.filter_contours_by_size_range(
        CNTS1, width_range=(13, 20), height_range=(13, 20))
    assert(len(results) == 10)
    results = rect_proc.filter_contours_by_size_range(
        CNTS1, width_range=(130, 140))
    assert(len(results) == 1)


def test_size_in_range():
    c = TEST_CNTS_SCALED[0]
    result = rect_proc.size_in_range(
        c, width_range=None, height_range=None)
    assert(result)
    result = rect_proc.size_in_range(
        c, width_range=(10, 10), height_range=None)
    assert(not result)
    result = rect_proc.size_in_range(
        c, width_range=(30, 35), height_range=None)
    assert(result)
    result = rect_proc.size_in_range(
        c, height_range=(30, 35))
    assert(result)
    result = rect_proc.size_in_range(
        c, width_range=(30, 35), height_range=(30, 35))
    assert(result)
