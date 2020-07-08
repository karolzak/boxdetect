import pytest
import sys
sys.path.append(".")
sys.path.append("../.")
from boxdetect import config
from boxdetect import pipelines


def test_save_load_config(capsys):
    cfg = config.PipelinesConfig()
    cfg.morph_kernels_thickness = 10
    cfg.save_yaml('test_cfg.yaml')
    cfg2 = config.PipelinesConfig('test_cfg.yaml')
    assert(cfg.__dict__ == cfg2.__dict__)
    cfg.new_var = 10
    cfg.save_yaml('test_cfg.yaml')
    cfg2.load_yaml('test_cfg.yaml')
    captured = capsys.readouterr()
    assert("WARNING" in captured.out)


def test_update_num_iterations():
    cfg = config.PipelinesConfig()
    cfg.height_range = (5, 5)
    cfg.width_range = [(10, 10), (20, 20)]
    cfg.update_num_iterations()
    assert(cfg.num_iterations == 2)
    assert(len(cfg.height_range) == 2)
    assert(len(cfg.width_range) == 2)


def test_autoconfig_simple():
    box_sizes = [(42, 44), (41, 47), (41, 44), (41, 44), (125, 54), (92, 103)]
    file_path = "tests/data/autoconfig_simple/dummy_example.png"

    cfg = config.PipelinesConfig()
    cfg.autoconfigure(box_sizes)

    checkboxes = pipelines.get_checkboxes(
        file_path, cfg=cfg, px_threshold=0.01, plot=False, verbose=False)
    assert(len(checkboxes) == 12)

    cfg = config.PipelinesConfig()
    cfg.autoconfigure(box_sizes)

    rects, groups, _, _ = pipelines.get_boxes(
        file_path, cfg=cfg, plot=False)
    assert(len(rects) == 23)
    assert(len(groups) == 14)


def test_autoconfig_from_vott_simple():
    vott_dir = "tests/data/autoconfig_simple"
    file_path = "tests/data/autoconfig_simple/dummy_example.png"

    cfg = config.PipelinesConfig()
    cfg.autoconfigure_from_vott(vott_dir, class_tags=['box'])

    checkboxes = pipelines.get_checkboxes(
        file_path, cfg=cfg, px_threshold=0.01, plot=False, verbose=False)
    assert(len(checkboxes) == 12)

    cfg = config.PipelinesConfig()
    cfg.autoconfigure_from_vott(vott_dir, class_tags=['box'])

    rects, groups, _, _ = pipelines.get_boxes(
        file_path, cfg=cfg, plot=False)
    assert(len(rects) == 23)
    assert(len(groups) == 14)