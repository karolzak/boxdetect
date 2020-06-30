import pytest
import sys
sys.path.append(".")
sys.path.append("../.")
from boxdetect import config


def test_config(capsys):
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
