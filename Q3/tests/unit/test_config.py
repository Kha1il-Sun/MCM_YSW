"""
配置模块单元测试
"""

import pytest
import tempfile
import yaml
from pathlib import Path

from src.config import MCMConfig, load_config, ConfigError


def test_mcm_config_creation():
    """测试MCMConfig创建"""
    config = MCMConfig()
    
    assert config.seed == 42
    assert config.threshold == 0.04
    assert config.tau == 0.90
    assert len(config.clinical_bounds) == 2


def test_mcm_config_validation():
    """测试配置验证"""
    # 测试无效的clinical_bounds
    with pytest.raises(ValueError):
        MCMConfig(clinical_bounds=[25.0, 12.0])  # min > max
    
    # 测试无效的tau
    with pytest.raises(ValueError):
        MCMConfig(tau=1.1)  # > 1


def test_load_config_success():
    """测试成功加载配置"""
    config_data = {
        'seed': 123,
        'threshold': 0.05,
        'tau': 0.85,
        'clinical_bounds': [10.0, 30.0],
        'weights': {'w1': 1.0, 'w2': 0.5, 'w3': 0.2},
        'mi': {'M': 10, 'q': 0.01}
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(config_data, f)
        temp_path = f.name
    
    try:
        config = load_config(temp_path)
        assert config.seed == 123
        assert config.threshold == 0.05
        assert config.weights.w1 == 1.0
        assert config.mi.M == 10
    finally:
        Path(temp_path).unlink()


def test_load_config_file_not_found():
    """测试配置文件不存在"""
    with pytest.raises(ConfigError):
        load_config("nonexistent.yaml")


def test_load_config_invalid_yaml():
    """测试无效YAML"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write("invalid: yaml: content:")
        temp_path = f.name
    
    try:
        with pytest.raises(ConfigError):
            load_config(temp_path)
    finally:
        Path(temp_path).unlink()