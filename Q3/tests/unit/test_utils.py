"""
工具函数单元测试
"""

import pytest
import numpy as np

from src.utils import (
    logdiffexp, 
    rng_for_id, 
    ensure_numpy, 
    safe_divide,
    validate_probability,
    create_grid
)


def test_logdiffexp():
    """测试数值稳定的log-diff-exp"""
    # 正常情况
    a, b = 2.0, 1.0
    result = logdiffexp(a, b)
    expected = np.log(np.exp(a) - np.exp(b))
    np.testing.assert_allclose(result, expected, rtol=1e-10)
    
    # 边界情况
    with pytest.raises(ValueError):
        logdiffexp(1.0, 2.0)  # b > a


def test_rng_for_id():
    """测试按ID的随机数生成器"""
    rng1 = rng_for_id(42, "patient_1")
    rng2 = rng_for_id(42, "patient_1")
    rng3 = rng_for_id(42, "patient_2")
    
    # 相同ID应该产生相同的随机数
    assert rng1.random() == rng2.random()
    
    # 不同ID应该产生不同的随机数
    rng1 = rng_for_id(42, "patient_1")
    rng3 = rng_for_id(42, "patient_2")
    assert rng1.random() != rng3.random()


def test_ensure_numpy():
    """测试numpy转换函数"""
    # 列表
    result = ensure_numpy([1, 2, 3])
    assert isinstance(result, np.ndarray)
    np.testing.assert_array_equal(result, [1, 2, 3])
    
    # 标量
    result = ensure_numpy(5)
    assert isinstance(result, np.ndarray)
    np.testing.assert_array_equal(result, [5])
    
    # 已经是numpy数组
    arr = np.array([1, 2, 3])
    result = ensure_numpy(arr)
    assert result is arr


def test_safe_divide():
    """测试安全除法"""
    numerator = np.array([1, 2, 3])
    denominator = np.array([1, 0, 2])
    
    result = safe_divide(numerator, denominator, fill_value=999)
    expected = np.array([1.0, 999.0, 1.5])
    
    np.testing.assert_array_equal(result, expected)


def test_validate_probability():
    """测试概率验证"""
    # 有效概率
    valid_probs = np.array([0.0, 0.5, 1.0])
    result = validate_probability(valid_probs)
    np.testing.assert_array_equal(result, valid_probs)
    
    # 无效概率 - 超出范围
    with pytest.raises(ValueError):
        validate_probability(np.array([-0.1, 0.5]))
    
    with pytest.raises(ValueError):
        validate_probability(np.array([0.5, 1.1]))
    
    # 无效概率 - NaN
    with pytest.raises(ValueError):
        validate_probability(np.array([0.5, np.nan]))


def test_create_grid():
    """测试网格创建"""
    # 线性网格
    grid = create_grid((0, 10), 11, log_scale=False)
    expected = np.linspace(0, 10, 11)
    np.testing.assert_array_almost_equal(grid, expected)
    
    # 对数网格
    grid = create_grid((1, 100), 3, log_scale=True)
    expected = np.array([1, 10, 100])
    np.testing.assert_array_almost_equal(grid, expected)
    
    # 对数网格无效范围
    with pytest.raises(ValueError):
        create_grid((0, 10), 5, log_scale=True)