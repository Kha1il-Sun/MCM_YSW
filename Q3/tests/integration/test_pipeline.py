"""
管道集成测试
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path

from src.config import MCMConfig
from src.mu_sigma_models import create_mu_sigma_model
from src.mi_interval_imputer import IntervalImputer
from src.aft_models import create_aft_model
from src.risk_objective import create_risk_objective
from src.surv_predict import create_survival_predictor


@pytest.fixture
def sample_long_data():
    """创建示例长格式数据"""
    data = {
        'id': [1, 1, 1, 2, 2, 2, 3, 3, 3],
        't': [1, 2, 3, 1, 2, 3, 1, 2, 3],
        'BMI': [25.0, 25.1, 25.0, 22.0, 22.1, 22.0, 28.0, 28.2, 28.1],
        'Y_frac': [0.10, 0.15, 0.25, 0.08, 0.12, 0.18, 0.12, 0.20, 0.35],
        'Z1': [0.5, 0.5, 0.5, 0.3, 0.3, 0.3, 0.7, 0.7, 0.7]
    }
    return pd.DataFrame(data)


@pytest.fixture
def sample_config():
    """创建示例配置"""
    return MCMConfig(
        seed=42,
        threshold=0.04,
        tau=0.90,
        mi={'M': 3, 'q': 0.02, 'use_local_sigma': False, 'deterministic_by_id': True},
        weights={'w1': 1.0, 'w2': 0.6, 'w3': 0.3},
        delay={'pref_week': 15.0, 'scale': 10.0}
    )


def test_mu_sigma_model_fitting(sample_long_data):
    """测试μ/σ模型拟合"""
    model = create_mu_sigma_model("empirical")
    model.fit(sample_long_data)
    
    assert model.is_fitted
    
    # 测试预测
    t = np.array([1, 2, 3])
    BMI = np.array([25.0, 25.0, 25.0])
    
    mu_pred, sigma_pred = model.predict(t, BMI)
    
    assert len(mu_pred) == 3
    assert len(sigma_pred) == 3
    assert np.all(sigma_pred > 0)


def test_multiple_imputation_basic(sample_long_data, sample_config):
    """测试基础多重插补功能"""
    # 拟合μ/σ模型
    mu_sigma_model = create_mu_sigma_model("empirical")
    mu_sigma_model.fit(sample_long_data)
    
    # 创建插补器
    imputer = IntervalImputer(
        mu_sigma_model=mu_sigma_model,
        threshold=sample_config.threshold,
        M=sample_config.mi.M,
        q=sample_config.mi.q,
        deterministic_by_id=True,
        global_seed=sample_config.seed,
        n_jobs=1
    )
    
    # 执行插补
    imputed_datasets = imputer.impute_intervals(sample_long_data)
    
    assert len(imputed_datasets) == sample_config.mi.M
    
    for dataset in imputed_datasets:
        assert isinstance(dataset, pd.DataFrame)
        assert len(dataset) == sample_long_data['id'].nunique()
        assert 'L' in dataset.columns
        assert 'R' in dataset.columns
        assert 'censor_type' in dataset.columns


def test_aft_model_fitting():
    """测试AFT模型拟合"""
    # 创建简单的测试数据
    np.random.seed(42)
    n = 50
    
    X = np.random.randn(n, 2)
    X = np.column_stack([np.ones(n), X])  # 添加截距
    
    # 生成模拟的删失数据
    L = np.random.exponential(10, n)
    R = L + np.random.exponential(5, n)
    censor_type = np.array(['interval'] * n)
    
    # 拟合模型
    model = create_aft_model("lognormal")
    model.fit(X, L, R, censor_type)
    
    assert model.is_fitted
    assert model.coef_ is not None
    assert model.scale_ > 0
    
    # 测试预测
    t_grid = np.array([5, 10, 15])
    F_curves = model.cumulative_density_function(t_grid, X[:5])
    
    assert F_curves.shape == (3, 5)
    assert np.all(F_curves >= 0)
    assert np.all(F_curves <= 1)


@pytest.mark.slow
def test_end_to_end_pipeline(sample_long_data, sample_config):
    """端到端管道测试"""
    # 1. μ/σ模型
    mu_sigma_model = create_mu_sigma_model("empirical")
    mu_sigma_model.fit(sample_long_data)
    
    # 2. 多重插补
    imputer = IntervalImputer(
        mu_sigma_model=mu_sigma_model,
        threshold=sample_config.threshold,
        M=2,  # 减少M以加速测试
        q=sample_config.mi.q,
        deterministic_by_id=True,
        global_seed=sample_config.seed,
        n_jobs=1
    )
    
    imputed_datasets = imputer.impute_intervals(sample_long_data)
    
    # 3. AFT模型拟合
    dataset = imputed_datasets[0]
    
    # 准备特征
    X = dataset[['BMI_base']].values
    X = np.column_stack([np.ones(len(X)), X])
    
    L = dataset['L'].values
    R = dataset['R'].fillna(np.inf).values
    censor_type = dataset['censor_type'].values
    
    aft_model = create_aft_model("lognormal")
    aft_model.fit(X, L, R, censor_type)
    
    # 4. 生存预测
    survival_predictor = create_survival_predictor(aft_model)
    
    t_grid, F_curves = survival_predictor.predict_survival_curves(X)
    
    assert len(t_grid) > 0
    assert F_curves.shape[1] == len(X)
    
    # 5. 风险优化
    config_dict = {
        'weights': {'w1': 1.0, 'w2': 0.6, 'w3': 0.3},
        'delay': {'pref_week': 15.0, 'scale': 10.0},
        'tau': 0.90,
        'clinical_bounds': [12.0, 25.0]
    }
    
    risk_objective = create_risk_objective(survival_predictor, config_dict)
    
    # 简化的优化测试
    w_test = 15.0
    risk_result = risk_objective.compute_risk(w_test, X)
    
    assert 'total_risk' in risk_result
    assert 'mean_success_prob' in risk_result
    assert isinstance(risk_result['total_risk'], float)
    assert 0 <= risk_result['mean_success_prob'] <= 1


if __name__ == "__main__":
    pytest.main([__file__])