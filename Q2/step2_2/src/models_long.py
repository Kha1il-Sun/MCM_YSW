"""
纵向通道模型
μ(t,b)/p_hit(t,b) 建模
"""

import pandas as pd
import numpy as np
from typing import Callable, Optional, Dict, Any, Tuple
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import cross_val_score, GroupKFold
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import norm
import logging

# 移除pygam依赖，使用sklearn替代
PYGAM_AVAILABLE = False

logger = logging.getLogger(__name__)


class GAMQuantileModel:
    """
    基于GradientBoosting的分位数回归模型（替代GAM）
    """
    
    def __init__(self, tau: float = 0.5, n_splines: int = 20):
        """
        初始化分位数模型
        
        Parameters:
        -----------
        tau : float
            分位数水平 (0, 1)
        n_splines : int
            保留参数以保持接口兼容性（实际不使用）
        """
        self.tau = tau
        self.n_splines = n_splines
        self.model = None
        self.is_fitted = False
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'GAMQuantileModel':
        """
        拟合分位数模型（使用GradientBoosting）
        
        Parameters:
        -----------
        X : np.ndarray
            特征矩阵 [n_samples, 2] (week, BMI)
        y : np.ndarray
            目标变量 (Y浓度)
            
        Returns:
        --------
        self
        """
        # 准备特征工程：添加多项式特征和交互项
        poly = PolynomialFeatures(degree=2, include_bias=False, interaction_only=False)
        X_poly = poly.fit_transform(X)
        
        # 使用GradientBoostingRegressor进行分位数回归
        # 通过调整损失函数来近似分位数回归
        self.model = GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            random_state=42,
            loss='quantile',  # 使用分位数损失
            alpha=self.tau    # 分位数水平
        )
        
        # 拟合模型
        self.model.fit(X_poly, y)
        self.is_fitted = True
        
        logger.info(f"分位数模型拟合完成 (tau={self.tau})")
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        预测分位数
        
        Parameters:
        -----------
        X : np.ndarray
            特征矩阵 [n_samples, 2] (week, BMI)
            
        Returns:
        --------
        np.ndarray
            预测的分位数值
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        # 使用相同的特征工程
        poly = PolynomialFeatures(degree=2, include_bias=False, interaction_only=False)
        X_poly = poly.fit_transform(X)
        
        return self.model.predict(X_poly)
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        获取特征重要性
        
        Returns:
        --------
        Dict[str, float]
            特征重要性字典
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting importance")
        
        # 使用GradientBoosting的特征重要性
        importance = {}
        try:
            # 获取特征重要性
            feature_importance = self.model.feature_importances_
            
            # 对于多项式特征，我们需要映射回原始特征
            # 假设特征顺序为: [week, bmi, week^2, week*bmi, bmi^2]
            if len(feature_importance) >= 5:
                importance['week'] = feature_importance[0] + feature_importance[2]  # week + week^2
                importance['bmi'] = feature_importance[1] + feature_importance[4]   # bmi + bmi^2
                importance['interaction'] = feature_importance[3]  # week*bmi
            else:
                # 如果特征数量不匹配，使用默认值
                importance = {'week': 0.33, 'bmi': 0.33, 'interaction': 0.34}
        except:
            # 如果无法获取特征重要性，使用默认值
            importance = {'week': 0.33, 'bmi': 0.33, 'interaction': 0.34}
        
        return importance


class LongitudinalModel:
    """
    纵向模型基类
    """
    
    def __init__(self, model_type: str = "GBR", features: str = "poly+interactions"):
        """
        初始化纵向模型
        
        Parameters:
        -----------
        model_type : str
            模型类型: "GBR", "GAM"
        features : str
            特征工程类型: "poly+interactions", "simple"
        """
        self.model_type = model_type
        self.features = features
        self.model = None
        self.feature_transformer = None
        self.is_fitted = False
        
    def _create_features(self, X: np.ndarray) -> np.ndarray:
        """
        创建特征
        
        Parameters:
        -----------
        X : np.ndarray
            输入特征 [t, b]
            
        Returns:
        --------
        np.ndarray
            转换后的特征
        """
        if self.features == "simple":
            return X
        elif self.features == "poly+interactions":
            if self.feature_transformer is None:
                self.feature_transformer = PolynomialFeatures(
                    degree=2, include_bias=False, interaction_only=False
                )
            return self.feature_transformer.fit_transform(X)
        else:
            raise ValueError(f"不支持的特征类型: {self.features}")
    
    def fit(self, X: np.ndarray, y: np.ndarray, groups: Optional[np.ndarray] = None) -> 'LongitudinalModel':
        """
        拟合模型
        
        Parameters:
        -----------
        X : np.ndarray
            特征矩阵 [n_samples, 2] (t, b)
        y : np.ndarray
            目标变量
        groups : np.ndarray, optional
            分组信息（用于交叉验证）
            
        Returns:
        --------
        self
        """
        # 创建特征
        X_transformed = self._create_features(X)
        
        # 选择模型
        if self.model_type == "GBR":
            self.model = GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=3,
                random_state=42
            )
        else:
            raise ValueError(f"不支持的模型类型: {self.model_type}")
        
        # 拟合模型
        self.model.fit(X_transformed, y)
        self.is_fitted = True
        
        # 交叉验证评估
        if groups is not None:
            cv_scores = cross_val_score(
                self.model, X_transformed, y, 
                cv=GroupKFold(n_splits=5), groups=groups,
                scoring='neg_mean_squared_error'
            )
            logger.info(f"交叉验证MSE: {-cv_scores.mean():.6f} ± {cv_scores.std():.6f}")
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        预测
        
        Parameters:
        -----------
        X : np.ndarray
            特征矩阵 [n_samples, 2] (t, b)
            
        Returns:
        --------
        np.ndarray
            预测值
        """
        if not self.is_fitted:
            raise ValueError("模型尚未拟合")
        
        X_transformed = self._create_features(X)
        return self.model.predict(X_transformed)


def fit_quantile_models(long_df: pd.DataFrame, tau: float = 0.5, 
                       features: str = "poly+interactions", 
                       model: str = "GBR", use_gam: bool = False) -> LongitudinalModel:
    """
    拟合分位数回归模型
    
    Parameters:
    -----------
    long_df : pd.DataFrame
        长表数据
    tau : float
        分位数 (0, 1)
    features : str
        特征工程类型
    model : str
        模型类型: "GBR", "GAM"
    use_gam : bool
        是否使用GAM模型
        
    Returns:
    --------
    LongitudinalModel or GAMQuantileModel
        拟合的模型
    """
    logger.info(f"拟合分位数回归模型: τ={tau}, 特征={features}, 模型={model}, use_gam={use_gam}")
    
    # 准备数据
    X = long_df[['week', 'BMI_used']].values
    y = long_df['Y_frac'].values
    groups = long_df['id'].values
    
    # 选择模型类型
    if use_gam:
        logger.info("使用分位数回归模型（GradientBoosting）")
        model_obj = GAMQuantileModel(tau=tau, n_splines=20)
        model_obj.fit(X, y)
        
        # 评估模型性能
        y_pred = model_obj.predict(X)
        mse = mean_squared_error(y, y_pred)
        r2 = r2_score(y, y_pred)
        
        logger.info(f"分位数模型性能: MSE={mse:.6f}, R²={r2:.4f}")
        
        # 获取特征重要性
        importance = model_obj.get_feature_importance()
        logger.info(f"特征重要性: {importance}")
        
    else:
        # 使用传统的LongitudinalModel
        model_obj = LongitudinalModel(model_type=model, features=features)
        model_obj.fit(X, y, groups=groups)
        
        # 评估模型性能
        y_pred = model_obj.predict(X)
        mse = mean_squared_error(y, y_pred)
        r2 = r2_score(y, y_pred)
        
        logger.info(f"传统模型性能: MSE={mse:.6f}, R²={r2:.4f}")
    
    return model_obj


def make_p_hit(mu_model: LongitudinalModel, sigma_func: Callable, 
               alpha: float = 0.05) -> Callable:
    """
    创建命中概率函数 p_hit(t,b)
    
    Parameters:
    -----------
    mu_model : LongitudinalModel
        均值模型
    sigma_func : callable
        σ(t,b) 函数
    alpha : float
        置信水平
        
    Returns:
    --------
    callable
        命中概率函数 p_hit(t, b)
    """
    from scipy.stats import norm
    z_alpha = norm.ppf(1 - alpha)
    
    def p_hit(t: float, b: float) -> float:
        """
        计算命中概率
        
        Parameters:
        -----------
        t : float
            孕周
        b : float
            BMI
            
        Returns:
        --------
        float
            命中概率 P(Y ≥ thr_adj(t,b))
        """
        # 预测均值
        mu = mu_model.predict(np.array([[t, b]]))[0]
        
        # 获取σ
        sigma = sigma_func(t, b)
        
        # 计算调整后的阈值
        thr_adj = 0.04 + z_alpha * sigma
        
        # 计算命中概率
        if sigma > 0:
            z_score = (mu - thr_adj) / sigma
            p = norm.cdf(z_score)
        else:
            p = 1.0 if mu >= thr_adj else 0.0
        
        return p
    
    return p_hit


def make_p_hit_batch(mu_model: LongitudinalModel, sigma_func: Callable,
                    alpha: float = 0.05) -> Callable:
    """
    创建批量命中概率函数
    
    Parameters:
    -----------
    mu_model : LongitudinalModel
        均值模型
    sigma_func : callable
        σ(t,b) 函数
    alpha : float
        置信水平
        
    Returns:
    --------
    callable
        批量命中概率函数 p_hit_batch(T, B)
    """
    from scipy.stats import norm
    z_alpha = norm.ppf(1 - alpha)
    
    def p_hit_batch(T: np.ndarray, B: np.ndarray) -> np.ndarray:
        """
        批量计算命中概率
        
        Parameters:
        -----------
        T : np.ndarray
            孕周数组
        B : np.ndarray
            BMI数组
            
        Returns:
        --------
        np.ndarray
            命中概率数组
        """
        # 确保输入是数组
        T = np.asarray(T)
        B = np.asarray(B)
        
        # 预测均值
        X = np.column_stack([T, B])
        mu = mu_model.predict(X)
        
        # 获取σ
        sigma = np.array([sigma_func(t, b) for t, b in zip(T, B)])
        
        # 计算调整后的阈值
        thr_adj = 0.04 + z_alpha * sigma
        
        # 计算命中概率
        z_scores = (mu - thr_adj) / np.maximum(sigma, 1e-8)
        p = norm.cdf(z_scores)
        
        return p
    
    return p_hit_batch


def evaluate_model_performance(mu_model: LongitudinalModel, long_df: pd.DataFrame,
                             sigma_func: Callable, alpha: float = 0.05) -> Dict[str, Any]:
    """
    评估模型性能
    
    Parameters:
    -----------
    mu_model : LongitudinalModel
        均值模型
    long_df : pd.DataFrame
        长表数据
    sigma_func : callable
        σ(t,b) 函数
    alpha : float
        置信水平
        
    Returns:
    --------
    dict
        性能评估结果
    """
    logger.info("评估模型性能...")
    
    # 准备数据
    X = long_df[['week', 'BMI_used']].values
    y_true = long_df['Y_frac'].values
    
    # 预测均值
    y_pred = mu_model.predict(X)
    
    # 计算σ
    sigma_values = np.array([sigma_func(t, b) for t, b in X])
    
    # 计算调整后的阈值
    z_alpha = norm.ppf(1 - alpha)
    thr_adj = 0.04 + z_alpha * sigma_values
    
    # 计算命中概率
    z_scores = (y_pred - thr_adj) / np.maximum(sigma_values, 1e-8)
    p_hit_pred = norm.cdf(z_scores)
    
    # 实际命中情况
    y_hit = (y_true >= 0.04).astype(int)
    
    # 性能指标
    performance = {
        'mse': mean_squared_error(y_true, y_pred),
        'r2': r2_score(y_true, y_pred),
        'mae': np.mean(np.abs(y_true - y_pred)),
        'hit_accuracy': np.mean((p_hit_pred > 0.5) == y_hit),
        'hit_auc': None,  # 可以添加AUC计算
        'sigma_stats': {
            'mean': np.mean(sigma_values),
            'std': np.std(sigma_values),
            'min': np.min(sigma_values),
            'max': np.max(sigma_values)
        }
    }
    
    logger.info(f"性能评估完成: MSE={performance['mse']:.6f}, "
               f"R²={performance['r2']:.4f}, "
               f"命中准确率={performance['hit_accuracy']:.4f}")
    
    return performance


def create_quantile_curves(mu_model: LongitudinalModel, sigma_func: Callable,
                          t_range: Tuple[float, float], b_range: Tuple[float, float],
                          quantiles: list = [0.1, 0.25, 0.5, 0.75, 0.9],
                          resolution: int = 100) -> Dict[str, np.ndarray]:
    """
    创建分位数曲线
    
    Parameters:
    -----------
    mu_model : LongitudinalModel
        均值模型
    sigma_func : callable
        σ(t,b) 函数
    t_range : tuple
        孕周范围
    b_range : tuple
        BMI范围
    quantiles : list
        分位数列表
    resolution : int
        分辨率
        
    Returns:
    --------
    dict
        分位数曲线数据
    """
    logger.info(f"创建分位数曲线: {len(quantiles)} 个分位数")
    
    # 创建网格
    t_grid = np.linspace(t_range[0], t_range[1], resolution)
    b_grid = np.linspace(b_range[0], b_range[1], resolution)
    T, B = np.meshgrid(t_grid, b_grid)
    
    # 预测均值和σ
    X = np.column_stack([T.ravel(), B.ravel()])
    mu = mu_model.predict(X).reshape(T.shape)
    sigma = np.array([sigma_func(t, b) for t, b in zip(T.ravel(), B.ravel())]).reshape(T.shape)
    
    # 计算分位数
    quantile_curves = {}
    for q in quantiles:
        z_q = norm.ppf(q)
        quantile_curves[f'q{q}'] = mu + z_q * sigma
    
    quantile_curves['t_grid'] = t_grid
    quantile_curves['b_grid'] = b_grid
    quantile_curves['mu'] = mu
    quantile_curves['sigma'] = sigma
    
    return quantile_curves


def cross_validate_longitudinal_model(long_df: pd.DataFrame, tau: float = 0.5,
                                    features: str = "poly+interactions",
                                    model: str = "GBR", cv_folds: int = 5) -> Dict[str, Any]:
    """
    交叉验证纵向模型
    
    Parameters:
    -----------
    long_df : pd.DataFrame
        长表数据
    tau : float
        分位数
    features : str
        特征工程类型
    model : str
        模型类型
    cv_folds : int
        交叉验证折数
        
    Returns:
    --------
    dict
        交叉验证结果
    """
    logger.info(f"交叉验证纵向模型: {cv_folds} 折")
    
    # 准备数据
    X = long_df[['week', 'BMI_used']].values
    y = long_df['Y_frac'].values
    groups = long_df['id'].values
    
    # 创建特征
    if features == "poly+interactions":
        feature_transformer = PolynomialFeatures(degree=2, include_bias=False)
        X_transformed = feature_transformer.fit_transform(X)
    else:
        X_transformed = X
    
    # 创建模型
    if model == "GBR":
        model_obj = GradientBoostingRegressor(
            n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42
        )
    else:
        raise ValueError(f"不支持的模型类型: {model}")
    
    # 交叉验证
    cv = GroupKFold(n_splits=cv_folds)
    cv_scores = cross_val_score(
        model_obj, X_transformed, y, cv=cv, groups=groups,
        scoring='neg_mean_squared_error'
    )
    
    # 计算统计量
    cv_results = {
        'mean_mse': -cv_scores.mean(),
        'std_mse': cv_scores.std(),
        'cv_scores': -cv_scores,
        'n_folds': cv_folds
    }
    
    logger.info(f"交叉验证结果: MSE={cv_results['mean_mse']:.6f} ± {cv_results['std_mse']:.6f}")
    
    return cv_results
