"""
μ/σ先验模型模块

实现用于多重插补的μ(t, BMI, Z, Assay)和σ(t, BMI, Z, Assay)先验模型。
支持经验模型、GAM和树模型等多种建模方法。
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
import logging

from .exceptions import ModelFittingError, ComputationError
from .utils import ensure_numpy, timer, clip_to_bounds

logger = logging.getLogger(__name__)


class MuSigmaModel(ABC):
    """μ/σ模型抽象基类"""
    
    def __init__(self, name: str):
        self.name = name
        self.is_fitted = False
        self.feature_names = None
    
    @abstractmethod
    def fit(self, df: pd.DataFrame) -> None:
        """
        拟合模型
        
        Args:
            df: 训练数据，包含 t, BMI, Y_frac 和协变量
        """
        pass
    
    @abstractmethod
    def mu(self, t: np.ndarray, BMI: np.ndarray, 
           Z: Optional[np.ndarray] = None, 
           Assay: Optional[np.ndarray] = None) -> np.ndarray:
        """
        计算μ(t, BMI, Z, Assay)
        
        Args:
            t: 时间点数组
            BMI: BMI数组
            Z: 协变量矩阵 (n_samples, n_covariates)
            Assay: 检测类型数组
            
        Returns:
            μ值数组
        """
        pass
    
    @abstractmethod
    def sigma(self, t: np.ndarray, BMI: np.ndarray,
              Z: Optional[np.ndarray] = None,
              Assay: Optional[np.ndarray] = None) -> np.ndarray:
        """
        计算σ(t, BMI, Z, Assay)
        
        Args:
            t: 时间点数组
            BMI: BMI数组
            Z: 协变量矩阵
            Assay: 检测类型数组
            
        Returns:
            σ值数组
        """
        pass
    
    def predict(self, t: np.ndarray, BMI: np.ndarray,
                Z: Optional[np.ndarray] = None,
                Assay: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        同时预测μ和σ
        
        Returns:
            (μ, σ) 元组
        """
        if not self.is_fitted:
            raise ModelFittingError(f"模型 {self.name} 尚未拟合")
        
        mu_pred = self.mu(t, BMI, Z, Assay)
        sigma_pred = self.sigma(t, BMI, Z, Assay)
        
        return mu_pred, sigma_pred
    
    def _prepare_features(self, t: np.ndarray, BMI: np.ndarray,
                         Z: Optional[np.ndarray] = None,
                         Assay: Optional[np.ndarray] = None) -> np.ndarray:
        """准备特征矩阵"""
        t = ensure_numpy(t)
        BMI = ensure_numpy(BMI)
        
        # 基础特征
        features = [t, BMI]
        
        # 添加协变量
        if Z is not None:
            Z = ensure_numpy(Z)
            if Z.ndim == 1:
                Z = Z.reshape(-1, 1)
            for i in range(Z.shape[1]):
                features.append(Z[:, i])
        
        # 添加检测类型（如果是分类变量，需要编码）
        if Assay is not None:
            Assay = ensure_numpy(Assay)
            # 简单处理：假设Assay已经是数值编码
            features.append(Assay)
        
        return np.column_stack(features)
    
    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        return {
            "name": self.name,
            "is_fitted": self.is_fitted,
            "feature_names": self.feature_names
        }


class EmpiricalMuSigmaModel(MuSigmaModel):
    """
    基于Q2经验公式的μ/σ模型
    
    使用简单的经验公式或多项式拟合
    """
    
    def __init__(self, 
                 mu_formula: str = "polynomial",
                 sigma_formula: str = "constant",
                 polynomial_degree: int = 2,
                 sigma_factors: Optional[Dict[str, float]] = None):
        """
        初始化经验模型
        
        Args:
            mu_formula: μ的公式类型 ("polynomial", "linear", "custom")
            sigma_formula: σ的公式类型 ("constant", "linear", "assay_dependent")
            polynomial_degree: 多项式阶数
            sigma_factors: σ的调整因子
        """
        super().__init__("EmpiricalMuSigma")
        self.mu_formula = mu_formula
        self.sigma_formula = sigma_formula
        self.polynomial_degree = polynomial_degree
        self.sigma_factors = sigma_factors or {}
        
        # 模型参数
        self.mu_model = None
        self.sigma_base = 1.0
        self.sigma_params = {}
    
    def fit(self, df: pd.DataFrame) -> None:
        """拟合经验模型"""
        logger.info(f"拟合经验模型: μ={self.mu_formula}, σ={self.sigma_formula}")
        
        with timer("经验模型拟合", logger):
            # 准备数据
            required_cols = ['t', 'BMI', 'Y_frac']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                raise ModelFittingError(f"缺少必需列: {missing_cols}")
            
            # 拟合μ模型
            self._fit_mu_model(df)
            
            # 拟合σ模型
            self._fit_sigma_model(df)
            
            self.is_fitted = True
            logger.info("经验模型拟合完成")
    
    def _fit_mu_model(self, df: pd.DataFrame) -> None:
        """拟合μ模型"""
        # 准备特征
        t = df['t'].values
        BMI = df['BMI'].values
        y = df['Y_frac'].values
        
        # 获取协变量
        Z_cols = [col for col in df.columns if col.startswith('Z')]
        Assay_cols = [col for col in df.columns if col.startswith('Assay_')]
        
        features = [t, BMI]
        feature_names = ['t', 'BMI']
        
        if Z_cols:
            Z = df[Z_cols].fillna(0).values
            features.extend([Z[:, i] for i in range(Z.shape[1])])
            feature_names.extend(Z_cols)
        
        if Assay_cols:
            # 简单处理：使用第一个非零的Assay列作为类型指示
            Assay_values = df[Assay_cols].fillna(0).values
            Assay_type = np.argmax(Assay_values, axis=1)  # 主要检测类型
            features.append(Assay_type)
            feature_names.append('Assay_type')
        
        X = np.column_stack(features)
        self.feature_names = feature_names
        
        # 根据公式类型选择模型
        if self.mu_formula == "polynomial":
            poly_features = PolynomialFeatures(degree=self.polynomial_degree, 
                                             include_bias=True)
            self.mu_model = Pipeline([
                ('poly', poly_features),
                ('linear', LinearRegression())
            ])
        elif self.mu_formula == "linear":
            self.mu_model = LinearRegression()
        else:
            raise ModelFittingError(f"不支持的μ公式类型: {self.mu_formula}")
        
        # 拟合模型
        try:
            self.mu_model.fit(X, y)
        except Exception as e:
            raise ModelFittingError(f"μ模型拟合失败: {e}")
    
    def _fit_sigma_model(self, df: pd.DataFrame) -> None:
        """拟合σ模型"""
        if self.sigma_formula == "constant":
            # 使用固定的σ值
            self.sigma_base = self.sigma_factors.get('base', 1.0)
        
        elif self.sigma_formula == "linear":
            # 基于特征的线性σ模型
            # 这里可以基于残差或其他方法估计σ
            t = df['t'].values
            BMI = df['BMI'].values
            
            # 简单的线性关系：σ = σ0 + σ1*t + σ2*BMI
            self.sigma_params = {
                'base': 1.0,
                't_coef': 0.01,
                'bmi_coef': 0.02
            }
        
        elif self.sigma_formula == "assay_dependent":
            # 根据检测类型调整σ
            Assay_cols = [col for col in df.columns if col.startswith('Assay_')]
            if Assay_cols:
                self.sigma_params['assay_factors'] = {
                    'Assay_A': self.sigma_factors.get('Assay_A', 1.0),
                    'Assay_B': self.sigma_factors.get('Assay_B', 1.2),
                    'Assay_C': self.sigma_factors.get('Assay_C', 0.8)
                }
            self.sigma_base = self.sigma_factors.get('base', 1.0)
        
        else:
            raise ModelFittingError(f"不支持的σ公式类型: {self.sigma_formula}")
    
    def mu(self, t: np.ndarray, BMI: np.ndarray,
           Z: Optional[np.ndarray] = None,
           Assay: Optional[np.ndarray] = None) -> np.ndarray:
        """计算μ值"""
        if not self.is_fitted:
            raise ModelFittingError("模型尚未拟合")
        
        # 准备特征矩阵
        features = [ensure_numpy(t), ensure_numpy(BMI)]
        
        if Z is not None:
            Z = ensure_numpy(Z)
            if Z.ndim == 1:
                Z = Z.reshape(-1, 1)
            features.extend([Z[:, i] for i in range(Z.shape[1])])
        
        if Assay is not None:
            Assay = ensure_numpy(Assay)
            features.append(Assay)
        
        X = np.column_stack(features)
        
        try:
            mu_pred = self.mu_model.predict(X)
            # 确保μ在合理范围内
            mu_pred = clip_to_bounds(mu_pred, (0.0, 1.0), warn=False)
            return mu_pred
        except Exception as e:
            raise ComputationError(f"μ计算失败: {e}")
    
    def sigma(self, t: np.ndarray, BMI: np.ndarray,
              Z: Optional[np.ndarray] = None,
              Assay: Optional[np.ndarray] = None) -> np.ndarray:
        """计算σ值"""
        if not self.is_fitted:
            raise ModelFittingError("模型尚未拟合")
        
        t = ensure_numpy(t)
        BMI = ensure_numpy(BMI)
        n_samples = len(t)
        
        if self.sigma_formula == "constant":
            sigma_pred = np.full(n_samples, self.sigma_base)
        
        elif self.sigma_formula == "linear":
            sigma_pred = (self.sigma_params['base'] + 
                         self.sigma_params['t_coef'] * t +
                         self.sigma_params['bmi_coef'] * BMI)
        
        elif self.sigma_formula == "assay_dependent":
            sigma_pred = np.full(n_samples, self.sigma_base)
            
            if Assay is not None:
                Assay = ensure_numpy(Assay)
                assay_factors = self.sigma_params.get('assay_factors', {})
                
                # 根据检测类型调整
                for i, assay_val in enumerate(Assay):
                    if assay_val == 0:  # Assay_A
                        factor = assay_factors.get('Assay_A', 1.0)
                    elif assay_val == 1:  # Assay_B
                        factor = assay_factors.get('Assay_B', 1.0)
                    elif assay_val == 2:  # Assay_C
                        factor = assay_factors.get('Assay_C', 1.0)
                    else:
                        factor = 1.0
                    
                    sigma_pred[i] *= factor
        
        else:
            sigma_pred = np.full(n_samples, 1.0)
        
        # 确保σ为正
        sigma_pred = np.maximum(sigma_pred, 0.01)
        
        return sigma_pred


class RandomForestMuSigmaModel(MuSigmaModel):
    """
    基于随机森林的μ/σ模型
    
    使用两个独立的随机森林分别预测μ和σ
    """
    
    def __init__(self, 
                 n_estimators: int = 100,
                 max_depth: Optional[int] = None,
                 min_samples_split: int = 5,
                 random_state: int = 42):
        """
        初始化随机森林模型
        
        Args:
            n_estimators: 树的数量
            max_depth: 最大深度
            min_samples_split: 分裂所需的最小样本数
            random_state: 随机种子
        """
        super().__init__("RandomForestMuSigma")
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.random_state = random_state
        
        # 模型实例
        self.mu_model = None
        self.sigma_model = None
    
    def fit(self, df: pd.DataFrame) -> None:
        """拟合随机森林模型"""
        logger.info(f"拟合随机森林模型: n_estimators={self.n_estimators}")
        
        with timer("随机森林模型拟合", logger):
            # 准备数据
            X, y_mu, y_sigma = self._prepare_training_data(df)
            
            # 初始化模型
            rf_params = {
                'n_estimators': self.n_estimators,
                'max_depth': self.max_depth,
                'min_samples_split': self.min_samples_split,
                'random_state': self.random_state,
                'n_jobs': -1
            }
            
            self.mu_model = RandomForestRegressor(**rf_params)
            self.sigma_model = RandomForestRegressor(**rf_params)
            
            # 拟合μ模型
            try:
                self.mu_model.fit(X, y_mu)
            except Exception as e:
                raise ModelFittingError(f"μ随机森林拟合失败: {e}")
            
            # 拟合σ模型（使用残差的绝对值作为目标）
            try:
                mu_pred = self.mu_model.predict(X)
                residuals = np.abs(y_mu - mu_pred)
                self.sigma_model.fit(X, residuals)
            except Exception as e:
                raise ModelFittingError(f"σ随机森林拟合失败: {e}")
            
            self.is_fitted = True
            logger.info("随机森林模型拟合完成")
    
    def _prepare_training_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """准备训练数据"""
        # 基础特征
        t = df['t'].values
        BMI = df['BMI'].values
        y_mu = df['Y_frac'].values
        
        features = [t, BMI]
        feature_names = ['t', 'BMI']
        
        # 添加协变量
        Z_cols = [col for col in df.columns if col.startswith('Z')]
        if Z_cols:
            Z = df[Z_cols].fillna(0).values
            features.extend([Z[:, i] for i in range(Z.shape[1])])
            feature_names.extend(Z_cols)
        
        # 添加检测类型
        Assay_cols = [col for col in df.columns if col.startswith('Assay_')]
        if Assay_cols:
            # One-hot编码或使用主要类型
            Assay_values = df[Assay_cols].fillna(0).values
            features.extend([Assay_values[:, i] for i in range(Assay_values.shape[1])])
            feature_names.extend(Assay_cols)
        
        # 添加交互特征
        features.extend([
            t * BMI,  # 时间-BMI交互
            t ** 2,   # 时间平方
            BMI ** 2  # BMI平方
        ])
        feature_names.extend(['t_BMI', 't_squared', 'BMI_squared'])
        
        X = np.column_stack(features)
        self.feature_names = feature_names
        
        # σ的目标值（这里使用简单的启发式方法）
        y_sigma = np.ones_like(y_mu) * 0.1  # 初始值
        
        return X, y_mu, y_sigma
    
    def mu(self, t: np.ndarray, BMI: np.ndarray,
           Z: Optional[np.ndarray] = None,
           Assay: Optional[np.ndarray] = None) -> np.ndarray:
        """预测μ值"""
        if not self.is_fitted:
            raise ModelFittingError("模型尚未拟合")
        
        X = self._prepare_prediction_features(t, BMI, Z, Assay)
        
        try:
            mu_pred = self.mu_model.predict(X)
            mu_pred = clip_to_bounds(mu_pred, (0.0, 1.0), warn=False)
            return mu_pred
        except Exception as e:
            raise ComputationError(f"μ预测失败: {e}")
    
    def sigma(self, t: np.ndarray, BMI: np.ndarray,
              Z: Optional[np.ndarray] = None,
              Assay: Optional[np.ndarray] = None) -> np.ndarray:
        """预测σ值"""
        if not self.is_fitted:
            raise ModelFittingError("模型尚未拟合")
        
        X = self._prepare_prediction_features(t, BMI, Z, Assay)
        
        try:
            sigma_pred = self.sigma_model.predict(X)
            sigma_pred = np.maximum(sigma_pred, 0.01)  # 确保为正
            return sigma_pred
        except Exception as e:
            raise ComputationError(f"σ预测失败: {e}")
    
    def _prepare_prediction_features(self, t: np.ndarray, BMI: np.ndarray,
                                   Z: Optional[np.ndarray] = None,
                                   Assay: Optional[np.ndarray] = None) -> np.ndarray:
        """准备预测特征"""
        t = ensure_numpy(t)
        BMI = ensure_numpy(BMI)
        n_samples = len(t)
        
        features = [t, BMI]
        
        # 协变量（用零填充缺失的）
        if Z is not None:
            Z = ensure_numpy(Z)
            if Z.ndim == 1:
                Z = Z.reshape(-1, 1)
            features.extend([Z[:, i] for i in range(Z.shape[1])])
        else:
            # 填充零值以匹配训练时的特征数量
            n_z_features = len([name for name in self.feature_names if name.startswith('Z')])
            for i in range(n_z_features):
                features.append(np.zeros(n_samples))
        
        # 检测类型
        if Assay is not None:
            Assay = ensure_numpy(Assay)
            if Assay.ndim == 1:
                # 假设是one-hot编码
                n_assay_features = len([name for name in self.feature_names if name.startswith('Assay_')])
                for i in range(n_assay_features):
                    features.append((Assay == i).astype(float))
            else:
                features.extend([Assay[:, i] for i in range(Assay.shape[1])])
        else:
            # 填充零值
            n_assay_features = len([name for name in self.feature_names if name.startswith('Assay_')])
            for i in range(n_assay_features):
                features.append(np.zeros(n_samples))
        
        # 交互特征
        features.extend([
            t * BMI,
            t ** 2,
            BMI ** 2
        ])
        
        return np.column_stack(features)
    
    def get_feature_importance(self) -> Dict[str, Dict[str, float]]:
        """获取特征重要性"""
        if not self.is_fitted:
            raise ModelFittingError("模型尚未拟合")
        
        return {
            "mu_importance": dict(zip(self.feature_names, self.mu_model.feature_importances_)),
            "sigma_importance": dict(zip(self.feature_names, self.sigma_model.feature_importances_))
        }


def create_mu_sigma_model(model_type: str = "empirical", **kwargs) -> MuSigmaModel:
    """
    创建μ/σ模型的工厂函数
    
    Args:
        model_type: 模型类型 ("empirical", "random_forest")
        **kwargs: 模型参数
        
    Returns:
        μ/σ模型实例
    """
    if model_type == "empirical":
        return EmpiricalMuSigmaModel(**kwargs)
    elif model_type == "random_forest":
        return RandomForestMuSigmaModel(**kwargs)
    else:
        raise ValueError(f"不支持的模型类型: {model_type}")


def evaluate_mu_sigma_model(model: MuSigmaModel, 
                           test_df: pd.DataFrame) -> Dict[str, float]:
    """
    评估μ/σ模型性能
    
    Args:
        model: 训练好的模型
        test_df: 测试数据
        
    Returns:
        评估指标字典
    """
    if not model.is_fitted:
        raise ModelFittingError("模型尚未拟合")
    
    # 准备测试数据
    t = test_df['t'].values
    BMI = test_df['BMI'].values
    y_true = test_df['Y_frac'].values
    
    # 获取协变量
    Z_cols = [col for col in test_df.columns if col.startswith('Z')]
    Z = test_df[Z_cols].fillna(0).values if Z_cols else None
    
    Assay_cols = [col for col in test_df.columns if col.startswith('Assay_')]
    if Assay_cols:
        Assay_values = test_df[Assay_cols].fillna(0).values
        Assay = np.argmax(Assay_values, axis=1)
    else:
        Assay = None
    
    # 预测
    mu_pred, sigma_pred = model.predict(t, BMI, Z, Assay)
    
    # 计算评估指标
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    
    metrics = {
        "mu_mse": mean_squared_error(y_true, mu_pred),
        "mu_mae": mean_absolute_error(y_true, mu_pred),
        "mu_r2": r2_score(y_true, mu_pred),
        "sigma_mean": np.mean(sigma_pred),
        "sigma_std": np.std(sigma_pred),
        "sigma_min": np.min(sigma_pred),
        "sigma_max": np.max(sigma_pred)
    }
    
    return metrics