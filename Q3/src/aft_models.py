"""
AFT模型拟合和预测模块

实现加速失效时间(AFT)模型的拟合、预测和集成。
支持log-normal、log-logistic、Weibull等分布族。
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from scipy import stats, optimize
from scipy.special import expit, logit
import warnings
from joblib import Parallel, delayed
import logging

from .exceptions import ModelFittingError, ComputationError, OptimizationError
from .utils import logdiffexp, log_sum_exp, ensure_numpy, timer, clip_to_bounds
from .io_utils import save_model, load_model

logger = logging.getLogger(__name__)


class AFTModel(ABC):
    """AFT模型抽象基类"""
    
    def __init__(self, family: str):
        self.family = family
        self.is_fitted = False
        self.coef_ = None
        self.scale_ = None
        self.feature_names = None
        self.aic_ = None
        self.bic_ = None
        self.loglik_ = None
    
    @abstractmethod
    def fit(self, X: np.ndarray, L: np.ndarray, R: np.ndarray, 
            censor_type: np.ndarray) -> None:
        """
        拟合AFT模型
        
        Args:
            X: 协变量矩阵 (n_samples, n_features)
            L: 左端点数组
            R: 右端点数组 (可包含NaN表示右删失)
            censor_type: 删失类型数组
        """
        pass
    
    @abstractmethod
    def survival_function(self, t: np.ndarray, X: np.ndarray) -> np.ndarray:
        """
        计算生存函数 S(t|X) = P(T > t | X)
        
        Args:
            t: 时间点数组
            X: 协变量矩阵
            
        Returns:
            生存概率矩阵 (len(t), n_samples)
        """
        pass
    
    @abstractmethod
    def cumulative_density_function(self, t: np.ndarray, X: np.ndarray) -> np.ndarray:
        """
        计算累积分布函数 F(t|X) = P(T <= t | X)
        
        Args:
            t: 时间点数组
            X: 协变量矩阵
            
        Returns:
            累积概率矩阵 (len(t), n_samples)
        """
        pass
    
    def probability_density_function(self, t: np.ndarray, X: np.ndarray) -> np.ndarray:
        """
        计算概率密度函数 f(t|X)
        
        默认实现通过数值微分计算
        """
        dt = 1e-6
        t_plus = t + dt
        F_plus = self.cumulative_density_function(t_plus, X)
        F = self.cumulative_density_function(t, X)
        return (F_plus - F) / dt
    
    def predict_percentiles(self, X: np.ndarray, 
                          percentiles: np.ndarray = np.array([0.1, 0.25, 0.5, 0.75, 0.9])
                          ) -> np.ndarray:
        """
        预测分位数
        
        Args:
            X: 协变量矩阵
            percentiles: 分位数数组
            
        Returns:
            分位数矩阵 (n_samples, len(percentiles))
        """
        if not self.is_fitted:
            raise ModelFittingError("模型尚未拟合")
        
        n_samples = X.shape[0]
        n_percentiles = len(percentiles)
        quantiles = np.zeros((n_samples, n_percentiles))
        
        for i in range(n_samples):
            X_i = X[i:i+1]  # 保持二维
            
            for j, p in enumerate(percentiles):
                # 使用二分搜索找分位数
                def objective(t):
                    return self.cumulative_density_function(np.array([t]), X_i)[0, 0] - p
                
                try:
                    # 初始搜索范围
                    t_low, t_high = 0.1, 200.0
                    
                    # 确保搜索范围包含根
                    while objective(t_low) > 0:
                        t_low /= 2
                        if t_low < 1e-6:
                            t_low = 1e-6
                            break
                    
                    while objective(t_high) < 0:
                        t_high *= 2
                        if t_high > 1e6:
                            t_high = 1e6
                            break
                    
                    result = optimize.brentq(objective, t_low, t_high)
                    quantiles[i, j] = result
                    
                except (ValueError, RuntimeError):
                    # 如果优化失败，使用默认值
                    quantiles[i, j] = np.nan
        
        return quantiles
    
    def log_likelihood(self, X: np.ndarray, L: np.ndarray, R: np.ndarray,
                      censor_type: np.ndarray) -> float:
        """计算对数似然"""
        if not self.is_fitted:
            raise ModelFittingError("模型尚未拟合")
        
        loglik = 0.0
        n_samples = X.shape[0]
        
        for i in range(n_samples):
            X_i = X[i:i+1]
            L_i = L[i]
            R_i = R[i]
            ctype = censor_type[i]
            
            if ctype == 'interval':
                # 区间删失: log(F(R) - F(L))
                F_R = self.cumulative_density_function(np.array([R_i]), X_i)[0, 0]
                F_L = self.cumulative_density_function(np.array([L_i]), X_i)[0, 0]
                
                if F_R > F_L:
                    # 使用数值稳定的log-diff-exp
                    log_F_R = np.log(np.maximum(F_R, 1e-15))
                    log_F_L = np.log(np.maximum(F_L, 1e-15))
                    loglik += logdiffexp(log_F_R, log_F_L)
                else:
                    loglik += -np.inf  # 无效区间
                    
            elif ctype == 'right':
                # 右删失: log(S(L))
                S_L = self.survival_function(np.array([L_i]), X_i)[0, 0]
                loglik += np.log(np.maximum(S_L, 1e-15))
                
            elif ctype == 'left':
                # 左删失: log(F(R))
                F_R = self.cumulative_density_function(np.array([R_i]), X_i)[0, 0]
                loglik += np.log(np.maximum(F_R, 1e-15))
                
            elif ctype == 'exact':
                # 精确观测: log(f(L))
                f_L = self.probability_density_function(np.array([L_i]), X_i)[0, 0]
                loglik += np.log(np.maximum(f_L, 1e-15))
        
        return loglik
    
    def compute_information_criteria(self, X: np.ndarray, L: np.ndarray, 
                                   R: np.ndarray, censor_type: np.ndarray) -> Dict[str, float]:
        """计算信息准则"""
        loglik = self.log_likelihood(X, L, R, censor_type)
        n_params = len(self.coef_) + 1  # 回归系数 + 尺度参数
        n_samples = X.shape[0]
        
        aic = -2 * loglik + 2 * n_params
        bic = -2 * loglik + n_params * np.log(n_samples)
        
        self.aic_ = aic
        self.bic_ = bic
        self.loglik_ = loglik
        
        return {"AIC": aic, "BIC": bic, "loglik": loglik}
    
    def get_model_summary(self) -> Dict[str, Any]:
        """获取模型摘要"""
        if not self.is_fitted:
            return {"family": self.family, "is_fitted": False}
        
        return {
            "family": self.family,
            "is_fitted": True,
            "n_features": len(self.coef_),
            "coef": self.coef_.tolist() if self.coef_ is not None else None,
            "scale": self.scale_,
            "AIC": self.aic_,
            "BIC": self.bic_,
            "loglik": self.loglik_,
            "feature_names": self.feature_names
        }


class LogNormalAFTModel(AFTModel):
    """Log-normal AFT模型"""
    
    def __init__(self):
        super().__init__("lognormal")
    
    def fit(self, X: np.ndarray, L: np.ndarray, R: np.ndarray, 
            censor_type: np.ndarray) -> None:
        """拟合log-normal AFT模型"""
        logger.debug("拟合log-normal AFT模型")
        
        X = ensure_numpy(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        n_samples, n_features = X.shape
        
        # 初始参数估计
        initial_params = np.zeros(n_features + 1)  # beta + log(sigma)
        
        # 定义负对数似然函数
        def neg_log_likelihood(params):
            beta = params[:-1]
            log_sigma = params[-1]
            sigma = np.exp(log_sigma)
            
            loglik = 0.0
            
            for i in range(n_samples):
                x_beta = np.dot(X[i], beta)
                
                if censor_type[i] == 'interval':
                    # 区间删失
                    z_L = (np.log(L[i]) - x_beta) / sigma
                    z_R = (np.log(R[i]) - x_beta) / sigma
                    
                    F_R = stats.norm.cdf(z_R)
                    F_L = stats.norm.cdf(z_L)
                    
                    if F_R > F_L:
                        loglik += np.log(F_R - F_L)
                    else:
                        return np.inf
                        
                elif censor_type[i] == 'right':
                    # 右删失
                    z_L = (np.log(L[i]) - x_beta) / sigma
                    S_L = 1 - stats.norm.cdf(z_L)
                    loglik += np.log(S_L)
                    
                elif censor_type[i] == 'left':
                    # 左删失
                    z_R = (np.log(R[i]) - x_beta) / sigma
                    F_R = stats.norm.cdf(z_R)
                    loglik += np.log(F_R)
                    
                elif censor_type[i] == 'exact':
                    # 精确观测
                    z = (np.log(L[i]) - x_beta) / sigma
                    f = stats.norm.pdf(z) / (sigma * L[i])
                    loglik += np.log(f)
            
            return -loglik
        
        # 优化
        try:
            result = optimize.minimize(
                neg_log_likelihood,
                initial_params,
                method='L-BFGS-B',
                options={'maxiter': 1000, 'ftol': 1e-9}
            )
            
            if not result.success:
                raise OptimizationError(f"优化失败: {result.message}")
            
            self.coef_ = result.x[:-1]
            self.scale_ = np.exp(result.x[-1])
            self.is_fitted = True
            
            # 计算信息准则
            self.compute_information_criteria(X, L, R, censor_type)
            
        except Exception as e:
            raise ModelFittingError(f"Log-normal AFT模型拟合失败: {e}")
    
    def survival_function(self, t: np.ndarray, X: np.ndarray) -> np.ndarray:
        """计算生存函数"""
        if not self.is_fitted:
            raise ModelFittingError("模型尚未拟合")
        
        t = ensure_numpy(t)
        X = ensure_numpy(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        # S(t|x) = 1 - Φ((log(t) - x'β) / σ)
        linear_pred = X @ self.coef_  # (n_samples,)
        
        result = np.zeros((len(t), X.shape[0]))
        
        for i, t_val in enumerate(t):
            if t_val <= 0:
                result[i, :] = 1.0
            else:
                z = (np.log(t_val) - linear_pred) / self.scale_
                result[i, :] = 1 - stats.norm.cdf(z)
        
        return result
    
    def cumulative_density_function(self, t: np.ndarray, X: np.ndarray) -> np.ndarray:
        """计算累积分布函数"""
        return 1 - self.survival_function(t, X)


class LogLogisticAFTModel(AFTModel):
    """Log-logistic AFT模型"""
    
    def __init__(self):
        super().__init__("loglogistic")
    
    def fit(self, X: np.ndarray, L: np.ndarray, R: np.ndarray, 
            censor_type: np.ndarray) -> None:
        """拟合log-logistic AFT模型"""
        logger.debug("拟合log-logistic AFT模型")
        
        X = ensure_numpy(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        n_samples, n_features = X.shape
        
        # 初始参数
        initial_params = np.zeros(n_features + 1)
        
        def neg_log_likelihood(params):
            beta = params[:-1]
            log_sigma = params[-1]
            sigma = np.exp(log_sigma)
            
            loglik = 0.0
            
            for i in range(n_samples):
                x_beta = np.dot(X[i], beta)
                
                if censor_type[i] == 'interval':
                    # F(t) = 1 / (1 + exp(-(log(t) - x'β) / σ))
                    z_L = (np.log(L[i]) - x_beta) / sigma
                    z_R = (np.log(R[i]) - x_beta) / sigma
                    
                    F_R = expit(z_R)  # 1 / (1 + exp(-z))
                    F_L = expit(z_L)
                    
                    if F_R > F_L:
                        loglik += np.log(F_R - F_L)
                    else:
                        return np.inf
                        
                elif censor_type[i] == 'right':
                    z_L = (np.log(L[i]) - x_beta) / sigma
                    S_L = 1 - expit(z_L)
                    loglik += np.log(S_L)
                    
                elif censor_type[i] == 'left':
                    z_R = (np.log(R[i]) - x_beta) / sigma
                    F_R = expit(z_R)
                    loglik += np.log(F_R)
                    
                elif censor_type[i] == 'exact':
                    z = (np.log(L[i]) - x_beta) / sigma
                    # f(t) = (1/σt) * exp(z) / (1 + exp(z))^2
                    exp_z = np.exp(z)
                    f = (1 / (sigma * L[i])) * exp_z / (1 + exp_z)**2
                    loglik += np.log(f)
            
            return -loglik
        
        try:
            result = optimize.minimize(
                neg_log_likelihood,
                initial_params,
                method='L-BFGS-B',
                options={'maxiter': 1000, 'ftol': 1e-9}
            )
            
            if not result.success:
                raise OptimizationError(f"优化失败: {result.message}")
            
            self.coef_ = result.x[:-1]
            self.scale_ = np.exp(result.x[-1])
            self.is_fitted = True
            
            self.compute_information_criteria(X, L, R, censor_type)
            
        except Exception as e:
            raise ModelFittingError(f"Log-logistic AFT模型拟合失败: {e}")
    
    def survival_function(self, t: np.ndarray, X: np.ndarray) -> np.ndarray:
        """计算生存函数"""
        if not self.is_fitted:
            raise ModelFittingError("模型尚未拟合")
        
        t = ensure_numpy(t)
        X = ensure_numpy(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        linear_pred = X @ self.coef_
        result = np.zeros((len(t), X.shape[0]))
        
        for i, t_val in enumerate(t):
            if t_val <= 0:
                result[i, :] = 1.0
            else:
                z = (np.log(t_val) - linear_pred) / self.scale_
                result[i, :] = 1 - expit(z)
        
        return result
    
    def cumulative_density_function(self, t: np.ndarray, X: np.ndarray) -> np.ndarray:
        """计算累积分布函数"""
        return 1 - self.survival_function(t, X)


class WeibullAFTModel(AFTModel):
    """Weibull AFT模型"""
    
    def __init__(self):
        super().__init__("weibull")
    
    def fit(self, X: np.ndarray, L: np.ndarray, R: np.ndarray, 
            censor_type: np.ndarray) -> None:
        """拟合Weibull AFT模型"""
        logger.debug("拟合Weibull AFT模型")
        
        X = ensure_numpy(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        n_samples, n_features = X.shape
        
        # 初始参数
        initial_params = np.zeros(n_features + 1)
        
        def neg_log_likelihood(params):
            beta = params[:-1]
            log_sigma = params[-1]
            sigma = np.exp(log_sigma)
            
            loglik = 0.0
            
            for i in range(n_samples):
                x_beta = np.dot(X[i], beta)
                
                if censor_type[i] == 'interval':
                    # Weibull: F(t) = 1 - exp(-((t/λ)^k))
                    # AFT形式: F(t) = 1 - exp(-exp((log(t) - x'β) / σ))
                    z_L = (np.log(L[i]) - x_beta) / sigma
                    z_R = (np.log(R[i]) - x_beta) / sigma
                    
                    F_R = 1 - np.exp(-np.exp(z_R))
                    F_L = 1 - np.exp(-np.exp(z_L))
                    
                    if F_R > F_L:
                        loglik += np.log(F_R - F_L)
                    else:
                        return np.inf
                        
                elif censor_type[i] == 'right':
                    z_L = (np.log(L[i]) - x_beta) / sigma
                    S_L = np.exp(-np.exp(z_L))
                    loglik += np.log(S_L)
                    
                elif censor_type[i] == 'left':
                    z_R = (np.log(R[i]) - x_beta) / sigma
                    F_R = 1 - np.exp(-np.exp(z_R))
                    loglik += np.log(F_R)
                    
                elif censor_type[i] == 'exact':
                    z = (np.log(L[i]) - x_beta) / sigma
                    # f(t) = (1/σt) * exp(z - exp(z))
                    f = (1 / (sigma * L[i])) * np.exp(z - np.exp(z))
                    loglik += np.log(f)
            
            return -loglik
        
        try:
            result = optimize.minimize(
                neg_log_likelihood,
                initial_params,
                method='L-BFGS-B',
                options={'maxiter': 1000, 'ftol': 1e-9}
            )
            
            if not result.success:
                raise OptimizationError(f"优化失败: {result.message}")
            
            self.coef_ = result.x[:-1]
            self.scale_ = np.exp(result.x[-1])
            self.is_fitted = True
            
            self.compute_information_criteria(X, L, R, censor_type)
            
        except Exception as e:
            raise ModelFittingError(f"Weibull AFT模型拟合失败: {e}")
    
    def survival_function(self, t: np.ndarray, X: np.ndarray) -> np.ndarray:
        """计算生存函数"""
        if not self.is_fitted:
            raise ModelFittingError("模型尚未拟合")
        
        t = ensure_numpy(t)
        X = ensure_numpy(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        linear_pred = X @ self.coef_
        result = np.zeros((len(t), X.shape[0]))
        
        for i, t_val in enumerate(t):
            if t_val <= 0:
                result[i, :] = 1.0
            else:
                z = (np.log(t_val) - linear_pred) / self.scale_
                result[i, :] = np.exp(-np.exp(z))
        
        return result
    
    def cumulative_density_function(self, t: np.ndarray, X: np.ndarray) -> np.ndarray:
        """计算累积分布函数"""
        return 1 - self.survival_function(t, X)


class AFTEnsemble:
    """AFT模型集成"""
    
    def __init__(self, models: List[AFTModel], 
                 ensemble_method: str = "stacking",
                 selection_criterion: str = "AIC",
                 penalty: float = 1e-3):
        """
        初始化AFT集成模型
        
        Args:
            models: AFT模型列表
            ensemble_method: 集成方法 ("stacking", "avg", "best")
            selection_criterion: 模型选择准则 ("AIC", "BIC")
            penalty: 正则化惩罚参数
        """
        self.models = models
        self.ensemble_method = ensemble_method
        self.selection_criterion = selection_criterion
        self.penalty = penalty
        self.weights_ = None
        self.best_model_idx_ = None
        self.is_fitted = False
    
    def fit(self, X: np.ndarray, L: np.ndarray, R: np.ndarray, 
            censor_type: np.ndarray) -> None:
        """拟合集成模型"""
        logger.info(f"拟合AFT集成模型: {len(self.models)} 个基模型")
        
        with timer("AFT集成模型拟合", logger):
            # 拟合所有基模型
            for i, model in enumerate(self.models):
                try:
                    logger.debug(f"拟合第 {i+1} 个模型: {model.family}")
                    model.fit(X, L, R, censor_type)
                except Exception as e:
                    logger.warning(f"模型 {model.family} 拟合失败: {e}")
                    continue
            
            # 检查成功拟合的模型
            fitted_models = [m for m in self.models if m.is_fitted]
            if not fitted_models:
                raise ModelFittingError("没有模型成功拟合")
            
            logger.info(f"成功拟合 {len(fitted_models)} 个模型")
            
            # 根据集成方法确定权重
            if self.ensemble_method == "best":
                self._select_best_model(fitted_models)
            elif self.ensemble_method == "avg":
                self._equal_weights(fitted_models)
            elif self.ensemble_method == "stacking":
                self._stacking_weights(fitted_models, X, L, R, censor_type)
            else:
                raise ValueError(f"不支持的集成方法: {self.ensemble_method}")
            
            self.models = fitted_models
            self.is_fitted = True
    
    def _select_best_model(self, fitted_models: List[AFTModel]) -> None:
        """选择最佳单模型"""
        criteria_values = []
        
        for model in fitted_models:
            if self.selection_criterion == "AIC":
                criteria_values.append(model.aic_)
            elif self.selection_criterion == "BIC":
                criteria_values.append(model.bic_)
            else:
                raise ValueError(f"不支持的选择准则: {self.selection_criterion}")
        
        self.best_model_idx_ = np.argmin(criteria_values)
        self.weights_ = np.zeros(len(fitted_models))
        self.weights_[self.best_model_idx_] = 1.0
        
        logger.info(f"选择最佳模型: {fitted_models[self.best_model_idx_].family}")
    
    def _equal_weights(self, fitted_models: List[AFTModel]) -> None:
        """等权重平均"""
        n_models = len(fitted_models)
        self.weights_ = np.ones(n_models) / n_models
        logger.info("使用等权重平均")
    
    def _stacking_weights(self, fitted_models: List[AFTModel],
                         X: np.ndarray, L: np.ndarray, R: np.ndarray,
                         censor_type: np.ndarray) -> None:
        """学习stacking权重"""
        logger.info("学习stacking权重")
        
        n_models = len(fitted_models)
        n_samples = X.shape[0]
        
        # 获取所有模型的预测概率
        predictions = np.zeros((n_samples, n_models))
        
        for i, model in enumerate(fitted_models):
            for j in range(n_samples):
                X_j = X[j:j+1]
                
                if censor_type[j] == 'interval':
                    F_R = model.cumulative_density_function(np.array([R[j]]), X_j)[0, 0]
                    F_L = model.cumulative_density_function(np.array([L[j]]), X_j)[0, 0]
                    predictions[j, i] = F_R - F_L
                elif censor_type[j] == 'right':
                    S_L = model.survival_function(np.array([L[j]]), X_j)[0, 0]
                    predictions[j, i] = S_L
                elif censor_type[j] == 'left':
                    F_R = model.cumulative_density_function(np.array([R[j]]), X_j)[0, 0]
                    predictions[j, i] = F_R
                elif censor_type[j] == 'exact':
                    f_L = model.probability_density_function(np.array([L[j]]), X_j)[0, 0]
                    predictions[j, i] = f_L
        
        # 优化权重
        def objective(weights):
            weights = np.maximum(weights, 1e-15)  # 确保正权重
            weights = weights / np.sum(weights)   # 归一化
            
            ensemble_pred = predictions @ weights
            ensemble_pred = np.maximum(ensemble_pred, 1e-15)
            
            # 负对数似然 + L2正则化
            nll = -np.sum(np.log(ensemble_pred))
            regularization = self.penalty * np.sum(weights**2)
            
            return nll + regularization
        
        # 约束优化
        constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        bounds = [(0, 1) for _ in range(n_models)]
        
        initial_weights = np.ones(n_models) / n_models
        
        try:
            result = optimize.minimize(
                objective,
                initial_weights,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 1000}
            )
            
            if result.success:
                self.weights_ = result.x
            else:
                logger.warning("Stacking权重优化失败，使用等权重")
                self.weights_ = initial_weights
                
        except Exception as e:
            logger.warning(f"Stacking权重优化异常: {e}，使用等权重")
            self.weights_ = initial_weights
        
        logger.info(f"Stacking权重: {dict(zip([m.family for m in fitted_models], self.weights_))}")
    
    def survival_function(self, t: np.ndarray, X: np.ndarray) -> np.ndarray:
        """集成生存函数预测"""
        if not self.is_fitted:
            raise ModelFittingError("集成模型尚未拟合")
        
        # 获取所有模型的预测
        predictions = []
        for model in self.models:
            pred = model.survival_function(t, X)
            predictions.append(pred)
        
        # 加权平均
        ensemble_pred = np.zeros_like(predictions[0])
        for i, pred in enumerate(predictions):
            ensemble_pred += self.weights_[i] * pred
        
        return ensemble_pred
    
    def cumulative_density_function(self, t: np.ndarray, X: np.ndarray) -> np.ndarray:
        """集成累积分布函数预测"""
        return 1 - self.survival_function(t, X)
    
    def predict_percentiles(self, X: np.ndarray, 
                          percentiles: np.ndarray = np.array([0.1, 0.25, 0.5, 0.75, 0.9])
                          ) -> np.ndarray:
        """集成分位数预测"""
        if not self.is_fitted:
            raise ModelFittingError("集成模型尚未拟合")
        
        if self.ensemble_method == "best" and self.best_model_idx_ is not None:
            # 使用最佳单模型
            return self.models[self.best_model_idx_].predict_percentiles(X, percentiles)
        else:
            # 加权平均各模型的分位数预测
            all_quantiles = []
            for model in self.models:
                quantiles = model.predict_percentiles(X, percentiles)
                all_quantiles.append(quantiles)
            
            # 加权平均
            ensemble_quantiles = np.zeros_like(all_quantiles[0])
            for i, quantiles in enumerate(all_quantiles):
                ensemble_quantiles += self.weights_[i] * quantiles
            
            return ensemble_quantiles
    
    def get_model_weights(self) -> Dict[str, float]:
        """获取模型权重"""
        if not self.is_fitted:
            return {}
        
        return dict(zip([m.family for m in self.models], self.weights_))
    
    def get_ensemble_summary(self) -> Dict[str, Any]:
        """获取集成模型摘要"""
        summary = {
            "ensemble_method": self.ensemble_method,
            "n_models": len(self.models),
            "is_fitted": self.is_fitted
        }
        
        if self.is_fitted:
            summary.update({
                "model_families": [m.family for m in self.models],
                "weights": self.get_model_weights(),
                "best_model": self.models[self.best_model_idx_].family if self.best_model_idx_ is not None else None
            })
            
            # 添加各模型的信息准则
            summary["model_criteria"] = {}
            for model in self.models:
                summary["model_criteria"][model.family] = {
                    "AIC": model.aic_,
                    "BIC": model.bic_,
                    "loglik": model.loglik_
                }
        
        return summary


def create_aft_model(family: str) -> AFTModel:
    """
    创建AFT模型的工厂函数
    
    Args:
        family: 分布族名称
        
    Returns:
        AFT模型实例
    """
    if family == "lognormal":
        return LogNormalAFTModel()
    elif family == "loglogistic":
        return LogLogisticAFTModel()
    elif family == "weibull":
        return WeibullAFTModel()
    else:
        raise ValueError(f"不支持的分布族: {family}")


def fit_aft_models(families: List[str], X: np.ndarray, L: np.ndarray, 
                  R: np.ndarray, censor_type: np.ndarray,
                  ensemble_config: Optional[Dict[str, Any]] = None,
                  n_jobs: int = 1) -> Union[AFTModel, AFTEnsemble]:
    """
    拟合多个AFT模型并可选地创建集成
    
    Args:
        families: 分布族列表
        X: 协变量矩阵
        L: 左端点数组
        R: 右端点数组
        censor_type: 删失类型数组
        ensemble_config: 集成配置
        n_jobs: 并行作业数
        
    Returns:
        单个模型或集成模型
    """
    logger.info(f"拟合AFT模型: {families}")
    
    # 创建模型
    models = [create_aft_model(family) for family in families]
    
    if len(models) == 1:
        # 单模型
        models[0].fit(X, L, R, censor_type)
        return models[0]
    
    # 多模型集成
    ensemble_config = ensemble_config or {}
    ensemble = AFTEnsemble(models, **ensemble_config)
    ensemble.fit(X, L, R, censor_type)
    
    return ensemble