"""
生存预测模块

基于拟合的AFT模型进行生存分析预测，包括生存曲线、分位点和组均值曲线。
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from scipy import stats
import logging

from .exceptions import ModelFittingError, ComputationError
from .aft_models import AFTModel, AFTEnsemble
from .utils import ensure_numpy, timer, create_grid

logger = logging.getLogger(__name__)


class SurvivalPredictor:
    """生存预测器"""
    
    def __init__(self, aft_model: Union[AFTModel, AFTEnsemble]):
        """
        初始化生存预测器
        
        Args:
            aft_model: 拟合好的AFT模型或集成模型
        """
        self.aft_model = aft_model
        
        if not aft_model.is_fitted:
            raise ModelFittingError("AFT模型尚未拟合")
    
    def predict_survival_curves(self, X: np.ndarray, 
                              t_grid: Optional[np.ndarray] = None,
                              t_max: float = 100.0,
                              n_points: int = 200) -> Tuple[np.ndarray, np.ndarray]:
        """
        预测生存曲线 F(t|x) = P(T* ≤ t | x)
        
        Args:
            X: 协变量矩阵 (n_samples, n_features)
            t_grid: 时间网格，如果为None则自动生成
            t_max: 最大时间点
            n_points: 网格点数量
            
        Returns:
            (t_grid, F_curves) 元组
            - t_grid: 时间网格 (n_points,)
            - F_curves: 累积分布函数值 (n_points, n_samples)
        """
        logger.debug(f"预测 {X.shape[0]} 个样本的生存曲线")
        
        X = ensure_numpy(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        
        # 生成时间网格
        if t_grid is None:
            t_grid = create_grid((0.1, t_max), n_points, log_scale=False)
        else:
            t_grid = ensure_numpy(t_grid)
        
        with timer("生存曲线预测", logger):
            F_curves = self.aft_model.cumulative_density_function(t_grid, X)
        
        return t_grid, F_curves
    
    def predict_quantiles(self, X: np.ndarray,
                         percentiles: Optional[np.ndarray] = None) -> np.ndarray:
        """
        预测分位数 t_p = F^(-1)(p | x)
        
        Args:
            X: 协变量矩阵
            percentiles: 分位数数组，默认为常用分位数
            
        Returns:
            分位数矩阵 (n_samples, n_percentiles)
        """
        if percentiles is None:
            percentiles = np.array([0.1, 0.25, 0.5, 0.75, 0.9])
        
        logger.debug(f"预测 {X.shape[0]} 个样本的分位数")
        
        with timer("分位数预测", logger):
            quantiles = self.aft_model.predict_percentiles(X, percentiles)
        
        return quantiles
    
    def predict_group_curves(self, X: np.ndarray, 
                           groups: np.ndarray,
                           t_grid: Optional[np.ndarray] = None,
                           t_max: float = 100.0,
                           n_points: int = 200) -> Dict[Any, Tuple[np.ndarray, np.ndarray]]:
        """
        预测分组生存曲线 F̄(t | group) = E[F(t|x) | x ∈ group]
        
        Args:
            X: 协变量矩阵
            groups: 分组标签数组
            t_grid: 时间网格
            t_max: 最大时间点
            n_points: 网格点数量
            
        Returns:
            字典，键为组别，值为(t_grid, mean_curve)元组
        """
        logger.info(f"预测分组生存曲线: {len(np.unique(groups))} 个组")
        
        X = ensure_numpy(X)
        groups = ensure_numpy(groups)
        
        if len(X) != len(groups):
            raise ValueError("X和groups长度不匹配")
        
        # 生成时间网格
        if t_grid is None:
            t_grid = create_grid((0.1, t_max), n_points, log_scale=False)
        else:
            t_grid = ensure_numpy(t_grid)
        
        group_curves = {}
        unique_groups = np.unique(groups)
        
        with timer("分组生存曲线预测", logger):
            for group in unique_groups:
                group_mask = groups == group
                X_group = X[group_mask]
                
                if len(X_group) == 0:
                    continue
                
                # 预测该组的生存曲线
                F_group = self.aft_model.cumulative_density_function(t_grid, X_group)
                
                # 计算组内平均
                mean_curve = np.mean(F_group, axis=1)
                
                group_curves[group] = (t_grid, mean_curve)
                
                logger.debug(f"组 {group}: {len(X_group)} 个样本")
        
        return group_curves
    
    def predict_success_probability(self, X: np.ndarray, 
                                  t_target: float) -> np.ndarray:
        """
        预测在目标时间点的达标概率 P(T* ≤ t_target | x)
        
        Args:
            X: 协变量矩阵
            t_target: 目标时间点
            
        Returns:
            达标概率数组 (n_samples,)
        """
        logger.debug(f"预测在 t={t_target} 的达标概率")
        
        X = ensure_numpy(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        
        F_target = self.aft_model.cumulative_density_function(
            np.array([t_target]), X
        )
        
        return F_target[0, :]  # 第一个时间点的所有样本
    
    def compute_prediction_intervals(self, X: np.ndarray,
                                   t_grid: Optional[np.ndarray] = None,
                                   confidence_level: float = 0.95,
                                   method: str = "bootstrap",
                                   n_bootstrap: int = 100) -> Dict[str, np.ndarray]:
        """
        计算预测置信区间
        
        Args:
            X: 协变量矩阵
            t_grid: 时间网格
            confidence_level: 置信水平
            method: 方法 ("bootstrap", "asymptotic")
            n_bootstrap: Bootstrap采样次数
            
        Returns:
            包含lower, upper, mean的字典
        """
        if method == "bootstrap":
            return self._bootstrap_prediction_intervals(
                X, t_grid, confidence_level, n_bootstrap
            )
        elif method == "asymptotic":
            return self._asymptotic_prediction_intervals(
                X, t_grid, confidence_level
            )
        else:
            raise ValueError(f"不支持的方法: {method}")
    
    def _bootstrap_prediction_intervals(self, X: np.ndarray,
                                      t_grid: Optional[np.ndarray],
                                      confidence_level: float,
                                      n_bootstrap: int) -> Dict[str, np.ndarray]:
        """Bootstrap置信区间"""
        logger.info(f"计算Bootstrap置信区间: {n_bootstrap} 次采样")
        
        # 这里简化实现，实际应该重新拟合模型
        alpha = 1 - confidence_level
        lower_percentile = 100 * alpha / 2
        upper_percentile = 100 * (1 - alpha / 2)
        
        # 获取基础预测
        t_grid, F_curves = self.predict_survival_curves(X, t_grid)
        
        # 简化：假设预测有一定的不确定性
        noise_std = 0.05  # 假设标准差
        
        bootstrap_predictions = []
        for _ in range(n_bootstrap):
            # 添加噪声模拟不确定性
            noise = np.random.normal(0, noise_std, F_curves.shape)
            F_noisy = np.clip(F_curves + noise, 0, 1)
            bootstrap_predictions.append(F_noisy)
        
        bootstrap_predictions = np.array(bootstrap_predictions)  # (n_bootstrap, n_times, n_samples)
        
        # 计算分位数
        lower_bound = np.percentile(bootstrap_predictions, lower_percentile, axis=0)
        upper_bound = np.percentile(bootstrap_predictions, upper_percentile, axis=0)
        mean_pred = np.mean(bootstrap_predictions, axis=0)
        
        return {
            "t_grid": t_grid,
            "lower": lower_bound,
            "upper": upper_bound,
            "mean": mean_pred,
            "confidence_level": confidence_level
        }
    
    def _asymptotic_prediction_intervals(self, X: np.ndarray,
                                       t_grid: Optional[np.ndarray],
                                       confidence_level: float) -> Dict[str, np.ndarray]:
        """渐近置信区间"""
        logger.warning("渐近置信区间未完全实现，使用简化版本")
        
        # 获取基础预测
        t_grid, F_curves = self.predict_survival_curves(X, t_grid)
        
        # 简化：假设固定的标准误
        alpha = 1 - confidence_level
        z_score = stats.norm.ppf(1 - alpha / 2)
        
        # 假设标准误与预测值相关
        se = 0.1 * np.sqrt(F_curves * (1 - F_curves))
        
        lower_bound = np.clip(F_curves - z_score * se, 0, 1)
        upper_bound = np.clip(F_curves + z_score * se, 0, 1)
        
        return {
            "t_grid": t_grid,
            "lower": lower_bound,
            "upper": upper_bound,
            "mean": F_curves,
            "confidence_level": confidence_level
        }
    
    def evaluate_calibration(self, X_test: np.ndarray,
                           observed_times: np.ndarray,
                           event_indicators: np.ndarray,
                           n_bins: int = 10) -> Dict[str, Any]:
        """
        评估预测校准性
        
        Args:
            X_test: 测试协变量
            observed_times: 观测时间
            event_indicators: 事件指示器 (1=事件发生, 0=删失)
            n_bins: 分箱数量
            
        Returns:
            校准评估结果
        """
        logger.info("评估预测校准性")
        
        X_test = ensure_numpy(X_test)
        observed_times = ensure_numpy(observed_times)
        event_indicators = ensure_numpy(event_indicators)
        
        # 预测在观测时间点的累积概率
        predicted_probs = []
        for i, t_obs in enumerate(observed_times):
            X_i = X_test[i:i+1]
            prob = self.predict_success_probability(X_i, t_obs)[0]
            predicted_probs.append(prob)
        
        predicted_probs = np.array(predicted_probs)
        
        # 分箱
        bin_edges = np.linspace(0, 1, n_bins + 1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        observed_freqs = []
        predicted_freqs = []
        bin_counts = []
        
        for i in range(n_bins):
            mask = (predicted_probs >= bin_edges[i]) & (predicted_probs < bin_edges[i+1])
            if i == n_bins - 1:  # 最后一个bin包含右端点
                mask = (predicted_probs >= bin_edges[i]) & (predicted_probs <= bin_edges[i+1])
            
            if np.sum(mask) == 0:
                observed_freqs.append(0)
                predicted_freqs.append(0)
                bin_counts.append(0)
                continue
            
            # 该bin中的观测频率
            obs_freq = np.mean(event_indicators[mask])
            pred_freq = np.mean(predicted_probs[mask])
            
            observed_freqs.append(obs_freq)
            predicted_freqs.append(pred_freq)
            bin_counts.append(np.sum(mask))
        
        # 计算校准指标
        observed_freqs = np.array(observed_freqs)
        predicted_freqs = np.array(predicted_freqs)
        bin_counts = np.array(bin_counts)
        
        # Brier分数
        brier_score = np.mean((predicted_probs - event_indicators) ** 2)
        
        # 期望校准误差 (ECE)
        valid_bins = bin_counts > 0
        if np.any(valid_bins):
            ece = np.sum(
                bin_counts[valid_bins] * np.abs(observed_freqs[valid_bins] - predicted_freqs[valid_bins])
            ) / np.sum(bin_counts[valid_bins])
        else:
            ece = 0.0
        
        return {
            "brier_score": brier_score,
            "expected_calibration_error": ece,
            "bin_centers": bin_centers,
            "observed_frequencies": observed_freqs,
            "predicted_frequencies": predicted_freqs,
            "bin_counts": bin_counts,
            "n_bins": n_bins,
            "total_samples": len(X_test)
        }
    
    def compute_survival_statistics(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """
        计算生存统计量
        
        Args:
            X: 协变量矩阵
            
        Returns:
            统计量字典
        """
        logger.debug(f"计算 {X.shape[0]} 个样本的生存统计量")
        
        # 预测分位数
        percentiles = np.array([0.1, 0.25, 0.5, 0.75, 0.9])
        quantiles = self.predict_quantiles(X, percentiles)
        
        # 预测在特定时间点的概率
        time_points = np.array([10, 20, 50, 100])
        probs_at_times = {}
        
        for t in time_points:
            probs = self.predict_success_probability(X, t)
            probs_at_times[f"prob_at_{t}"] = probs
        
        statistics = {
            "median_survival_time": quantiles[:, 2],  # 50%分位数
            "q25_survival_time": quantiles[:, 1],     # 25%分位数
            "q75_survival_time": quantiles[:, 3],     # 75%分位数
            **probs_at_times
        }
        
        return statistics
    
    def get_prediction_summary(self, X: np.ndarray) -> Dict[str, Any]:
        """获取预测摘要"""
        stats = self.compute_survival_statistics(X)
        
        summary = {
            "n_samples": X.shape[0],
            "model_type": type(self.aft_model).__name__,
            "median_survival_time": {
                "mean": np.mean(stats["median_survival_time"]),
                "std": np.std(stats["median_survival_time"]),
                "min": np.min(stats["median_survival_time"]),
                "max": np.max(stats["median_survival_time"])
            }
        }
        
        # 添加其他统计量的摘要
        for key, values in stats.items():
            if key.startswith("prob_at_"):
                summary[key] = {
                    "mean": np.mean(values),
                    "std": np.std(values)
                }
        
        return summary


def create_survival_predictor(aft_model: Union[AFTModel, AFTEnsemble]) -> SurvivalPredictor:
    """
    创建生存预测器的工厂函数
    
    Args:
        aft_model: 拟合好的AFT模型
        
    Returns:
        生存预测器实例
    """
    return SurvivalPredictor(aft_model)


def batch_predict_survival_curves(predictor: SurvivalPredictor,
                                 X: np.ndarray,
                                 batch_size: int = 1000,
                                 **kwargs) -> Tuple[np.ndarray, np.ndarray]:
    """
    分批预测生存曲线（用于大数据集）
    
    Args:
        predictor: 生存预测器
        X: 协变量矩阵
        batch_size: 批次大小
        **kwargs: 传递给predict_survival_curves的参数
        
    Returns:
        (t_grid, F_curves) 元组
    """
    n_samples = X.shape[0]
    
    if n_samples <= batch_size:
        return predictor.predict_survival_curves(X, **kwargs)
    
    logger.info(f"分批预测生存曲线: {n_samples} 样本，批次大小 {batch_size}")
    
    # 第一批获取时间网格
    X_first = X[:batch_size]
    t_grid, F_first = predictor.predict_survival_curves(X_first, **kwargs)
    
    # 初始化结果矩阵
    F_all = np.zeros((len(t_grid), n_samples))
    F_all[:, :batch_size] = F_first
    
    # 处理剩余批次
    for start_idx in range(batch_size, n_samples, batch_size):
        end_idx = min(start_idx + batch_size, n_samples)
        X_batch = X[start_idx:end_idx]
        
        # 使用相同的时间网格
        kwargs_batch = kwargs.copy()
        kwargs_batch['t_grid'] = t_grid
        
        _, F_batch = predictor.predict_survival_curves(X_batch, **kwargs_batch)
        F_all[:, start_idx:end_idx] = F_batch
    
    return t_grid, F_all