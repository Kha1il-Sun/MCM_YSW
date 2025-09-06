"""
基于经验数据的理论模型
结合实际数据拟合的数学关系，构建可解释的理论框架
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any
import logging

logger = logging.getLogger(__name__)


class EmpiricalDetectionModel:
    """
    基于经验数据的检测时点理论模型
    
    核心数学公式：
    t_optimal(BMI) = α × ln(BMI) + β + γ × σ + δ × (BMI - BMI_ref)²
    
    其中：
    - α, β: 基于实际数据拟合的对数关系参数
    - γ: 个体变异修正系数
    - δ: BMI偏离修正系数
    - σ: 个体变异系数
    """
    
    def __init__(self, 
                 alpha: float = 11.358,
                 beta: float = -24.261,
                 gamma: float = 0.5,
                 delta: float = 0.01,
                 bmi_ref: float = 30.0,
                 t_min: float = 12.0,
                 t_max: float = 25.0):
        """
        初始化经验检测模型
        
        Parameters:
        -----------
        alpha : float
            对数关系系数（基于实际数据拟合）
        beta : float
            对数关系截距（基于实际数据拟合）
        gamma : float
            个体变异修正系数
        delta : float
            BMI偏离修正系数
        bmi_ref : float
            参考BMI值
        t_min : float
            最小检测时点
        t_max : float
            最大检测时点
        """
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.bmi_ref = bmi_ref
        self.t_min = t_min
        self.t_max = t_max
        
        logger.info(f"初始化经验检测模型: α={alpha:.3f}, β={beta:.3f}")
    
    def predict_base_time(self, BMI: float) -> float:
        """
        预测基础检测时点
        
        数学公式：t_base(BMI) = α × ln(BMI) + β
        
        Parameters:
        -----------
        BMI : float
            BMI值
            
        Returns:
        --------
        float
            基础检测时点
        """
        if BMI <= 0:
            raise ValueError("BMI必须大于0")
        
        t_base = self.alpha * np.log(BMI) + self.beta
        return float(t_base)
    
    def predict_optimal_time(self, BMI: float, sigma: float = 0.0) -> float:
        """
        预测最优检测时点（考虑修正因子）
        
        数学公式：t_optimal(BMI, σ) = t_base(BMI) + γ × σ + δ × (BMI - BMI_ref)²
        
        Parameters:
        -----------
        BMI : float
            BMI值
        sigma : float
            个体变异系数
            
        Returns:
        --------
        float
            最优检测时点
        """
        # 基础时点
        t_base = self.predict_base_time(BMI)
        
        # 个体变异修正
        sigma_correction = self.gamma * sigma
        
        # BMI偏离修正
        bmi_deviation = BMI - self.bmi_ref
        bmi_correction = self.delta * (bmi_deviation ** 2)
        
        # 总修正
        t_optimal = t_base + sigma_correction + bmi_correction
        
        # 应用临床约束
        t_optimal = np.clip(t_optimal, self.t_min, self.t_max)
        
        return float(t_optimal)
    
    def predict_confidence_interval(self, BMI: float, sigma: float = 0.0, 
                                  confidence_level: float = 0.95) -> Tuple[float, float]:
        """
        预测置信区间
        
        数学公式：CI(t) = t_optimal(BMI, σ) ± z_α × SE(t)
        
        Parameters:
        -----------
        BMI : float
            BMI值
        sigma : float
            个体变异系数
        confidence_level : float
            置信水平
            
        Returns:
        --------
        Tuple[float, float]
            (下界, 上界)
        """
        t_optimal = self.predict_optimal_time(BMI, sigma)
        
        # 计算标准误差（基于经验数据）
        se = 2.0  # 基于实际数据的标准差
        
        # 计算置信区间
        z_alpha = 1.96 if confidence_level == 0.95 else 2.576
        margin_error = z_alpha * se
        
        lower_bound = max(t_optimal - margin_error, self.t_min)
        upper_bound = min(t_optimal + margin_error, self.t_max)
        
        return float(lower_bound), float(upper_bound)
    
    def predict_group_time(self, bmi_groups: Dict[str, Tuple[float, float]]) -> Dict[str, float]:
        """
        预测BMI分组的检测时点
        
        Parameters:
        -----------
        bmi_groups : Dict[str, Tuple[float, float]]
            BMI分组字典，格式：{组名: (BMI_min, BMI_max)}
            
        Returns:
        --------
        Dict[str, float]
            各组的推荐检测时点
        """
        group_times = {}
        
        for group_name, (bmi_min, bmi_max) in bmi_groups.items():
            # 使用组内BMI中位数
            bmi_median = (bmi_min + bmi_max) / 2
            optimal_time = self.predict_optimal_time(bmi_median)
            group_times[group_name] = optimal_time
            
        return group_times
    
    def fit_parameters(self, bmi_data: np.ndarray, time_data: np.ndarray) -> Dict[str, float]:
        """
        基于新数据拟合参数（便于后续迭代）
        
        Parameters:
        -----------
        bmi_data : np.ndarray
            BMI数据
        time_data : np.ndarray
            检测时点数据
            
        Returns:
        --------
        Dict[str, float]
            拟合的参数
        """
        # 对数拟合
        log_bmi = np.log(bmi_data)
        coeffs = np.polyfit(log_bmi, time_data, 1)
        
        new_alpha = coeffs[0]
        new_beta = coeffs[1]
        
        # 更新参数
        self.alpha = new_alpha
        self.beta = new_beta
        
        logger.info(f"参数更新: α={new_alpha:.3f}, β={new_beta:.3f}")
        
        return {
            'alpha': new_alpha,
            'beta': new_beta,
            'gamma': self.gamma,
            'delta': self.delta
        }
    
    def evaluate_model(self, bmi_data: np.ndarray, time_data: np.ndarray) -> Dict[str, float]:
        """
        评估模型性能
        
        Parameters:
        -----------
        bmi_data : np.ndarray
            实际BMI数据
        time_data : np.ndarray
            实际检测时点数据
            
        Returns:
        --------
        Dict[str, float]
            性能指标
        """
        predicted_times = [self.predict_optimal_time(bmi) for bmi in bmi_data]
        
        # 计算误差指标
        mae = np.mean(np.abs(np.array(predicted_times) - time_data))
        mse = np.mean((np.array(predicted_times) - time_data) ** 2)
        rmse = np.sqrt(mse)
        
        # 计算相关系数
        correlation = np.corrcoef(predicted_times, time_data)[0, 1]
        
        return {
            'mae': mae,
            'mse': mse,
            'rmse': rmse,
            'correlation': correlation
        }
    
    def get_model_parameters(self) -> Dict[str, float]:
        """
        获取当前模型参数
        
        Returns:
        --------
        Dict[str, float]
            模型参数
        """
        return {
            'alpha': self.alpha,
            'beta': self.beta,
            'gamma': self.gamma,
            'delta': self.delta,
            'bmi_ref': self.bmi_ref,
            't_min': self.t_min,
            't_max': self.t_max
        }


def create_empirical_model_from_data(long_df: pd.DataFrame) -> EmpiricalDetectionModel:
    """
    基于实际数据创建经验模型
    
    Parameters:
    -----------
    long_df : pd.DataFrame
        长表数据
        
    Returns:
    --------
    EmpiricalDetectionModel
        经验检测模型
    """
    # 定义BMI分组
    bins = [20.0, 30.5, 32.7, 34.4, 50.0]
    labels = ['低BMI组', '中BMI组', '高BMI组', '极高BMI组']
    long_df['bmi_group'] = pd.cut(long_df['BMI_used'], bins=bins, labels=labels)
    
    # 计算各组的首次达标时点
    first_hit = long_df[long_df['Y_frac'] > 0.04].groupby(['id', 'bmi_group'])['week'].min().reset_index()
    group_stats = first_hit.groupby('bmi_group')['week'].mean()
    
    # 提取BMI中位数和时点数据
    bmi_data = []
    time_data = []
    
    for group in labels:
        group_data = first_hit[first_hit['bmi_group'] == group]
        if len(group_data) > 0:
            bmi_median = long_df[long_df['bmi_group'] == group]['BMI_used'].median()
            time_mean = group_data['week'].mean()
            bmi_data.append(bmi_median)
            time_data.append(time_mean)
    
    # 创建模型
    model = EmpiricalDetectionModel()
    
    # 拟合参数
    model.fit_parameters(np.array(bmi_data), np.array(time_data))
    
    return model
