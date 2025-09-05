"""
网格搜索优化模块
基于双重网格搜索的BMI分组和最佳NIPT时点优化
"""

import numpy as np
import pandas as pd
from typing import Callable, Tuple, List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


def find_w_star_for_BMI(BMI: float, q_pred: Callable, p_fail: Callable, 
                       thr_adj: float, delta: float, w_min: float, w_max: float, 
                       step: float) -> float:
    """
    寻找特定BMI下的最佳检测时点
    
    Parameters:
    -----------
    BMI : float
        BMI值
    q_pred : callable
        浓度预测函数 q_pred(BMI, w, which="tau")
    p_fail : callable
        失败概率函数 p_fail(BMI, w)
    thr_adj : float
        调整后的浓度阈值（4%）
    delta : float
        失败风险阈值（10%）
    w_min : float
        最小孕周
    w_max : float
        最大孕周
    step : float
        搜索步长
        
    Returns:
    --------
    float
        最佳检测时点
    """
    logger.debug(f"为BMI={BMI:.2f}寻找最佳时点")
    
    # 创建孕周网格
    W = np.arange(w_min, w_max + 1e-9, step)
    
    # 遍历每个孕周点
    for w in W:
        try:
            # 条件1：浓度达标
            cond1 = (q_pred(BMI, w, which="tau") >= thr_adj)
            # 条件2：失败风险控制
            cond2 = (p_fail(BMI, w) <= delta)
            
            if cond1 and cond2:
                logger.debug(f"BMI={BMI:.2f}, 时点={w:.1f}满足条件")
                return float(w)
        except Exception as e:
            logger.warning(f"BMI={BMI:.2f}, 时点={w:.1f}计算失败: {e}")
            continue
    
    # 如果都不满足，返回最晚时点
    logger.warning(f"BMI={BMI:.2f}未找到满足条件的时点，返回最晚时点{w_max:.1f}")
    return float(W[-1])


def grid_search_w_star_curve(long_df: pd.DataFrame, q_pred: Callable, p_fail: Callable,
                            thr_adj: float = 0.04, delta: float = 0.1,
                            w_min: float = 8.0, w_max: float = 22.0, w_step: float = 0.5,
                            b_resolution: int = 40) -> pd.DataFrame:
    """
    使用网格搜索求解w*(b)曲线
    
    Parameters:
    -----------
    long_df : pd.DataFrame
        长表数据
    q_pred : callable
        浓度预测函数
    p_fail : callable
        失败概率函数
    thr_adj : float
        调整后的浓度阈值
    delta : float
        失败风险阈值
    w_min : float
        最小孕周
    w_max : float
        最大孕周
    w_step : float
        孕周搜索步长
    b_resolution : int
        BMI网格分辨率
        
    Returns:
    --------
    pd.DataFrame
        w*(b)曲线数据
    """
    logger.info("开始网格搜索求解w*(b)曲线")
    
    # 创建BMI网格
    B_min = np.nanpercentile(long_df["BMI_used"], 1)   # BMI下界（1%分位数）
    B_max = np.nanpercentile(long_df["BMI_used"], 99)  # BMI上界（99%分位数）
    B_grid = np.linspace(B_min, B_max, b_resolution)
    
    logger.info(f"BMI网格范围: {B_min:.2f} - {B_max:.2f}, 分辨率: {b_resolution}")
    logger.info(f"孕周搜索范围: {w_min} - {w_max}, 步长: {w_step}")
    
    # 在BMI网格上求w*(B)
    logger.info("正在计算BMI网格上的最佳时点...")
    records = []
    
    for i, B in enumerate(B_grid):
        if i % 10 == 0:
            logger.info(f"进度: {i+1}/{len(B_grid)}")
        
        w_star = find_w_star_for_BMI(B, q_pred, p_fail, thr_adj, delta, w_min, w_max, w_step)
        records.append({"BMI": B, "w_star": w_star})
    
    # 转换为DataFrame
    wstar_curve = pd.DataFrame(records)
    
    # 添加平滑后的w_star
    wstar_curve['w_star_smooth'] = wstar_curve['w_star']
    
    # 添加min_risk列（用于兼容性）
    wstar_curve['min_risk'] = 0.1  # 默认风险值
    
    logger.info(f"网格搜索完成，共计算{len(records)}个BMI点")
    
    return wstar_curve


def create_q_pred_function(mu_model, sigma_lookup, alpha=0.05):
    """
    创建浓度预测函数
    
    Parameters:
    -----------
    mu_model : object
        分位数回归模型 (LongitudinalModel or GAMQuantileModel)
    sigma_lookup : object
        σ查表器
    alpha : float
        置信水平
        
    Returns:
    --------
    callable
        浓度预测函数
    """
    def q_pred(BMI, w, which="tau"):
        """
        预测胎儿Y染色体浓度
        
        Parameters:
        -----------
        BMI : float
            BMI值
        w : float
            孕周
        which : str
            返回类型: "tau" (分位数), "mean" (均值)
            
        Returns:
        --------
        float
            预测的Y染色体浓度
        """
        try:
            # 使用分位数回归模型预测
            if hasattr(mu_model, 'predict'):
                # 检查是否为GAM模型
                if hasattr(mu_model, 'model') and hasattr(mu_model.model, 'predict'):
                    # GAM模型：直接使用 (week, BMI) 特征
                    features = np.array([[w, BMI]]).reshape(1, -1)
                    mu_pred = mu_model.predict(features)[0]
                else:
                    # 传统模型：使用多项式特征
                    features = np.array([[BMI, w, BMI*w, BMI**2, w**2]]).reshape(1, -1)
                    mu_pred = mu_model.predict(features)[0]
            else:
                # 简单的线性预测
                mu_pred = 0.01 + 0.001 * BMI + 0.002 * w
            
            # 获取σ值
            if hasattr(sigma_lookup, 'lookup'):
                sigma = sigma_lookup.lookup(BMI, w)
            else:
                sigma = 0.01  # 默认σ值
            
            if which == "tau":
                # 对于GAM模型，直接返回预测值（已经是分位数）
                if hasattr(mu_model, 'model') and hasattr(mu_model.model, 'predict'):
                    return max(0.0, mu_pred)
                else:
                    # 传统模型：返回调整后的分位数（考虑置信区间）
                    tau_adj = mu_pred - 1.96 * sigma  # 95%置信下界
                    return max(0.0, tau_adj)
            else:
                return mu_pred
                
        except Exception as e:
            logger.warning(f"浓度预测失败 BMI={BMI}, w={w}: {e}")
            return 0.0
    
    return q_pred


def create_p_fail_function(scenario_params):
    """
    创建失败概率函数
    
    Parameters:
    -----------
    scenario_params : dict
        情景参数
        
    Returns:
    --------
    callable
        失败概率函数
    """
    def p_fail(BMI, w):
        """
        预测检测失败概率
        
        Parameters:
        -----------
        BMI : float
            BMI值
        w : float
            孕周
            
        Returns:
        --------
        float
            失败概率
        """
        try:
            # 基础失败率
            base_fail_rate = scenario_params.get('base_fail_rate', 0.05)
            q_fail = base_fail_rate
            
            # 早孕周惩罚
            early_week_threshold = scenario_params.get('early_week_threshold', 15.0)
            if w < early_week_threshold:
                early_week_penalty = scenario_params.get('early_week_penalty', 0.02)
                q_fail += early_week_penalty * (early_week_threshold - w) / early_week_threshold
            
            # 高BMI惩罚
            high_bmi_threshold = scenario_params.get('high_bmi_threshold', 30.0)
            if BMI > high_bmi_threshold:
                high_bmi_penalty = scenario_params.get('high_bmi_penalty', 0.01)
                q_fail += high_bmi_penalty * (BMI - high_bmi_threshold) / high_bmi_threshold
            
            return min(1.0, max(0.0, q_fail))
            
        except Exception as e:
            logger.warning(f"失败概率预测失败 BMI={BMI}, w={w}: {e}")
            return 0.1  # 默认失败率
    
    return p_fail


def solve_w_star_curve_with_grid_search(long_df: pd.DataFrame, mu_model, sigma_lookup, 
                                       scenario_params: dict, 
                                       thr_adj: float = 0.04, delta: float = 0.1,
                                       w_min: float = 8.0, w_max: float = 22.0, 
                                       w_step: float = 0.5, b_resolution: int = 40) -> pd.DataFrame:
    """
    使用网格搜索求解w*(b)曲线（主接口函数）
    
    Parameters:
    -----------
    long_df : pd.DataFrame
        长表数据
    mu_model : object
        分位数回归模型
    sigma_lookup : object
        σ查表器
    scenario_params : dict
        情景参数
    thr_adj : float
        调整后的浓度阈值
    delta : float
        失败风险阈值
    w_min : float
        最小孕周
    w_max : float
        最大孕周
    w_step : float
        孕周搜索步长
    b_resolution : int
        BMI网格分辨率
        
    Returns:
    --------
    pd.DataFrame
        w*(b)曲线数据
    """
    logger.info("使用网格搜索方法求解w*(b)曲线")
    
    # 创建预测函数
    q_pred = create_q_pred_function(mu_model, sigma_lookup)
    p_fail = create_p_fail_function(scenario_params)
    
    # 执行网格搜索
    wstar_curve = grid_search_w_star_curve(
        long_df, q_pred, p_fail, thr_adj, delta, 
        w_min, w_max, w_step, b_resolution
    )
    
    return wstar_curve
