"""
σ(t,b) 查表/插值模块
优先局部网格，回退全局σ
"""

import pandas as pd
import numpy as np
from typing import Callable, Optional, Dict, Any, Tuple
from scipy.interpolate import griddata, interp2d
from scipy.spatial.distance import cdist
import logging

logger = logging.getLogger(__name__)


class SigmaLookup:
    """
    σ(t,b) 查表/插值类
    """
    
    def __init__(self, global_sigma: float, local_sigma_grid: Optional[pd.DataFrame] = None,
                 shrinkage_lambda: float = 0.1, interpolation_method: str = "linear"):
        """
        初始化σ查表器
        
        Parameters:
        -----------
        global_sigma : float
            全局σ值
        local_sigma_grid : pd.DataFrame, optional
            局部σ网格，包含列 ['t', 'b', 'sigma', 'n_samples']
        shrinkage_lambda : float
            向全局σ收缩的权重
        interpolation_method : str
            插值方法: 'linear', 'cubic', 'nearest'
        """
        self.global_sigma = global_sigma
        self.local_sigma_grid = local_sigma_grid
        self.shrinkage_lambda = shrinkage_lambda
        self.interpolation_method = interpolation_method
        
        # 如果有局部网格，构建插值器
        self.interpolator = None
        if local_sigma_grid is not None and len(local_sigma_grid) > 0:
            self._build_interpolator()
        
        logger.info(f"初始化σ查表器: 全局σ={global_sigma:.6f}, "
                   f"局部网格点数={len(local_sigma_grid) if local_sigma_grid is not None else 0}")
    
    def _build_interpolator(self):
        """构建插值器"""
        if self.local_sigma_grid is None or len(self.local_sigma_grid) == 0:
            return
        
        # 提取网格点
        t_points = self.local_sigma_grid['t'].values
        b_points = self.local_sigma_grid['b'].values
        sigma_points = self.local_sigma_grid['sigma'].values
        n_samples = self.local_sigma_grid.get('n_samples', np.ones(len(sigma_points))).values
        
        # 应用收缩估计
        sigma_shrunk = (1 - self.shrinkage_lambda) * sigma_points + self.shrinkage_lambda * self.global_sigma
        
        # 构建插值器
        try:
            self.interpolator = interp2d(t_points, b_points, sigma_shrunk, 
                                       kind=self.interpolation_method, bounds_error=False, fill_value=None)
            logger.info(f"构建插值器成功: {len(t_points)} 个网格点")
        except Exception as e:
            logger.warning(f"插值器构建失败: {e}, 将使用全局σ")
            self.interpolator = None
    
    def __call__(self, t: float, b: float) -> float:
        """
        查询σ(t,b)
        
        Parameters:
        -----------
        t : float
            孕周
        b : float
            BMI
            
        Returns:
        --------
        float
            σ值
        """
        # 如果有插值器，尝试局部插值
        if self.interpolator is not None:
            try:
                sigma_local = self.interpolator(t, b)
                if not np.isnan(sigma_local) and sigma_local > 0:
                    return float(sigma_local)
            except Exception as e:
                logger.debug(f"局部插值失败 ({t}, {b}): {e}")
        
        # 回退到全局σ
        return self.global_sigma
    
    def get_sigma_grid(self, t_range: Tuple[float, float], b_range: Tuple[float, float], 
                      resolution: int = 50) -> np.ndarray:
        """
        获取σ网格用于可视化
        
        Parameters:
        -----------
        t_range : tuple
            孕周范围 (t_min, t_max)
        b_range : tuple
            BMI范围 (b_min, b_max)
        resolution : int
            网格分辨率
            
        Returns:
        --------
        np.ndarray
            σ网格值
        """
        t_grid = np.linspace(t_range[0], t_range[1], resolution)
        b_grid = np.linspace(b_range[0], b_range[1], resolution)
        T, B = np.meshgrid(t_grid, b_grid)
        
        sigma_grid = np.zeros_like(T)
        for i in range(resolution):
            for j in range(resolution):
                sigma_grid[i, j] = self(t_grid[j], b_grid[i])
        
        return sigma_grid, t_grid, b_grid


def build_sigma_lookup(report: pd.DataFrame, sigma_grid_path: Optional[str] = None,
                      shrinkage_lambda: float = 0.1, interpolation_method: str = "linear") -> SigmaLookup:
    """
    构建σ查表器
    
    Parameters:
    -----------
    report : pd.DataFrame
        处理报告，包含σ估计信息
    sigma_grid_path : str, optional
        局部σ网格文件路径
    shrinkage_lambda : float
        收缩权重
    interpolation_method : str
        插值方法
        
    Returns:
    --------
    SigmaLookup
        σ查表器实例
    """
    logger.info("构建σ查表器...")
    
    # 从报告中提取全局σ
    error_info_str = report.iloc[-1]['检测误差建模']
    import ast
    try:
        error_info = ast.literal_eval(error_info_str)
        global_sigma = error_info.get('Y浓度残差标准差', 0.0059)
    except:
        logger.warning("无法解析报告中的σ信息，使用默认值")
        global_sigma = 0.0059
    
    # 尝试加载局部σ网格
    local_sigma_grid = None
    if sigma_grid_path and os.path.exists(sigma_grid_path):
        try:
            local_sigma_grid = pd.read_csv(sigma_grid_path)
            logger.info(f"加载局部σ网格: {len(local_sigma_grid)} 个点")
        except Exception as e:
            logger.warning(f"加载局部σ网格失败: {e}")
    else:
        logger.info("未提供局部σ网格路径，将使用全局σ")
    
    # 创建查表器
    lookup = SigmaLookup(
        global_sigma=global_sigma,
        local_sigma_grid=local_sigma_grid,
        shrinkage_lambda=shrinkage_lambda,
        interpolation_method=interpolation_method
    )
    
    return lookup


def estimate_local_sigma(long_df: pd.DataFrame, t_bins: int = 5, b_bins: int = 3,
                        threshold_range: Tuple[float, float] = (0.03, 0.05)) -> pd.DataFrame:
    """
    估计局部σ网格
    
    Parameters:
    -----------
    long_df : pd.DataFrame
        长表数据
    t_bins : int
        孕周分箱数
    b_bins : int
        BMI分箱数
    threshold_range : tuple
        阈值附近范围
        
    Returns:
    --------
    pd.DataFrame
        局部σ网格
    """
    logger.info(f"估计局部σ网格: {t_bins}×{b_bins} 分箱")
    
    # 筛选阈值附近的样本
    threshold_mask = (long_df['Y_frac'] >= threshold_range[0]) & (long_df['Y_frac'] <= threshold_range[1])
    threshold_data = long_df[threshold_mask].copy()
    
    if len(threshold_data) == 0:
        logger.warning("阈值附近无样本，返回空网格")
        return pd.DataFrame(columns=['t', 'b', 'sigma', 'n_samples'])
    
    # 创建分箱
    t_edges = np.linspace(long_df['week'].min(), long_df['week'].max(), t_bins + 1)
    b_edges = np.linspace(long_df['BMI_used'].min(), long_df['BMI_used'].max(), b_bins + 1)
    
    threshold_data['t_bin'] = pd.cut(threshold_data['week'], bins=t_edges, include_lowest=True)
    threshold_data['b_bin'] = pd.cut(threshold_data['BMI_used'], bins=b_edges, include_lowest=True)
    
    # 计算每个分箱的σ
    sigma_grid = []
    for t_bin in threshold_data['t_bin'].cat.categories:
        for b_bin in threshold_data['b_bin'].cat.categories:
            mask = (threshold_data['t_bin'] == t_bin) & (threshold_data['b_bin'] == b_bin)
            bin_data = threshold_data[mask]
            
            if len(bin_data) >= 3:  # 至少3个样本才估计σ
                # 计算残差（假设真实值为阈值）
                residuals = bin_data['Y_frac'] - 0.04
                sigma = np.std(residuals)
                
                sigma_grid.append({
                    't': t_bin.mid,
                    'b': b_bin.mid,
                    'sigma': sigma,
                    'n_samples': len(bin_data)
                })
    
    sigma_df = pd.DataFrame(sigma_grid)
    logger.info(f"局部σ网格构建完成: {len(sigma_df)} 个有效分箱")
    
    return sigma_df


def validate_sigma_lookup(sigma_lookup: SigmaLookup, test_points: Optional[list] = None) -> Dict[str, Any]:
    """
    验证σ查表器的有效性
    
    Parameters:
    -----------
    sigma_lookup : SigmaLookup
        σ查表器
    test_points : list, optional
        测试点列表 [(t1, b1), (t2, b2), ...]
        
    Returns:
    --------
    dict
        验证结果
    """
    if test_points is None:
        test_points = [(12, 22), (16, 25), (20, 30), (24, 35)]
    
    validation_results = {
        'test_points': test_points,
        'sigma_values': [],
        'is_positive': True,
        'is_finite': True,
        'range': [float('inf'), -float('inf')]
    }
    
    for t, b in test_points:
        sigma = sigma_lookup(t, b)
        validation_results['sigma_values'].append(sigma)
        
        if sigma <= 0:
            validation_results['is_positive'] = False
        if not np.isfinite(sigma):
            validation_results['is_finite'] = False
        
        validation_results['range'][0] = min(validation_results['range'][0], sigma)
        validation_results['range'][1] = max(validation_results['range'][1], sigma)
    
    validation_results['is_valid'] = (validation_results['is_positive'] and 
                                    validation_results['is_finite'])
    
    logger.info(f"σ查表器验证: 有效={validation_results['is_valid']}, "
               f"范围=[{validation_results['range'][0]:.6f}, {validation_results['range'][1]:.6f}]")
    
    return validation_results


def create_threshold_adjustment_function(sigma_lookup: SigmaLookup, alpha: float = 0.05) -> Callable:
    """
    创建阈值调整函数
    
    Parameters:
    -----------
    sigma_lookup : SigmaLookup
        σ查表器
    alpha : float
        置信水平
        
    Returns:
    --------
    callable
        阈值调整函数 thr_adj(t, b)
    """
    from scipy.stats import norm
    z_alpha = norm.ppf(1 - alpha)
    
    def thr_adj(t: float, b: float) -> float:
        """
        调整后的阈值
        
        Parameters:
        -----------
        t : float
            孕周
        b : float
            BMI
            
        Returns:
        --------
        float
            调整后的阈值
        """
        sigma = sigma_lookup(t, b)
        return 0.04 + z_alpha * sigma
    
    return thr_adj
