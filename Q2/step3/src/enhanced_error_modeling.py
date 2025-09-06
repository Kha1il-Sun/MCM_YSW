"""
增强的检测误差建模模块
实现多维度、自适应的检测误差估计
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, HuberRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from scipy import stats
from scipy.interpolate import griddata, RBFInterpolator
from scipy.spatial.distance import cdist
from typing import Dict, List, Tuple, Optional, Any, Callable
import logging
import warnings

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


class EnhancedSigmaEstimator:
    """增强的检测误差估计器"""
    
    def __init__(self, config: dict, report_df: pd.DataFrame = None):
        """
        初始化误差估计器
        
        Parameters:
        -----------
        config : dict
            误差建模配置
        report_df : pd.DataFrame, optional
            Step1报告数据
        """
        self.config = config
        self.report_df = report_df
        
        # 配置参数
        self.multi_dimensional = config.get('multi_dimensional', True)
        self.error_factors = config.get('error_factors', ['week', 'BMI_used'])
        self.local_config = config.get('local_estimation', {})
        self.shrinkage_config = config.get('shrinkage', {})
        self.dynamic_threshold_config = config.get('dynamic_threshold', {})
        
        # 估计结果
        self.global_sigma = None
        self.local_sigma_grid = None
        self.sigma_model = None
        self.local_sigma_func = None
        self.shrinkage_func = None
        self.threshold_adjustment_func = None
        
        # 从报告中提取全局sigma
        if report_df is not None:
            self._extract_global_sigma()
        
        logger.info(f"增强误差估计器初始化完成")
        logger.info(f"  - 多维度建模: {self.multi_dimensional}")
        logger.info(f"  - 误差影响因子: {self.error_factors}")
        logger.info(f"  - 全局sigma: {self.global_sigma}")
    
    def fit(self, df: pd.DataFrame, target_col: str = 'Y_frac') -> Dict[str, Any]:
        """
        拟合误差模型
        
        Parameters:
        -----------
        df : pd.DataFrame
            数据
        target_col : str
            目标列
            
        Returns:
        --------
        dict
            误差模型结果
        """
        logger.info("开始拟合增强误差模型...")
        
        # 1. 提取阈值附近数据
        threshold_data = self._extract_threshold_nearby_data(df, target_col)
        
        if len(threshold_data) < 20:
            logger.warning("阈值附近数据不足，使用全局估计")
            return self._create_fallback_model(df, target_col)
        
        # 2. 计算全局sigma
        if self.global_sigma is None:
            self.global_sigma = self._compute_global_sigma(threshold_data, target_col)
        
        # 3. 多维度局部sigma建模
        if self.multi_dimensional:
            self._fit_multidimensional_sigma(threshold_data, target_col)
        
        # 4. 构建局部sigma函数
        self._build_local_sigma_function(threshold_data)
        
        # 5. 构建收缩函数
        if self.shrinkage_config.get('enabled', True):
            self._build_shrinkage_function(threshold_data)
        
        # 6. 构建动态阈值调整函数
        if self.dynamic_threshold_config.get('enabled', True):
            self._build_threshold_adjustment_function()
        
        # 7. 评估误差模型
        model_performance = self._evaluate_error_model(threshold_data, target_col)
        
        # 构建返回结果
        results = {
            'global_sigma': self.global_sigma,
            'local_sigma_func': self.local_sigma_func,
            'shrinkage_func': self.shrinkage_func,
            'threshold_adjustment_func': self.threshold_adjustment_func,
            'local_factors': self.error_factors,
            'model_performance': model_performance,
            'n_threshold_samples': len(threshold_data),
            'sigma_model': self.sigma_model
        }
        
        logger.info("增强误差模型拟合完成")
        logger.info(f"  - 阈值附近样本: {len(threshold_data)}")
        logger.info(f"  - 全局sigma: {self.global_sigma:.4f}")
        logger.info(f"  - 模型性能: R² = {model_performance.get('r2', 'N/A')}")
        
        return results
    
    def _extract_global_sigma(self):
        """从报告中提取全局sigma"""
        
        if self.report_df is not None and not self.report_df.empty:
            # 查找检测误差相关字段
            error_fields = ['检测误差建模', 'Y浓度残差标准差', 'sigma_global']
            
            for _, row in self.report_df.iterrows():
                for field in error_fields:
                    if field in row:
                        try:
                            if isinstance(row[field], dict):
                                self.global_sigma = row[field].get('Y浓度残差标准差', None)
                            elif isinstance(row[field], (int, float)):
                                self.global_sigma = float(row[field])
                            
                            if self.global_sigma is not None:
                                logger.info(f"从报告提取全局sigma: {self.global_sigma:.4f}")
                                return
                        except:
                            continue
    
    def _extract_threshold_nearby_data(self, df: pd.DataFrame, target_col: str) -> pd.DataFrame:
        """提取阈值附近数据"""
        
        threshold = 0.04
        nearby_range = [0.03, 0.05]  # 3%-5%范围
        
        mask = (df[target_col] >= nearby_range[0]) & (df[target_col] <= nearby_range[1])
        nearby_data = df[mask].copy()
        
        logger.info(f"提取阈值附近数据: {len(nearby_data)} 样本 (范围: {nearby_range})")
        
        return nearby_data
    
    def _compute_global_sigma(self, data: pd.DataFrame, target_col: str) -> float:
        """计算全局sigma"""
        
        threshold = 0.04
        residuals = data[target_col] - threshold
        global_sigma = np.std(residuals)
        
        # 使用更稳健的估计
        mad = stats.median_abs_deviation(residuals)
        robust_sigma = mad * 1.4826  # MAD到标准差的转换因子
        
        # 取平均值
        final_sigma = (global_sigma + robust_sigma) / 2
        
        logger.info(f"全局sigma计算: 标准差={global_sigma:.4f}, 稳健估计={robust_sigma:.4f}, 最终={final_sigma:.4f}")
        
        return float(final_sigma)
    
    def _fit_multidimensional_sigma(self, data: pd.DataFrame, target_col: str):
        """拟合多维度sigma模型"""
        
        if not self.local_config.get('enabled', True):
            return
        
        logger.info("拟合多维度sigma模型...")
        
        # 准备特征
        available_factors = [f for f in self.error_factors if f in data.columns]
        if len(available_factors) < 2:
            logger.warning("可用误差因子不足，使用简化模型")
            available_factors = ['week', 'BMI_used']
        
        X = data[available_factors].fillna(data[available_factors].median())
        
        # 计算局部残差作为目标
        threshold = 0.04
        y_residuals = np.abs(data[target_col] - threshold)
        
        # 尝试不同的sigma建模方法
        models = {}
        
        # 1. 线性回归模型
        try:
            linear_model = LinearRegression()
            linear_model.fit(X, y_residuals)
            models['linear'] = linear_model
        except Exception as e:
            logger.warning(f"线性sigma模型拟合失败: {e}")
        
        # 2. 随机森林模型
        try:
            rf_model = RandomForestRegressor(n_estimators=50, random_state=42)
            rf_model.fit(X, y_residuals)
            models['rf'] = rf_model
        except Exception as e:
            logger.warning(f"随机森林sigma模型拟合失败: {e}")
        
        # 3. Huber回归（稳健）
        try:
            huber_model = HuberRegressor()
            huber_model.fit(X, y_residuals)
            models['huber'] = huber_model
        except Exception as e:
            logger.warning(f"Huber sigma模型拟合失败: {e}")
        
        # 选择最佳模型
        best_model = self._select_best_sigma_model(models, X, y_residuals)
        self.sigma_model = best_model
        
        if best_model is not None:
            logger.info(f"最佳sigma模型选择完成: {type(best_model).__name__}")
        
        # 构建局部sigma网格
        self._build_local_sigma_grid(data, target_col, available_factors)
    
    def _select_best_sigma_model(self, models: dict, X: pd.DataFrame, y: pd.Series):
        """选择最佳sigma模型"""
        
        if not models:
            return None
        
        best_model = None
        best_score = -np.inf
        
        for name, model in models.items():
            try:
                # 使用交叉验证评估
                scores = cross_val_score(model, X, y, cv=3, scoring='r2')
                avg_score = scores.mean()
                
                logger.info(f"Sigma模型 {name}: CV R² = {avg_score:.4f}")
                
                if avg_score > best_score:
                    best_score = avg_score
                    best_model = model
                    
            except Exception as e:
                logger.warning(f"评估sigma模型 {name} 失败: {e}")
        
        return best_model
    
    def _build_local_sigma_grid(self, data: pd.DataFrame, target_col: str, factors: list):
        """构建局部sigma网格"""
        
        logger.info("构建局部sigma网格...")
        
        # 创建网格
        grid_points = {}
        for factor in factors:
            if factor in data.columns:
                factor_data = data[factor].dropna()
                if factor == 'week':
                    bins = self.local_config.get('week_bins', 6)
                elif factor == 'BMI_used':
                    bins = self.local_config.get('bmi_bins', 4)
                elif factor == 'age':
                    bins = self.local_config.get('age_bins', 3)
                else:
                    bins = 4
                
                grid_points[factor] = np.linspace(
                    factor_data.quantile(0.05),
                    factor_data.quantile(0.95),
                    bins
                )
        
        # 计算网格点的sigma值
        threshold = 0.04
        grid_data = []
        
        for factor1_val in grid_points.get(factors[0], [data[factors[0]].median()]):
            for factor2_val in grid_points.get(factors[1] if len(factors) > 1 else factors[0], 
                                             [data[factors[1] if len(factors) > 1 else factors[0]].median()]):
                
                # 找到邻近数据点
                tolerance1 = (grid_points[factors[0]][1] - grid_points[factors[0]][0]) / 2 if len(grid_points[factors[0]]) > 1 else 1.0
                if len(factors) > 1:
                    tolerance2 = (grid_points[factors[1]][1] - grid_points[factors[1]][0]) / 2 if len(grid_points[factors[1]]) > 1 else 1.0
                else:
                    tolerance2 = tolerance1
                
                mask = (np.abs(data[factors[0]] - factor1_val) <= tolerance1)
                if len(factors) > 1:
                    mask = mask & (np.abs(data[factors[1]] - factor2_val) <= tolerance2)
                
                local_data = data[mask]
                
                if len(local_data) >= self.local_config.get('min_samples_per_bin', 5):
                    # 计算局部sigma
                    local_residuals = local_data[target_col] - threshold
                    local_sigma = np.std(local_residuals)
                    
                    grid_data.append({
                        factors[0]: factor1_val,
                        factors[1] if len(factors) > 1 else factors[0]: factor2_val,
                        'local_sigma': local_sigma,
                        'n_samples': len(local_data)
                    })
        
        if grid_data:
            self.local_sigma_grid = pd.DataFrame(grid_data)
            logger.info(f"局部sigma网格构建完成: {len(grid_data)} 个网格点")
        else:
            logger.warning("无法构建局部sigma网格")
    
    def _build_local_sigma_function(self, data: pd.DataFrame):
        """构建局部sigma函数"""
        
        def local_sigma_func(week: float, bmi: float, **kwargs) -> float:
            """
            局部sigma查询函数
            
            Parameters:
            -----------
            week : float
                孕周
            bmi : float
                BMI值
            **kwargs : dict
                其他参数
                
            Returns:
            --------
            float
                局部sigma值
            """
            
            # 方法1：使用sigma模型预测
            if self.sigma_model is not None:
                try:
                    input_data = pd.DataFrame([{
                        'week': week,
                        'BMI_used': bmi,
                        'age': kwargs.get('age', 30),
                        'height': kwargs.get('height', 160)
                    }])
                    
                    # 只使用模型支持的特征
                    available_features = [f for f in self.error_factors if f in input_data.columns]
                    if available_features:
                        model_input = input_data[available_features]
                        predicted_sigma = self.sigma_model.predict(model_input)[0]
                        
                        # 确保sigma在合理范围内
                        predicted_sigma = np.clip(predicted_sigma, 0.001, 0.05)
                        return float(predicted_sigma)
                        
                except Exception as e:
                    logger.debug(f"Sigma模型预测失败: {e}")
            
            # 方法2：使用局部sigma网格插值
            if self.local_sigma_grid is not None and len(self.local_sigma_grid) > 0:
                try:
                    # 找到最近的网格点
                    grid_points = self.local_sigma_grid[['week', 'BMI_used']].values
                    query_point = np.array([[week, bmi]])
                    
                    distances = cdist(query_point, grid_points)[0]
                    nearest_idx = np.argmin(distances)
                    
                    # 使用距离加权插值
                    if distances[nearest_idx] < 2.0:  # 如果很近，直接使用
                        return float(self.local_sigma_grid.iloc[nearest_idx]['local_sigma'])
                    else:
                        # 使用多个邻近点插值
                        k = min(3, len(self.local_sigma_grid))
                        nearest_indices = np.argpartition(distances, k)[:k]
                        
                        weights = 1.0 / (distances[nearest_indices] + 1e-8)
                        weights = weights / weights.sum()
                        
                        weighted_sigma = np.sum(weights * self.local_sigma_grid.iloc[nearest_indices]['local_sigma'])
                        return float(weighted_sigma)
                        
                except Exception as e:
                    logger.debug(f"网格插值失败: {e}")
            
            # 方法3：回退到简单函数
            base_sigma = self.global_sigma or 0.01
            
            # BMI效应：BMI越高，sigma越大
            bmi_effect = max(0, (bmi - 25) * 0.0002)
            
            # 孕周效应：早期sigma较大
            week_effect = max(0, (15 - week) * 0.0001) if week < 15 else 0
            
            local_sigma = base_sigma + bmi_effect + week_effect
            return float(np.clip(local_sigma, 0.001, 0.05))
        
        self.local_sigma_func = local_sigma_func
        logger.info("局部sigma函数构建完成")
    
    def _build_shrinkage_function(self, data: pd.DataFrame):
        """构建收缩函数"""
        
        lambda_shrink = self.shrinkage_config.get('lambda', 0.2)
        adaptive = self.shrinkage_config.get('adaptive', True)
        
        def shrinkage_func(local_sigma: float, n_local_samples: int = 10) -> float:
            """
            收缩函数
            
            Parameters:
            -----------
            local_sigma : float
                局部sigma估计
            n_local_samples : int
                局部样本数量
                
            Returns:
            --------
            float
                收缩后的sigma
            """
            
            global_sigma = self.global_sigma or 0.01
            
            if adaptive:
                # 自适应收缩：样本数少时收缩更多
                effective_lambda = lambda_shrink * np.exp(-n_local_samples / 20.0)
            else:
                effective_lambda = lambda_shrink
            
            shrunk_sigma = (1 - effective_lambda) * local_sigma + effective_lambda * global_sigma
            return float(shrunk_sigma)
        
        self.shrinkage_func = shrinkage_func
        logger.info(f"收缩函数构建完成，lambda={lambda_shrink}, adaptive={adaptive}")
    
    def _build_threshold_adjustment_function(self):
        """构建动态阈值调整函数"""
        
        method = self.dynamic_threshold_config.get('method', 'quantile')
        confidence_level = self.dynamic_threshold_config.get('confidence_level', 0.95)
        
        def threshold_adjustment_func(week: float, bmi: float, **kwargs) -> float:
            """
            动态阈值调整函数
            
            Parameters:
            -----------
            week : float
                孕周
            bmi : float
                BMI值
            **kwargs : dict
                其他参数
                
            Returns:
            --------
            float
                调整后的阈值
            """
            
            base_threshold = 0.04
            
            # 获取局部sigma
            if self.local_sigma_func is not None:
                local_sigma = self.local_sigma_func(week, bmi, **kwargs)
                
                if self.shrinkage_func is not None:
                    n_local = kwargs.get('n_local_samples', 10)
                    local_sigma = self.shrinkage_func(local_sigma, n_local)
            else:
                local_sigma = self.global_sigma or 0.01
            
            # 计算调整量
            if method == 'normal':
                z_alpha = stats.norm.ppf(confidence_level)
                adjustment = z_alpha * local_sigma
            elif method == 'quantile':
                # 使用更保守的调整
                adjustment = stats.norm.ppf(0.75) * local_sigma  # 75%分位数
            else:
                adjustment = 2.0 * local_sigma  # 简单的2倍sigma规则
            
            adjusted_threshold = base_threshold + adjustment
            
            # 确保阈值在合理范围内
            adjusted_threshold = np.clip(adjusted_threshold, 0.035, 0.055)
            
            return float(adjusted_threshold)
        
        self.threshold_adjustment_func = threshold_adjustment_func
        logger.info(f"动态阈值调整函数构建完成，方法={method}, 置信水平={confidence_level}")
    
    def _evaluate_error_model(self, data: pd.DataFrame, target_col: str) -> Dict[str, Any]:
        """评估误差模型性能"""
        
        performance = {}
        
        if self.sigma_model is not None:
            try:
                # 准备数据
                available_factors = [f for f in self.error_factors if f in data.columns]
                X = data[available_factors].fillna(data[available_factors].median())
                
                threshold = 0.04
                y_true = np.abs(data[target_col] - threshold)
                
                # 预测
                y_pred = self.sigma_model.predict(X)
                
                # 计算指标
                mse = np.mean((y_true - y_pred) ** 2)
                mae = np.mean(np.abs(y_true - y_pred))
                r2 = 1 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - y_true.mean()) ** 2)
                
                performance = {
                    'mse': float(mse),
                    'mae': float(mae),
                    'r2': float(r2),
                    'rmse': float(np.sqrt(mse))
                }
                
            except Exception as e:
                logger.warning(f"误差模型评估失败: {e}")
                performance = {'error': str(e)}
        
        return performance
    
    def _create_fallback_model(self, df: pd.DataFrame, target_col: str) -> Dict[str, Any]:
        """创建回退误差模型"""
        
        logger.info("创建回退误差模型...")
        
        # 使用全数据估计全局sigma
        if self.global_sigma is None:
            threshold = 0.04
            residuals = df[target_col] - threshold
            self.global_sigma = np.std(residuals)
        
        # 创建简单的sigma函数
        def simple_sigma_func(week: float, bmi: float, **kwargs) -> float:
            return self.global_sigma
        
        self.local_sigma_func = simple_sigma_func
        
        # 创建简单的阈值调整函数
        def simple_threshold_func(week: float, bmi: float, **kwargs) -> float:
            return 0.04 + 2.0 * self.global_sigma
        
        self.threshold_adjustment_func = simple_threshold_func
        
        return {
            'global_sigma': self.global_sigma,
            'local_sigma_func': self.local_sigma_func,
            'threshold_adjustment_func': self.threshold_adjustment_func,
            'local_factors': ['week', 'BMI_used'],
            'model_performance': {'type': 'fallback'},
            'n_threshold_samples': len(df),
            'sigma_model': None
        }


class AdaptiveSigmaEstimator(EnhancedSigmaEstimator):
    """自适应sigma估计器，包含在线学习能力"""
    
    def __init__(self, config: dict, report_df: pd.DataFrame = None):
        super().__init__(config, report_df)
        self.online_learning = config.get('online_learning', False)
        self.update_history = []
    
    def update_with_new_data(self, new_data: pd.DataFrame, target_col: str = 'Y_frac'):
        """使用新数据更新模型"""
        
        if not self.online_learning:
            logger.info("在线学习未启用")
            return
        
        logger.info(f"使用新数据更新sigma模型: {len(new_data)} 样本")
        
        # 提取新的阈值附近数据
        new_threshold_data = self._extract_threshold_nearby_data(new_data, target_col)
        
        if len(new_threshold_data) < 5:
            logger.warning("新数据不足，跳过更新")
            return
        
        # 更新全局sigma
        threshold = 0.04
        new_residuals = new_threshold_data[target_col] - threshold
        new_sigma = np.std(new_residuals)
        
        # 使用指数移动平均更新
        alpha = 0.1
        self.global_sigma = (1 - alpha) * self.global_sigma + alpha * new_sigma
        
        # 记录更新历史
        self.update_history.append({
            'timestamp': pd.Timestamp.now(),
            'n_new_samples': len(new_threshold_data),
            'new_sigma': new_sigma,
            'updated_global_sigma': self.global_sigma
        })
        
        logger.info(f"Sigma模型已更新: 新全局sigma = {self.global_sigma:.4f}")


# 工具函数
def create_sigma_lookup_function(sigma_models: Dict[str, Any]) -> Callable:
    """
    创建sigma查询函数
    
    Parameters:
    -----------
    sigma_models : dict
        sigma模型结果
        
    Returns:
    --------
    callable
        sigma查询函数
    """
    
    local_func = sigma_models.get('local_sigma_func')
    shrinkage_func = sigma_models.get('shrinkage_func')
    global_sigma = sigma_models.get('global_sigma', 0.01)
    
    def sigma_lookup(week: float, bmi: float, **kwargs) -> float:
        """
        查询sigma值
        
        Parameters:
        -----------
        week : float
            孕周
        bmi : float
            BMI
        **kwargs : dict
            其他参数
            
        Returns:
        --------
        float
            sigma值
        """
        
        if local_func is not None:
            local_sigma = local_func(week, bmi, **kwargs)
            
            if shrinkage_func is not None:
                n_local = kwargs.get('n_local_samples', 10)
                return shrinkage_func(local_sigma, n_local)
            else:
                return local_sigma
        else:
            return global_sigma
    
    return sigma_lookup


def create_dynamic_threshold_function(sigma_models: Dict[str, Any]) -> Callable:
    """
    创建动态阈值函数
    
    Parameters:
    -----------
    sigma_models : dict
        sigma模型结果
        
    Returns:
    --------
    callable
        动态阈值函数
    """
    
    threshold_func = sigma_models.get('threshold_adjustment_func')
    
    if threshold_func is not None:
        return threshold_func
    else:
        # 创建简单的动态阈值函数
        sigma_func = create_sigma_lookup_function(sigma_models)
        
        def simple_threshold(week: float, bmi: float, **kwargs) -> float:
            base_threshold = 0.04
            sigma = sigma_func(week, bmi, **kwargs)
            z_alpha = 1.96  # 95%置信水平
            return base_threshold + z_alpha * sigma
        
        return simple_threshold


def analyze_error_sensitivity(sigma_models: Dict[str, Any], 
                            test_data: pd.DataFrame,
                            sigma_multipliers: List[float] = [0.5, 1.0, 1.5, 2.0]) -> Dict[str, Any]:
    """
    分析检测误差敏感性
    
    Parameters:
    -----------
    sigma_models : dict
        sigma模型
    test_data : pd.DataFrame
        测试数据
    sigma_multipliers : list
        sigma倍数
        
    Returns:
    --------
    dict
        敏感性分析结果
    """
    
    sigma_func = create_sigma_lookup_function(sigma_models)
    threshold_func = create_dynamic_threshold_function(sigma_models)
    
    results = {}
    
    for multiplier in sigma_multipliers:
        
        # 修改sigma函数
        def modified_sigma_func(week, bmi, **kwargs):
            return sigma_func(week, bmi, **kwargs) * multiplier
        
        # 修改阈值函数
        def modified_threshold_func(week, bmi, **kwargs):
            original_threshold = threshold_func(week, bmi, **kwargs)
            base_threshold = 0.04
            adjustment = (original_threshold - base_threshold) * multiplier
            return base_threshold + adjustment
        
        # 计算影响
        sample_results = []
        for _, row in test_data.iterrows():
            week = row.get('week', 12)
            bmi = row.get('BMI_used', 25)
            
            original_sigma = sigma_func(week, bmi)
            modified_sigma = modified_sigma_func(week, bmi)
            
            original_threshold = threshold_func(week, bmi)
            modified_threshold = modified_threshold_func(week, bmi)
            
            sample_results.append({
                'week': week,
                'bmi': bmi,
                'original_sigma': original_sigma,
                'modified_sigma': modified_sigma,
                'original_threshold': original_threshold,
                'modified_threshold': modified_threshold,
                'sigma_change': (modified_sigma - original_sigma) / original_sigma,
                'threshold_change': (modified_threshold - original_threshold) / original_threshold
            })
        
        results[f'multiplier_{multiplier}'] = sample_results
    
    # 汇总结果
    summary = {}
    for multiplier in sigma_multipliers:
        key = f'multiplier_{multiplier}'
        data = results[key]
        
        sigma_changes = [d['sigma_change'] for d in data]
        threshold_changes = [d['threshold_change'] for d in data]
        
        summary[key] = {
            'avg_sigma_change': np.mean(sigma_changes),
            'avg_threshold_change': np.mean(threshold_changes),
            'max_sigma_change': np.max(sigma_changes),
            'max_threshold_change': np.max(threshold_changes)
        }
    
    return {
        'detailed_results': results,
        'summary': summary,
        'sigma_multipliers': sigma_multipliers
    }
