"""
问题3分析程序
基于BMI的NIPT最佳检测时点优化
借鉴Q2的核心方法，包括σ建模、分位数回归、网格搜索优化等
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.metrics import roc_auc_score, r2_score, mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from scipy.stats import norm
import warnings
import logging
from datetime import datetime
import os
from typing import Callable, Dict, Any, Tuple, List, Optional

warnings.filterwarnings('ignore')


# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SigmaLookup:
    """σ查表器，借鉴Q2的测量误差建模"""
    
    def __init__(self, data: pd.DataFrame, threshold: float = 0.04, 
                 threshold_range: Tuple[float, float] = (0.03, 0.05)):
        """
        初始化σ查表器
        
        Parameters:
        -----------
        data : pd.DataFrame
            数据
        threshold : float
            浓度阈值
        threshold_range : tuple
            阈值邻域范围
        """
        self.threshold = threshold
        self.threshold_range = threshold_range
        self.global_sigma = None
        self.local_sigma_grid = None
        self._build_sigma_lookup(data)
    
    def _build_sigma_lookup(self, data: pd.DataFrame):
        """构建σ查表"""
        logger.info("构建σ查表...")
        
        # 筛选阈值邻域数据
        mask = (data['Y染色体浓度'] >= self.threshold_range[0]) & \
               (data['Y染色体浓度'] <= self.threshold_range[1])
        
        if mask.sum() < 10:
            logger.warning("阈值邻域样本不足，使用全局σ")
            self.global_sigma = data['Y染色体浓度'].std()
            return
        
        # 计算全局σ
        y_near_threshold = data.loc[mask, 'Y染色体浓度']
        self.global_sigma = y_near_threshold.std()
        
        logger.info(f"全局σ: {self.global_sigma:.4f}")
        
        # 计算局部σ（按孕周和BMI分箱）
        if len(data.loc[mask]) > 20:
            self._build_local_sigma_grid(data.loc[mask])
        else:
            logger.warning("样本数不足，跳过局部σ计算")
    
    def _build_local_sigma_grid(self, data: pd.DataFrame):
        """构建局部σ网格"""
        # 创建孕周和BMI分箱
        data['week_bin'] = pd.cut(data['孕周'], bins=5, labels=False)
        data['bmi_bin'] = pd.cut(data['BMI'], bins=3, labels=False)
        
        # 计算每个分箱的局部方差
        local_vars = data.groupby(['week_bin', 'bmi_bin'])['Y染色体浓度'].var().dropna()
        
        if len(local_vars) > 0:
            # 创建局部σ网格
            self.local_sigma_grid = {}
            for (w_bin, b_bin), var in local_vars.items():
                self.local_sigma_grid[(w_bin, b_bin)] = np.sqrt(var)
            
            logger.info(f"局部σ网格: {len(self.local_sigma_grid)} 个分箱")
        else:
            logger.warning("局部σ计算失败")
    
    def lookup(self, week: float, bmi: float) -> float:
        """
        查表获取σ值
        
        Parameters:
        -----------
        week : float
            孕周
        bmi : float
            BMI
            
        Returns:
        --------
        float
            σ值
        """
        if self.local_sigma_grid is None:
            return self.global_sigma
        
        # 找到对应的分箱
        week_bins = pd.cut([week], bins=5, labels=False)[0]
        bmi_bins = pd.cut([bmi], bins=3, labels=False)[0]
        
        if (week_bins, bmi_bins) in self.local_sigma_grid:
            return self.local_sigma_grid[(week_bins, bmi_bins)]
        else:
            return self.global_sigma


class QuantileRegressionModel:
    """分位数回归模型，借鉴Q2的GAM分位数回归"""
    
    def __init__(self, tau: float = 0.5):
        """
        初始化分位数回归模型
        
        Parameters:
        -----------
        tau : float
            分位数水平
        """
        self.tau = tau
        self.model = None
        self.feature_transformer = None
        self.is_fitted = False
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'QuantileRegressionModel':
        """
        拟合分位数回归模型
        
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
        # 特征工程：添加多项式特征和交互项
        self.feature_transformer = PolynomialFeatures(
            degree=2, include_bias=False, interaction_only=False
        )
        X_poly = self.feature_transformer.fit_transform(X)
        
        # 使用GradientBoostingRegressor进行分位数回归
        self.model = GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            random_state=42,
            loss='quantile',
            alpha=self.tau
        )
        
        self.model.fit(X_poly, y)
        self.is_fitted = True
        
        logger.info(f"分位数回归模型拟合完成 (tau={self.tau})")
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
            raise ValueError("模型尚未拟合")
        
        X_poly = self.feature_transformer.transform(X)
        return self.model.predict(X_poly)


class GridSearchOptimizer:
    """网格搜索优化器，借鉴Q2的优化算法"""
    
    def __init__(self, week_range: Tuple[float, float] = (10, 25),
                 week_step: float = 0.5, bmi_resolution: int = 40):
        """
        初始化网格搜索优化器
        
        Parameters:
        -----------
        week_range : tuple
            孕周搜索范围
        week_step : float
            孕周搜索步长
        bmi_resolution : int
            BMI网格分辨率
        """
        self.week_range = week_range
        self.week_step = week_step
        self.bmi_resolution = bmi_resolution
    
    def find_optimal_time_for_bmi(self, bmi: float, mu_model: QuantileRegressionModel,
                                 sigma_lookup: SigmaLookup, success_model: RandomForestClassifier,
                                 threshold: float = 0.04, alpha: float = 0.05) -> Tuple[float, float]:
        """
        为特定BMI寻找最优检测时点
        
        Parameters:
        -----------
        bmi : float
            BMI值
        mu_model : QuantileRegressionModel
            均值模型
        sigma_lookup : SigmaLookup
            σ查表器
        success_model : RandomForestClassifier
            成功概率模型
        threshold : float
            浓度阈值
        alpha : float
            置信水平
            
        Returns:
        --------
        tuple
            (最优时点, 最小风险)
        """
        weeks = np.arange(self.week_range[0], self.week_range[1] + self.week_step, self.week_step)
        
        min_risk = float('inf')
        optimal_week = weeks[0]
        
        for week in weeks:
            try:
                # 预测浓度
                mu = mu_model.predict(np.array([[week, bmi]]))[0]
                sigma = sigma_lookup.lookup(week, bmi)
                
                # 计算调整后的阈值
                z_alpha = norm.ppf(1 - alpha)
                thr_adj = threshold + z_alpha * sigma
                
                # 计算达标概率
                if sigma > 0:
                    z_score = (mu - thr_adj) / sigma
                    attain_prob = norm.cdf(z_score)
                else:
                    attain_prob = 1.0 if mu >= thr_adj else 0.0
                
                # 预测成功概率
                success_prob = success_model.predict_proba(np.array([[week, bmi, 0.5, 1000, 30]]))[0, 1]
                
                # 计算检测概率
                detect_prob = attain_prob * success_prob
                
                # 计算风险函数
                risk = self._calculate_risk(week, detect_prob)
                
                if risk < min_risk:
                    min_risk = risk
                    optimal_week = week
                    
            except Exception as e:
                logger.warning(f"BMI={bmi}, week={week} 计算失败: {e}")
                continue
        
        return optimal_week, min_risk
    
    def _calculate_risk(self, week: float, detect_prob: float) -> float:
        """
        计算风险函数
        
        Parameters:
        -----------
        week : float
            孕周
        detect_prob : float
            检测概率
            
        Returns:
        --------
        float
            风险值
        """
        # 时间段风险
        if week <= 12:
            time_risk = 0.1
        elif week <= 27:
            time_risk = 1.0
        else:
            time_risk = 2.0
        
        # 检测失败风险
        failure_risk = 1.0 * (1 - detect_prob)
        
        return time_risk + failure_risk


class BMIGroupingStrategy:
    """BMI分组策略，借鉴Q2的分组算法"""
    
    def __init__(self, min_group_n: int = 10, min_cut_distance: float = 1.0):
        """
        初始化BMI分组策略
        
        Parameters:
        -----------
        min_group_n : int
            最小组内样本量
        min_cut_distance : float
            最小切点间距
        """
        self.min_group_n = min_group_n
        self.min_cut_distance = min_cut_distance
    
    def find_bmi_cuts(self, wstar_curve: pd.DataFrame, 
                     who_cuts: List[float] = [18.5, 25.0, 30.0]) -> Tuple[List[float], pd.DataFrame]:
        """
        寻找BMI切点
        
        Parameters:
        -----------
        wstar_curve : pd.DataFrame
            w*(b)曲线数据
        who_cuts : list
            WHO标准切点
            
        Returns:
        --------
        tuple
            (切点列表, 分组结果表)
        """
        logger.info("寻找BMI切点...")
        
        # 使用WHO切点为起点
        initial_cuts = who_cuts.copy()
        
        # 过滤超出数据范围的切点
        bmi_min = wstar_curve['BMI'].min()
        bmi_max = wstar_curve['BMI'].max()
        valid_cuts = [cut for cut in initial_cuts if bmi_min <= cut <= bmi_max]
        
        if not valid_cuts:
            # 使用数据四分位数
            valid_cuts = wstar_curve['BMI'].quantile([0.25, 0.5, 0.75]).tolist()
        
        # 应用约束
        final_cuts = self._apply_constraints(valid_cuts, wstar_curve)
        
        # 创建分组结果表
        groups_df = self._create_groups_table(wstar_curve, final_cuts)
        
        logger.info(f"BMI切点确定: {final_cuts}")
        
        return final_cuts, groups_df
    
    def _apply_constraints(self, cuts: List[float], wstar_curve: pd.DataFrame) -> List[float]:
        """应用约束条件"""
        if not cuts:
            return cuts
        
        # 排序
        cuts = sorted(cuts)
        
        # 应用最小间距约束
        constrained_cuts = [cuts[0]]
        for cut in cuts[1:]:
            if cut - constrained_cuts[-1] >= self.min_cut_distance:
                constrained_cuts.append(cut)
        
        # 应用最小组内样本量约束
        final_cuts = []
        bmi_values = wstar_curve['BMI'].values
        
        for i, cut in enumerate(constrained_cuts):
            # 计算该切点两侧的样本量
            if i == 0:
                left_count = np.sum(bmi_values < cut)
            else:
                left_count = np.sum((bmi_values >= constrained_cuts[i-1]) & (bmi_values < cut))
            
            if i == len(constrained_cuts) - 1:
                right_count = np.sum(bmi_values >= cut)
            else:
                right_count = np.sum((bmi_values >= cut) & (bmi_values < constrained_cuts[i+1]))
            
            # 检查样本量约束
            if left_count >= self.min_group_n and right_count >= self.min_group_n:
                final_cuts.append(cut)
            else:
                logger.warning(f"切点 {cut:.2f} 不满足样本量约束: 左={left_count}, 右={right_count}")
        
        return final_cuts
    
    def _create_groups_table(self, wstar_curve: pd.DataFrame, cuts: List[float]) -> pd.DataFrame:
        """创建分组结果表"""
        if not cuts:
            # 无切点，所有数据为一组
            groups_df = pd.DataFrame({
                'group_id': [1],
                'bmi_min': [wstar_curve['BMI'].min()],
                'bmi_max': [wstar_curve['BMI'].max()],
                'bmi_mean': [wstar_curve['BMI'].mean()],
                'n_points': [len(wstar_curve)],
                'optimal_time': [wstar_curve['optimal_week'].mean()],
                'time_std': [wstar_curve['optimal_week'].std()],
                'mean_risk': [wstar_curve['min_risk'].mean()]
            })
            return groups_df
        
        # 创建分组
        groups = []
        bmi_values = wstar_curve['BMI'].values
        optimal_weeks = wstar_curve['optimal_week'].values
        risks = wstar_curve['min_risk'].values
        
        # 第一组
        mask = bmi_values < cuts[0]
        if mask.sum() > 0:
            groups.append({
                'group_id': 1,
                'bmi_min': bmi_values[mask].min(),
                'bmi_max': bmi_values[mask].max(),
                'bmi_mean': bmi_values[mask].mean(),
                'n_points': mask.sum(),
                'optimal_time': optimal_weeks[mask].mean(),
                'time_std': optimal_weeks[mask].std(),
                'mean_risk': risks[mask].mean()
            })
        
        # 中间组
        for i in range(len(cuts) - 1):
            mask = (bmi_values >= cuts[i]) & (bmi_values < cuts[i+1])
            if mask.sum() > 0:
                groups.append({
                    'group_id': i + 2,
                    'bmi_min': bmi_values[mask].min(),
                    'bmi_max': bmi_values[mask].max(),
                    'bmi_mean': bmi_values[mask].mean(),
                    'n_points': mask.sum(),
                    'optimal_time': optimal_weeks[mask].mean(),
                    'time_std': optimal_weeks[mask].std(),
                    'mean_risk': risks[mask].mean()
                })
        
        # 最后一组
        mask = bmi_values >= cuts[-1]
        if mask.sum() > 0:
            groups.append({
                'group_id': len(cuts) + 1,
                'bmi_min': bmi_values[mask].min(),
                'bmi_max': bmi_values[mask].max(),
                'bmi_mean': bmi_values[mask].mean(),
                'n_points': mask.sum(),
                'optimal_time': optimal_weeks[mask].mean(),
                'time_std': optimal_weeks[mask].std(),
                'mean_risk': risks[mask].mean()
            })
        
        return pd.DataFrame(groups)


class Problem3Analyzer:
    """问题3分析器"""
    
    def __init__(self, config: Optional[Dict] = None):
        """初始化分析器"""
        self.config = config or self._get_default_config()
        self.data = None
        self.models = {}
        self.results = {}
        
        # 创建输出目录
        os.makedirs('outputs', exist_ok=True)
        os.makedirs('outputs/logs', exist_ok=True)
    
    def _get_default_config(self) -> Dict:
        """获取默认配置"""
        return {
            'thresholds': {
                'y_percent_threshold': 0.04,
                'y_percent_thresholds_sensitivity': [0.035, 0.04, 0.045]
            },
            'sigma_modeling': {
                'threshold_range': (0.03, 0.05),
                'local_binning': {'week_bins': 5, 'bmi_bins': 3}
            },
            'quantile_regression': {
                'tau': 0.9,
                'n_estimators': 100,
                'learning_rate': 0.1,
                'max_depth': 6
            },
            'grid_search': {
                'week_range': (10, 25),
                'week_step': 0.5,
                'bmi_resolution': 40
            },
            'bmi_grouping': {
                'min_group_n': 10,
                'min_cut_distance': 1.0,
                'who_cuts': [18.5, 25.0, 30.0]
            },
            'risk_function': {
                'gamma': 1.0,
                'time_weights': {'early_weeks': 0.1, 'mid_weeks': 1.0, 'late_weeks': 2.0}
            }
        }
    
    def load_and_preprocess_data(self, data_path: str = 'data/q3_preprocessed.csv') -> pd.DataFrame:
        """加载和预处理数据"""
        logger.info("1. 加载和预处理数据...")
        
        # 加载数据
        if data_path.endswith('.xlsx'):
            data = pd.read_excel(data_path)
        else:
            data = pd.read_csv(data_path)
        
        logger.info(f"原始数据形状: {data.shape}")
        
        # 数据清洗
        data = data[data['胎儿性别'] == '男'].copy()
        data = data[(data['孕周'] >= 10) & (data['孕周'] <= 25)].copy()
        
        # 处理缺失值
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            if data[col].isnull().sum() > 0:
                data[col] = data[col].fillna(data[col].median())
        
        # 计算达标标志
        data['达标'] = (data['Y染色体浓度'] >= self.config['thresholds']['y_percent_threshold']).astype(int)
        
        logger.info(f"清洗后数据形状: {data.shape}")
        logger.info(f"达标率: {data['达标'].mean():.3f}")
        logger.info(f"成功率: {data['是否成功'].mean():.3f}")
        
        self.data = data
        return data
    
    def train_models(self) -> Dict[str, Any]:
        """训练模型"""
        logger.info("\n2. 训练模型...")
        
        # 准备特征
        feature_columns = ['孕周', 'BMI', 'GC含量', 'read_count', '年龄']
        X = self.data[feature_columns].copy()
        y_attain = self.data['达标'].values
        y_success = self.data['是否成功'].values
        y_percent = self.data['Y染色体浓度'].values
        
        # 训练分位数回归模型
        logger.info("训练分位数回归模型...")
        self.models['quantile'] = QuantileRegressionModel(tau=self.config['quantile_regression']['tau'])
        X_quantile = X[['孕周', 'BMI']].values
        self.models['quantile'].fit(X_quantile, y_percent)
        
        # 训练成功概率模型
        logger.info("训练成功概率模型...")
        self.models['success'] = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=20,
            min_samples_leaf=10,
            random_state=42
        )
        self.models['success'].fit(X, y_success)
        
        # 构建σ查表器
        logger.info("构建σ查表器...")
        self.models['sigma_lookup'] = SigmaLookup(
            self.data,
            threshold=self.config['thresholds']['y_percent_threshold'],
            threshold_range=self.config['sigma_modeling']['threshold_range']
        )
        
        # 评估模型性能
        y_pred = self.models['quantile'].predict(X_quantile)
        quantile_score = r2_score(y_percent, y_pred)
        success_score = roc_auc_score(y_success, self.models['success'].predict_proba(X)[:, 1])
        
        logger.info(f"模型性能评估:")
        logger.info(f"  分位数回归模型 R²: {quantile_score:.4f}")
        logger.info(f"  成功概率模型 AUC: {success_score:.4f}")
        
        return {
            'quantile_score': quantile_score,
            'success_score': success_score
        }
    
    def optimize_individual_times(self) -> pd.DataFrame:
        """优化个体检测时点"""
        logger.info("\n3. 优化个体检测时点...")
        
        # 创建网格搜索优化器
        optimizer = GridSearchOptimizer(
            week_range=self.config['grid_search']['week_range'],
            week_step=self.config['grid_search']['week_step'],
            bmi_resolution=self.config['grid_search']['bmi_resolution']
        )
        
        # 创建BMI网格
        bmi_min = self.data['BMI'].min()
        bmi_max = self.data['BMI'].max()
        bmi_grid = np.linspace(bmi_min, bmi_max, self.config['grid_search']['bmi_resolution'])
        
        logger.info(f"BMI网格范围: {bmi_min:.2f} - {bmi_max:.2f}")
        
        # 在BMI网格上求最优时点
        records = []
        for i, bmi in enumerate(bmi_grid):
            if i % 10 == 0:
                logger.info(f"进度: {i+1}/{len(bmi_grid)}")
            
            optimal_week, min_risk = optimizer.find_optimal_time_for_bmi(
                bmi, self.models['quantile'], self.models['sigma_lookup'], 
                self.models['success']
            )
            
            records.append({
                'BMI': bmi,
                'optimal_week': optimal_week,
                'min_risk': min_risk
            })
        
        wstar_curve = pd.DataFrame(records)
        
        logger.info(f"个体优化完成，平均最优时点: {wstar_curve['optimal_week'].mean():.2f}周")
        
        self.results['wstar_curve'] = wstar_curve
        return wstar_curve
    
    def perform_bmi_grouping(self) -> Tuple[List[float], pd.DataFrame]:
        """执行BMI分组"""
        logger.info("\n4. 执行BMI分组...")
        
        wstar_curve = self.results['wstar_curve']
        
        # 创建BMI分组策略
        grouping_strategy = BMIGroupingStrategy(
            min_group_n=self.config['bmi_grouping']['min_group_n'],
            min_cut_distance=self.config['bmi_grouping']['min_cut_distance']
        )
        
        # 寻找BMI切点
        cuts, groups_df = grouping_strategy.find_bmi_cuts(
            wstar_curve, 
            who_cuts=self.config['bmi_grouping']['who_cuts']
        )
        
        logger.info("BMI分组结果:")
        for _, group in groups_df.iterrows():
            logger.info(f"  组 {group['group_id']}: BMI [{group['bmi_min']:.1f}, {group['bmi_max']:.1f}] -> {group['optimal_time']:.1f}周")
        
        self.results['bmi_groups'] = {
            'cuts': cuts,
            'groups_df': groups_df
        }
        
        return cuts, groups_df
    
    def perform_sensitivity_analysis(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """执行敏感性分析"""
        logger.info("\n5. 执行敏感性分析...")
        
        # 阈值敏感性分析
        thresholds = self.config['thresholds']['y_percent_thresholds_sensitivity']
        threshold_results = []
        
        for threshold in thresholds:
            # 重新计算达标概率
            self.data['达标_temp'] = (self.data['Y染色体浓度'] >= threshold).astype(int)
            
            # 重新训练分位数回归模型
            feature_columns = ['孕周', 'BMI', 'GC含量', 'read_count', '年龄']
            X = self.data[feature_columns].copy()
            y_attain = self.data['达标_temp'].values
            
            temp_model = QuantileRegressionModel(tau=0.9)
            X_quantile = X[['孕周', 'BMI']].values
            temp_model.fit(X_quantile, y_attain)
            
            # 计算平均检测概率
            y_pred = temp_model.predict(X_quantile)
            success_probs = self.models['success'].predict_proba(X)[:, 1]
            detect_probs = y_pred * success_probs
            
            threshold_results.append({
                'threshold': threshold,
                'avg_detect_prob': detect_probs.mean(),
                'std_detect_prob': detect_probs.std()
            })
        
        threshold_df = pd.DataFrame(threshold_results)
        
        # 误差敏感性分析
        error_factors = [0.8, 1.0, 1.2]
        error_results = []
        
        for factor in error_factors:
            # 模拟测量误差
            y_percent_noisy = self.data['Y染色体浓度'] + np.random.normal(0, 0.01 * factor, len(self.data))
            y_percent_noisy = np.clip(y_percent_noisy, 0, 1)
            
            # 重新计算达标概率
            attain_probs = (y_percent_noisy >= self.config['thresholds']['y_percent_threshold']).astype(float)
            success_probs = self.models['success'].predict_proba(X)[:, 1]
            detect_probs = attain_probs * success_probs
            
            error_results.append({
                'error_factor': factor,
                'avg_detect_prob': detect_probs.mean(),
                'std_detect_prob': detect_probs.std()
            })
        
        error_df = pd.DataFrame(error_results)
        
        logger.info("敏感性分析完成")
        logger.info(f"阈值敏感性: {threshold_df['avg_detect_prob'].std():.4f}")
        logger.info(f"误差敏感性: {error_df['avg_detect_prob'].std():.4f}")
        
        self.results['sensitivity'] = {
            'threshold': threshold_df,
            'error': error_df
        }
        
        return threshold_df, error_df
    
    
    
    
    
    def save_results(self):
        """保存结果"""
        logger.info("\n7. 保存结果...")
        
        # 保存主要结果表
        if 'bmi_groups' in self.results:
            groups_df = self.results['bmi_groups']['groups_df']
            groups_df.to_csv('outputs/bmi_groups_optimal.csv', index=False, encoding='utf-8')
            logger.info("主要结果表已保存: outputs/bmi_groups_optimal.csv")
        
        # 保存w*(b)曲线
        if 'wstar_curve' in self.results:
            wstar_curve = self.results['wstar_curve']
            wstar_curve.to_csv('outputs/wstar_curve.csv', index=False, encoding='utf-8')
            logger.info("w*(b)曲线已保存: outputs/wstar_curve.csv")
        
        # 保存敏感性分析结果
        if 'sensitivity' in self.results:
            threshold_df = self.results['sensitivity']['threshold']
            error_df = self.results['sensitivity']['error']
            threshold_df.to_csv('outputs/threshold_sensitivity.csv', index=False, encoding='utf-8')
            error_df.to_csv('outputs/error_sensitivity.csv', index=False, encoding='utf-8')
            logger.info("敏感性分析结果已保存")
    
    def print_final_results(self):
        """打印最终结果"""
        if 'bmi_groups' not in self.results:
            logger.warning("尚未完成分析，无法打印结果")
            return
        
        groups_df = self.results['bmi_groups']['groups_df']
        
        print("\n" + "=" * 80)
        print("问题3最终结果：基于BMI的NIPT最佳检测时点优化")
        print("=" * 80)
        
        print("\n【BMI分组与推荐检测时点】")
        print("-" * 50)
        for _, row in groups_df.iterrows():
            print(f"组 {row['group_id']}: BMI [{row['bmi_min']:.1f}, {row['bmi_max']:.1f}]")
            print(f"  推荐检测时点: {row['optimal_time']:.1f} ± {row['time_std']:.1f} 周")
            print(f"  样本数: {row['n_points']}")
            print(f"  平均风险: {row['mean_risk']:.3f}")
            print()
        
        if 'sensitivity' in self.results:
            threshold_df = self.results['sensitivity']['threshold']
            print("【敏感性分析结果】")
            print("-" * 50)
            print("阈值敏感性:")
            for _, row in threshold_df.iterrows():
                print(f"  阈值 {row['threshold']:.3f}: 检测概率 {row['avg_detect_prob']:.3f} ± {row['std_detect_prob']:.3f}")
            print()
        
        print("【主要特点】")
        print("-" * 50)
        print("1. 精确的σ建模技术，提高测量误差估计精度")
        print("2. 分位数回归模型，更适合浓度预测")
        print("3. 网格搜索优化，确保全局最优解")
        print("4. 约束条件保证分组质量")
        print("5. 敏感性分析评估结果稳健性")
        print("\n详细结果请查看outputs目录下的文件")
    
    def run_complete_analysis(self, data_path: str = 'data/q3_preprocessed.csv') -> Dict[str, Any]:
        """运行完整分析流程"""
        try:
            logger.info("=" * 60)
            logger.info("开始问题3分析流程")
            logger.info("=" * 60)
            
            # 1. 数据预处理
            self.load_and_preprocess_data(data_path)
            
            # 2. 模型训练
            self.train_models()
            
            # 3. 个体优化
            self.optimize_individual_times()
            
            # 4. BMI分组
            self.perform_bmi_grouping()
            
            # 5. 敏感性分析
            self.perform_sensitivity_analysis()
            
            
            # 7. 保存结果
            self.save_results()
            
            # 8. 打印结果
            self.print_final_results()
            
            logger.info("\n" + "=" * 60)
            logger.info("问题3分析流程完成")
            logger.info("=" * 60)
            
            return self.results
            
        except Exception as e:
            logger.error(f"分析过程中出现错误: {e}")
            raise


def main():
    """主函数"""
    try:
        # 创建分析器
        analyzer = Problem3Analyzer()
        
        # 运行完整分析
        results = analyzer.run_complete_analysis()
        
        return results
        
    except Exception as e:
        logger.error(f"程序运行失败: {e}")
        raise


if __name__ == "__main__":
    main()
