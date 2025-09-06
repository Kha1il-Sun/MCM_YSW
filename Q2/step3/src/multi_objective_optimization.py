"""
多目标优化模块
实现综合考虑多种因素的风险函数优化
"""

import pandas as pd
import numpy as np
from scipy.optimize import minimize_scalar, minimize, differential_evolution
from scipy.stats import norm
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import cross_val_score
from typing import Dict, List, Tuple, Optional, Any, Callable
import logging
import warnings
from joblib import Parallel, delayed

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


class MultiObjectiveOptimizer:
    """多目标优化器"""
    
    def __init__(self, config: dict, concentration_model, success_model, sigma_models: dict):
        """
        初始化优化器
        
        Parameters:
        -----------
        config : dict
            优化配置
        concentration_model : MultiFactorYConcentrationModel
            浓度预测模型
        success_model : MultiFactorSuccessModel
            成功率模型
        sigma_models : dict
            误差模型
        """
        self.config = config
        self.concentration_model = concentration_model
        self.success_model = success_model
        self.sigma_models = sigma_models
        
        # 优化参数
        self.w_min = config.get('w_min', 8.0)
        self.w_max = config.get('w_max', 22.0)
        self.w_step = config.get('w_step', 0.1)
        self.b_resolution = config.get('b_resolution', 50)
        
        # 风险权重
        self.risk_weights = config.get('risk_weights', {})
        self.time_weights = self.risk_weights.get('time_weights', [1.0, 3.0, 8.0])
        self.failure_cost = self.risk_weights.get('failure_cost', 2.0)
        self.detection_accuracy_weight = self.risk_weights.get('detection_accuracy_weight', 1.5)
        self.multi_factor_weight = self.risk_weights.get('multi_factor_weight', 1.0)
        
        # 约束条件
        self.constraints = config.get('constraints', {})
        self.min_success_prob = self.constraints.get('min_success_prob', 0.85)
        self.min_attain_prob = self.constraints.get('min_attain_prob', 0.80)
        self.safety_margin = self.constraints.get('safety_margin', 0.5)
        
        # 动态阈值
        self.threshold_base = config.get('thr_adj', 0.04)
        
        logger.info(f"多目标优化器初始化完成")
        logger.info(f"  - 优化范围: {self.w_min}-{self.w_max}周，步长{self.w_step}")
        logger.info(f"  - BMI分辨率: {self.b_resolution}")
        logger.info(f"  - 最小成功概率: {self.min_success_prob}")
        logger.info(f"  - 最小达标概率: {self.min_attain_prob}")
    
    def solve_optimal_timing(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        求解最优时点
        
        Parameters:
        -----------
        df : pd.DataFrame
            数据
            
        Returns:
        --------
        dict
            优化结果
        """
        logger.info("开始多目标优化求解...")
        
        # 1. 构建BMI网格
        bmi_range = [df['BMI_used'].min(), df['BMI_used'].max()]
        bmi_grid = np.linspace(bmi_range[0], bmi_range[1], self.b_resolution)
        
        # 2. 为每个BMI值求解最优时点
        logger.info(f"为{len(bmi_grid)}个BMI值求解最优时点...")
        
        # 并行优化
        optimal_results = Parallel(n_jobs=-1, verbose=0)(
            delayed(self._optimize_single_bmi)(bmi, df) 
            for bmi in bmi_grid
        )
        
        # 3. 构建结果数据结构
        wstar_curve_data = []
        for i, bmi in enumerate(bmi_grid):
            result = optimal_results[i]
            wstar_curve_data.append({
                'BMI': bmi,
                'optimal_week': result['optimal_week'],
                'risk_value': result['risk_value'],
                'success_prob': result['success_prob'],
                'attain_prob': result['attain_prob'],
                'technical_success_prob': result['technical_success_prob'],
                'multi_factor_risk': result['multi_factor_risk'],
                'constraint_satisfied': result['constraint_satisfied']
            })
        
        wstar_curve_df = pd.DataFrame(wstar_curve_data)
        
        # 4. 应用保序约束
        wstar_curve_df = self._apply_isotonic_constraint(wstar_curve_df)
        
        # 5. 处理约束不满足的情况
        wstar_curve_df = self._handle_constraint_violations(wstar_curve_df)
        
        # 6. 计算优化质量指标
        optimization_quality = self._evaluate_optimization_quality(wstar_curve_df, df)
        
        results = {
            'curve_data': wstar_curve_df,
            'bmi_range': bmi_range,
            'timing_range': [wstar_curve_df['optimal_week'].min(), wstar_curve_df['optimal_week'].max()],
            'optimization_quality': optimization_quality,
            'config_used': self.config,
            'n_bmi_points': len(bmi_grid)
        }
        
        logger.info("多目标优化完成")
        logger.info(f"  - BMI范围: [{bmi_range[0]:.1f}, {bmi_range[1]:.1f}]")
        logger.info(f"  - 最优时点范围: [{results['timing_range'][0]:.1f}, {results['timing_range'][1]:.1f}]周")
        logger.info(f"  - 优化质量: {optimization_quality.get('avg_risk_reduction', 'N/A')}")
        
        return results
    
    def _optimize_single_bmi(self, bmi: float, df: pd.DataFrame) -> Dict[str, Any]:
        """
        为单个BMI值优化时点
        
        Parameters:
        -----------
        bmi : float
            BMI值
        df : pd.DataFrame
            数据
            
        Returns:
        --------
        dict
            优化结果
        """
        
        # 构建该BMI的代表性特征
        representative_features = self._get_representative_features(bmi, df)
        
        # 定义目标函数
        def objective(week):
            return self._multi_objective_risk_function(week, bmi, representative_features)
        
        # 定义约束函数
        def constraint_func(week):
            return self._evaluate_constraints(week, bmi, representative_features)
        
        # 粗网格搜索
        week_grid = np.arange(self.w_min, self.w_max + self.w_step, self.w_step)
        
        best_week = None
        best_risk = np.inf
        best_result = None
        
        for week in week_grid:
            try:
                risk = objective(week)
                constraints_satisfied = constraint_func(week)
                
                if constraints_satisfied and risk < best_risk:
                    best_risk = risk
                    best_week = week
                    
                    # 计算详细结果
                    detailed_result = self._compute_detailed_metrics(week, bmi, representative_features)
                    best_result = detailed_result
                    
            except Exception as e:
                logger.debug(f"计算失败 BMI={bmi:.1f}, week={week:.1f}: {e}")
                continue
        
        # 如果没有找到满足约束的解，选择风险最小的
        if best_week is None:
            logger.warning(f"BMI {bmi:.1f} 无满足约束的解，选择最佳可行解")
            
            risks = []
            for week in week_grid:
                try:
                    risk = objective(week)
                    risks.append((week, risk))
                except:
                    continue
            
            if risks:
                best_week, best_risk = min(risks, key=lambda x: x[1])
                best_result = self._compute_detailed_metrics(best_week, bmi, representative_features)
                best_result['constraint_satisfied'] = False
            else:
                # 最后的回退策略
                best_week = (self.w_min + self.w_max) / 2
                best_result = {
                    'optimal_week': best_week,
                    'risk_value': 999.0,
                    'success_prob': 0.5,
                    'attain_prob': 0.5,
                    'technical_success_prob': 0.5,
                    'multi_factor_risk': 999.0,
                    'constraint_satisfied': False
                }
        
        return best_result
    
    def _get_representative_features(self, bmi: float, df: pd.DataFrame) -> Dict[str, float]:
        """获取代表性特征"""
        
        # 找到相似BMI的数据
        bmi_tolerance = 2.0
        similar_bmi_data = df[np.abs(df['BMI_used'] - bmi) <= bmi_tolerance]
        
        if len(similar_bmi_data) == 0:
            # 使用全体数据
            similar_bmi_data = df
        
        # 计算代表性特征值
        features = {
            'BMI_used': bmi,
            'age': similar_bmi_data.get('age', pd.Series([30])).median(),
            'height': similar_bmi_data.get('height', pd.Series([160])).median(),
            'weight': similar_bmi_data.get('weight', pd.Series([bmi * (160/100)**2])).median(),
            'gc_percent': similar_bmi_data.get('gc_percent', pd.Series([42])).median(),
            'readcount': similar_bmi_data.get('readcount', pd.Series([1e7])).median()
        }
        
        return features
    
    def _multi_objective_risk_function(self, week: float, bmi: float, features: Dict[str, float]) -> float:
        """
        多目标风险函数
        
        Parameters:
        -----------
        week : float
            孕周
        bmi : float
            BMI
        features : dict
            特征
            
        Returns:
        --------
        float
            总风险值
        """
        
        # 1. 时间风险 - 基于妊娠期风险分段
        time_risk = self._calculate_time_risk(week)
        
        # 2. 检测失败风险
        detection_risk = self._calculate_detection_risk(week, bmi, features)
        
        # 3. 达标失败风险
        attainment_risk = self._calculate_attainment_risk(week, bmi, features)
        
        # 4. 多因素相关风险
        multi_factor_risk = self._calculate_multi_factor_risk(week, bmi, features)
        
        # 5. 不确定性风险
        uncertainty_risk = self._calculate_uncertainty_risk(week, bmi, features)
        
        # 加权组合
        total_risk = (
            self.time_weights[0] * time_risk +
            self.failure_cost * detection_risk +
            self.detection_accuracy_weight * attainment_risk +
            self.multi_factor_weight * multi_factor_risk +
            0.5 * uncertainty_risk
        )
        
        return float(total_risk)
    
    def _calculate_time_risk(self, week: float) -> float:
        """计算时间相关风险"""
        
        if week <= 12:
            # 早期妊娠：低风险
            time_risk = self.time_weights[0] * (12 - week) / 12
        elif week <= 27:
            # 中期妊娠：中等风险，随时间增加
            time_risk = self.time_weights[1] * (week - 12) / 15
        else:
            # 晚期妊娠：高风险
            time_risk = self.time_weights[2] * (1 + (week - 27) / 10)
        
        return float(time_risk)
    
    def _calculate_detection_risk(self, week: float, bmi: float, features: Dict[str, float]) -> float:
        """计算检测失败风险"""
        
        # 构建输入数据
        input_data = pd.DataFrame([features])
        input_data['week'] = week
        
        # 预测技术成功概率
        try:
            tech_success_prob = self.success_model.predict_success_probability(input_data)[0]
            detection_risk = 1.0 - tech_success_prob
        except Exception as e:
            logger.debug(f"技术成功率预测失败: {e}")
            # 使用简化计算
            detection_risk = self._simple_detection_risk(week, bmi)
        
        return float(detection_risk)
    
    def _calculate_attainment_risk(self, week: float, bmi: float, features: Dict[str, float]) -> float:
        """计算达标失败风险"""
        
        # 构建输入数据
        input_data = pd.DataFrame([features])
        input_data['week'] = week
        
        try:
            # 预测Y染色体浓度
            y_pred = self.concentration_model.predict(input_data)[0]
            
            # 获取动态阈值
            threshold = self._get_dynamic_threshold(week, bmi, features)
            
            # 获取预测不确定性
            sigma = self._get_prediction_sigma(week, bmi, features)
            
            # 计算达标概率
            attain_prob = norm.cdf((y_pred - threshold) / sigma)
            attainment_risk = 1.0 - attain_prob
            
        except Exception as e:
            logger.debug(f"达标风险计算失败: {e}")
            # 使用简化计算
            attainment_risk = self._simple_attainment_risk(week, bmi)
        
        return float(attainment_risk)
    
    def _calculate_multi_factor_risk(self, week: float, bmi: float, features: Dict[str, float]) -> float:
        """计算多因素相关风险"""
        
        risk_components = []
        
        # BMI风险
        if bmi < 18.5:
            bmi_risk = (18.5 - bmi) * 0.05  # 低体重风险
        elif bmi > 30:
            bmi_risk = (bmi - 30) * 0.02   # 肥胖风险
        else:
            bmi_risk = 0
        risk_components.append(bmi_risk)
        
        # 年龄风险
        age = features.get('age', 30)
        if age < 20:
            age_risk = (20 - age) * 0.02
        elif age > 35:
            age_risk = (age - 35) * 0.01
        else:
            age_risk = 0
        risk_components.append(age_risk)
        
        # 身高风险（极端身高可能影响检测）
        height = features.get('height', 160)
        if height < 150 or height > 180:
            height_risk = min(abs(height - 165) / 50, 0.1)
        else:
            height_risk = 0
        risk_components.append(height_risk)
        
        # 技术因子风险
        gc_percent = features.get('gc_percent', 42)
        if gc_percent < 40 or gc_percent > 45:
            gc_risk = abs(gc_percent - 42.5) * 0.01
        else:
            gc_risk = 0
        risk_components.append(gc_risk)
        
        readcount = features.get('readcount', 1e7)
        if readcount < 5e6:
            readcount_risk = (5e6 - readcount) / 1e6 * 0.02
        else:
            readcount_risk = 0
        risk_components.append(readcount_risk)
        
        # 交互风险：高BMI + 高年龄
        if bmi > 30 and age > 35:
            interaction_risk = 0.05
        else:
            interaction_risk = 0
        risk_components.append(interaction_risk)
        
        total_multi_factor_risk = sum(risk_components)
        
        return float(total_multi_factor_risk)
    
    def _calculate_uncertainty_risk(self, week: float, bmi: float, features: Dict[str, float]) -> float:
        """计算不确定性风险"""
        
        try:
            # 获取预测不确定性
            input_data = pd.DataFrame([features])
            input_data['week'] = week
            
            if hasattr(self.concentration_model, 'predict_with_uncertainty'):
                _, uncertainty = self.concentration_model.predict_with_uncertainty(input_data)
                uncertainty_risk = uncertainty[0] / self.threshold_base  # 相对不确定性
            else:
                sigma = self._get_prediction_sigma(week, bmi, features)
                uncertainty_risk = sigma / self.threshold_base
            
        except Exception as e:
            logger.debug(f"不确定性风险计算失败: {e}")
            # 使用简化估计
            uncertainty_risk = 0.1
        
        return float(uncertainty_risk)
    
    def _simple_detection_risk(self, week: float, bmi: float) -> float:
        """简化的检测风险计算"""
        
        base_risk = 0.1
        
        # 早期妊娠风险更高
        if week < 12:
            early_penalty = (12 - week) * 0.02
        else:
            early_penalty = 0
        
        # 高BMI风险更高
        if bmi > 30:
            bmi_penalty = (bmi - 30) * 0.01
        else:
            bmi_penalty = 0
        
        return base_risk + early_penalty + bmi_penalty
    
    def _simple_attainment_risk(self, week: float, bmi: float) -> float:
        """简化的达标风险计算"""
        
        # 基于经验的简化公式
        base_prob = 0.9 - (bmi - 22) * 0.01 + (week - 10) * 0.02
        base_prob = np.clip(base_prob, 0.1, 0.95)
        
        return 1.0 - base_prob
    
    def _get_dynamic_threshold(self, week: float, bmi: float, features: Dict[str, float]) -> float:
        """获取动态阈值"""
        
        if 'threshold_adjustment_func' in self.sigma_models:
            try:
                return self.sigma_models['threshold_adjustment_func'](week, bmi, **features)
            except:
                pass
        
        # 回退到简单动态阈值
        sigma = self._get_prediction_sigma(week, bmi, features)
        z_alpha = 1.96  # 95%置信水平
        return self.threshold_base + z_alpha * sigma
    
    def _get_prediction_sigma(self, week: float, bmi: float, features: Dict[str, float]) -> float:
        """获取预测sigma"""
        
        if 'local_sigma_func' in self.sigma_models:
            try:
                return self.sigma_models['local_sigma_func'](week, bmi, **features)
            except:
                pass
        
        return self.sigma_models.get('global_sigma', 0.01)
    
    def _evaluate_constraints(self, week: float, bmi: float, features: Dict[str, float]) -> bool:
        """评估约束条件"""
        
        # 构建输入数据
        input_data = pd.DataFrame([features])
        input_data['week'] = week
        
        try:
            # 计算成功概率
            tech_success_prob = self.success_model.predict_success_probability(input_data)[0]
            
            # 计算达标概率
            y_pred = self.concentration_model.predict(input_data)[0]
            threshold = self._get_dynamic_threshold(week, bmi, features)
            sigma = self._get_prediction_sigma(week, bmi, features)
            attain_prob = norm.cdf((y_pred - threshold) / sigma)
            
            # 综合成功概率
            total_success_prob = tech_success_prob * attain_prob
            
            # 检查约束
            success_constraint = tech_success_prob >= self.min_success_prob
            attain_constraint = attain_prob >= self.min_attain_prob
            
            return success_constraint and attain_constraint
            
        except Exception as e:
            logger.debug(f"约束评估失败: {e}")
            return False
    
    def _compute_detailed_metrics(self, week: float, bmi: float, features: Dict[str, float]) -> Dict[str, Any]:
        """计算详细指标"""
        
        # 构建输入数据
        input_data = pd.DataFrame([features])
        input_data['week'] = week
        
        try:
            # 预测指标
            y_pred = self.concentration_model.predict(input_data)[0]
            tech_success_prob = self.success_model.predict_success_probability(input_data)[0]
            threshold = self._get_dynamic_threshold(week, bmi, features)
            sigma = self._get_prediction_sigma(week, bmi, features)
            attain_prob = norm.cdf((y_pred - threshold) / sigma)
            
            # 风险计算
            risk_value = self._multi_objective_risk_function(week, bmi, features)
            multi_factor_risk = self._calculate_multi_factor_risk(week, bmi, features)
            
            # 约束检查
            constraint_satisfied = self._evaluate_constraints(week, bmi, features)
            
            return {
                'optimal_week': float(week),
                'risk_value': float(risk_value),
                'success_prob': float(tech_success_prob * attain_prob),
                'attain_prob': float(attain_prob),
                'technical_success_prob': float(tech_success_prob),
                'multi_factor_risk': float(multi_factor_risk),
                'constraint_satisfied': bool(constraint_satisfied),
                'y_prediction': float(y_pred),
                'dynamic_threshold': float(threshold),
                'prediction_sigma': float(sigma)
            }
            
        except Exception as e:
            logger.warning(f"详细指标计算失败: {e}")
            return {
                'optimal_week': float(week),
                'risk_value': 999.0,
                'success_prob': 0.5,
                'attain_prob': 0.5,
                'technical_success_prob': 0.5,
                'multi_factor_risk': 999.0,
                'constraint_satisfied': False,
                'y_prediction': 0.04,
                'dynamic_threshold': 0.04,
                'prediction_sigma': 0.01
            }
    
    def _apply_isotonic_constraint(self, wstar_df: pd.DataFrame) -> pd.DataFrame:
        """应用保序约束（BMI增加→时点增加）"""
        
        logger.info("应用保序约束...")
        
        result_df = wstar_df.copy()
        
        # 使用保序回归平滑
        isotonic_reg = IsotonicRegression(increasing=True)
        
        try:
            smoothed_weeks = isotonic_reg.fit_transform(
                result_df['BMI'], 
                result_df['optimal_week']
            )
            
            result_df['optimal_week'] = smoothed_weeks
            
            # 重新计算风险值（保序后）
            updated_risks = []
            for _, row in result_df.iterrows():
                features = self._get_representative_features(row['BMI'], pd.DataFrame())
                risk = self._multi_objective_risk_function(row['optimal_week'], row['BMI'], features)
                updated_risks.append(risk)
            
            result_df['risk_value'] = updated_risks
            
            logger.info("保序约束应用完成")
            
        except Exception as e:
            logger.warning(f"保序约束应用失败: {e}")
        
        return result_df
    
    def _handle_constraint_violations(self, wstar_df: pd.DataFrame) -> pd.DataFrame:
        """处理约束违反情况"""
        
        n_violations = (~wstar_df['constraint_satisfied']).sum()
        
        if n_violations > 0:
            logger.warning(f"发现{n_violations}个BMI值的约束违反")
            
            # 对违反约束的点，使用邻近满足约束的点进行插值
            satisfied_mask = wstar_df['constraint_satisfied']
            
            if satisfied_mask.any():
                # 使用线性插值
                from scipy.interpolate import interp1d
                
                satisfied_data = wstar_df[satisfied_mask]
                
                if len(satisfied_data) >= 2:
                    try:
                        interp_func = interp1d(
                            satisfied_data['BMI'], 
                            satisfied_data['optimal_week'],
                            kind='linear',
                            fill_value='extrapolate'
                        )
                        
                        # 更新违反约束的点
                        violation_mask = ~satisfied_mask
                        wstar_df.loc[violation_mask, 'optimal_week'] = interp_func(
                            wstar_df.loc[violation_mask, 'BMI']
                        )
                        
                        # 标记为已修正
                        wstar_df.loc[violation_mask, 'constraint_satisfied'] = True
                        
                        logger.info("约束违反情况已通过插值修正")
                        
                    except Exception as e:
                        logger.warning(f"约束违反插值修正失败: {e}")
        
        return wstar_df
    
    def _evaluate_optimization_quality(self, wstar_df: pd.DataFrame, original_df: pd.DataFrame) -> Dict[str, Any]:
        """评估优化质量"""
        
        quality_metrics = {}
        
        try:
            # 1. 风险降低幅度
            baseline_risk = np.mean(wstar_df['risk_value'])  # 假设的基线风险
            optimized_risk = np.mean(wstar_df['risk_value'])
            risk_reduction = (baseline_risk - optimized_risk) / baseline_risk if baseline_risk > 0 else 0
            quality_metrics['avg_risk_reduction'] = float(risk_reduction)
            
            # 2. 约束满足率
            constraint_satisfaction_rate = wstar_df['constraint_satisfied'].mean()
            quality_metrics['constraint_satisfaction_rate'] = float(constraint_satisfaction_rate)
            
            # 3. 时点合理性
            week_range = wstar_df['optimal_week'].max() - wstar_df['optimal_week'].min()
            quality_metrics['optimal_week_range'] = float(week_range)
            quality_metrics['avg_optimal_week'] = float(wstar_df['optimal_week'].mean())
            
            # 4. 成功概率分布
            quality_metrics['avg_success_prob'] = float(wstar_df['success_prob'].mean())
            quality_metrics['min_success_prob'] = float(wstar_df['success_prob'].min())
            quality_metrics['success_prob_std'] = float(wstar_df['success_prob'].std())
            
            # 5. 保序性检查
            is_monotonic = (wstar_df['optimal_week'].diff().dropna() >= -0.1).all()  # 允许小幅下降
            quality_metrics['is_monotonic'] = bool(is_monotonic)
            
            # 6. 多因素风险分布
            quality_metrics['avg_multi_factor_risk'] = float(wstar_df['multi_factor_risk'].mean())
            quality_metrics['max_multi_factor_risk'] = float(wstar_df['multi_factor_risk'].max())
            
        except Exception as e:
            logger.warning(f"优化质量评估失败: {e}")
            quality_metrics = {'error': str(e)}
        
        return quality_metrics


# 工具函数
def create_risk_function(optimizer: MultiObjectiveOptimizer) -> Callable:
    """
    创建风险函数
    
    Parameters:
    -----------
    optimizer : MultiObjectiveOptimizer
        优化器
        
    Returns:
    --------
    callable
        风险函数
    """
    
    def risk_function(week: float, bmi: float, **features) -> float:
        """
        风险函数
        
        Parameters:
        -----------
        week : float
            孕周
        bmi : float
            BMI
        **features : dict
            其他特征
            
        Returns:
        --------
        float
            风险值
        """
        
        default_features = {
            'BMI_used': bmi,
            'age': 30,
            'height': 160,
            'weight': bmi * (160/100)**2,
            'gc_percent': 42,
            'readcount': 1e7
        }
        
        # 更新特征
        default_features.update(features)
        
        return optimizer._multi_objective_risk_function(week, bmi, default_features)
    
    return risk_function


def analyze_risk_components(optimizer: MultiObjectiveOptimizer, 
                          test_cases: List[Dict]) -> pd.DataFrame:
    """
    分析风险组成
    
    Parameters:
    -----------
    optimizer : MultiObjectiveOptimizer
        优化器
    test_cases : list
        测试案例
        
    Returns:
    --------
    pd.DataFrame
        风险分析结果
    """
    
    results = []
    
    for case in test_cases:
        week = case['week']
        bmi = case['bmi']
        features = case.get('features', {})
        
        # 默认特征
        default_features = {
            'BMI_used': bmi,
            'age': 30,
            'height': 160,
            'weight': bmi * (160/100)**2,
            'gc_percent': 42,
            'readcount': 1e7
        }
        default_features.update(features)
        
        # 计算各组成部分
        time_risk = optimizer._calculate_time_risk(week)
        detection_risk = optimizer._calculate_detection_risk(week, bmi, default_features)
        attainment_risk = optimizer._calculate_attainment_risk(week, bmi, default_features)
        multi_factor_risk = optimizer._calculate_multi_factor_risk(week, bmi, default_features)
        uncertainty_risk = optimizer._calculate_uncertainty_risk(week, bmi, default_features)
        
        total_risk = optimizer._multi_objective_risk_function(week, bmi, default_features)
        
        results.append({
            'week': week,
            'bmi': bmi,
            'time_risk': time_risk,
            'detection_risk': detection_risk,
            'attainment_risk': attainment_risk,
            'multi_factor_risk': multi_factor_risk,
            'uncertainty_risk': uncertainty_risk,
            'total_risk': total_risk
        })
    
    return pd.DataFrame(results)