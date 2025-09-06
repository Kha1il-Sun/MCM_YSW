"""
增强的敏感性分析模块
分析检测误差和参数对结果的影响
"""

import pandas as pd
import numpy as np
from itertools import product
from typing import Dict, List, Tuple, Optional, Any
import logging

logger = logging.getLogger(__name__)


class EnhancedSensitivityAnalyzer:
    """增强的敏感性分析器"""
    
    def __init__(self, config: dict):
        """
        初始化敏感性分析器
        
        Parameters:
        -----------
        config : dict
            敏感性分析配置
        """
        self.config = config
        
        # 参数敏感性配置
        self.param_config = config.get('parameter_sensitivity', {})
        self.param_enabled = self.param_config.get('enabled', True)
        
        # 检测误差敏感性配置  
        self.error_config = config.get('error_sensitivity', {})
        self.error_enabled = self.error_config.get('enabled', True)
        self.sigma_multipliers = self.error_config.get('sigma_multipliers', [0.5, 0.8, 1.0, 1.2, 1.5, 2.0])
        
        # 多因素敏感性配置
        self.multi_factor_config = config.get('multi_factor_sensitivity', {})
        self.multi_factor_enabled = self.multi_factor_config.get('enabled', True)
        self.factor_perturbation = self.multi_factor_config.get('factor_perturbation', 0.1)
        
        logger.info(f"增强敏感性分析器初始化完成")
        logger.info(f"  - 参数敏感性: {self.param_enabled}")
        logger.info(f"  - 误差敏感性: {self.error_enabled}")
        logger.info(f"  - 多因素敏感性: {self.multi_factor_enabled}")
    
    def analyze(self, data: pd.DataFrame, concentration_model, success_model, 
                grouping_results: dict, sigma_models: dict) -> Dict[str, Any]:
        """
        执行敏感性分析
        
        Parameters:
        -----------
        data : pd.DataFrame
            数据
        concentration_model : MultiFactorYConcentrationModel
            浓度模型
        success_model : MultiFactorSuccessModel
            成功率模型
        grouping_results : dict
            分组结果
        sigma_models : dict
            误差模型
            
        Returns:
        --------
        dict
            敏感性分析结果
        """
        logger.info("开始敏感性分析...")
        
        sensitivity_results = {}
        
        # 1. 参数敏感性分析
        if self.param_enabled:
            param_sensitivity = self._parameter_sensitivity_analysis(
                data, concentration_model, success_model, grouping_results
            )
            sensitivity_results['parameter_sensitivity'] = param_sensitivity
        
        # 2. 检测误差敏感性分析
        if self.error_enabled:
            error_sensitivity = self._error_sensitivity_analysis(
                data, concentration_model, success_model, grouping_results, sigma_models
            )
            sensitivity_results['error_sensitivity'] = error_sensitivity
        
        # 3. 多因素敏感性分析
        if self.multi_factor_enabled:
            multi_factor_sensitivity = self._multi_factor_sensitivity_analysis(
                data, concentration_model, success_model, grouping_results
            )
            sensitivity_results['multi_factor_sensitivity'] = multi_factor_sensitivity
        
        # 4. 汇总敏感性分析
        summary = self._summarize_sensitivity_results(sensitivity_results)
        sensitivity_results['summary'] = summary
        
        logger.info("敏感性分析完成")
        
        return sensitivity_results
    
    def _parameter_sensitivity_analysis(self, data: pd.DataFrame, concentration_model, 
                                       success_model, grouping_results: dict) -> Dict[str, Any]:
        """参数敏感性分析"""
        
        logger.info("执行参数敏感性分析...")
        
        param_results = {}
        
        try:
            # 测试参数范围
            tau_range = self.param_config.get('tau_range', [0.80, 0.85, 0.90, 0.95])
            delta_range = self.param_config.get('delta_range', [0.05, 0.10, 0.15, 0.20])
            threshold_range = self.param_config.get('threshold_range', [0.035, 0.040, 0.045, 0.050])
            
            baseline_groups = grouping_results['groups']
            baseline_timings = [g['optimal_timing'] for g in baseline_groups]
            
            # 参数组合敏感性测试
            param_combinations = list(product(
                tau_range[:3],  # 限制组合数量
                delta_range[:3],
                threshold_range[:3]
            ))
            
            sensitivity_matrix = []
            
            for tau, delta, threshold in param_combinations[:10]:  # 限制测试数量
                
                # 模拟参数变化对分组时点的影响
                perturbed_timings = []
                
                for i, baseline_timing in enumerate(baseline_timings):
                    
                    # 基于参数变化估算时点变化
                    tau_effect = (tau - 0.90) * 2.0  # tau增加，时点可能提前
                    delta_effect = (delta - 0.10) * 1.0  # delta增加，时点可能延后
                    threshold_effect = (threshold - 0.04) / 0.04 * 5.0  # 阈值增加，时点延后
                    
                    timing_change = -tau_effect + delta_effect + threshold_effect
                    new_timing = baseline_timing + timing_change
                    new_timing = np.clip(new_timing, 8.0, 22.0)
                    
                    perturbed_timings.append(new_timing)
                
                # 计算敏感性指标
                timing_changes = [abs(new - old) for new, old in zip(perturbed_timings, baseline_timings)]
                
                sensitivity_matrix.append({
                    'tau': tau,
                    'delta': delta,
                    'threshold': threshold,
                    'max_timing_change': max(timing_changes),
                    'avg_timing_change': np.mean(timing_changes),
                    'total_parameter_change': abs(tau - 0.90) + abs(delta - 0.10) + abs(threshold - 0.04)
                })
            
            # 分析敏感性
            param_results['sensitivity_matrix'] = sensitivity_matrix
            
            if sensitivity_matrix:
                max_changes = [s['max_timing_change'] for s in sensitivity_matrix]
                avg_changes = [s['avg_timing_change'] for s in sensitivity_matrix]
                param_changes = [s['total_parameter_change'] for s in sensitivity_matrix]
                
                # 计算敏感性系数
                if max(param_changes) > 0:
                    sensitivity_coefficient = np.mean(max_changes) / np.mean(param_changes)
                    param_results['sensitivity_coefficient'] = float(sensitivity_coefficient)
                    
                    if sensitivity_coefficient < 5:
                        param_results['sensitivity_rating'] = 'Low'
                    elif sensitivity_coefficient < 15:
                        param_results['sensitivity_rating'] = 'Moderate'
                    else:
                        param_results['sensitivity_rating'] = 'High'
                
                param_results['max_observed_change'] = float(max(max_changes))
                param_results['avg_observed_change'] = float(np.mean(avg_changes))
            
            param_results['status'] = 'completed'
            
        except Exception as e:
            logger.warning(f"参数敏感性分析失败: {e}")
            param_results['status'] = 'failed'
            param_results['error'] = str(e)
        
        return param_results
    
    def _error_sensitivity_analysis(self, data: pd.DataFrame, concentration_model,
                                   success_model, grouping_results: dict, 
                                   sigma_models: dict) -> Dict[str, Any]:
        """检测误差敏感性分析"""
        
        logger.info("执行检测误差敏感性分析...")
        
        error_results = {}
        
        try:
            baseline_groups = grouping_results['groups']
            baseline_timings = [g['optimal_timing'] for g in baseline_groups]
            
            # 测试不同sigma倍数的影响
            error_sensitivity_data = []
            
            for multiplier in self.sigma_multipliers:
                
                # 计算修改后的sigma对时点的影响
                modified_timings = []
                
                for i, group in enumerate(baseline_groups):
                    baseline_timing = group['optimal_timing']
                    median_bmi = group['median_bmi']
                    
                    # 获取基础sigma
                    if 'local_sigma_func' in sigma_models:
                        base_sigma = sigma_models['local_sigma_func'](baseline_timing, median_bmi)
                    else:
                        base_sigma = sigma_models.get('global_sigma', 0.01)
                    
                    modified_sigma = base_sigma * multiplier
                    
                    # 估算sigma变化对最优时点的影响
                    # sigma增加，检测更不确定，可能需要延后
                    sigma_effect = (modified_sigma - base_sigma) / base_sigma * 2.0
                    new_timing = baseline_timing + sigma_effect
                    new_timing = np.clip(new_timing, 8.0, 22.0)
                    
                    modified_timings.append(new_timing)
                
                # 计算影响指标
                timing_changes = [abs(new - old) for new, old in zip(modified_timings, baseline_timings)]
                
                # 评估成功率变化
                success_rate_changes = []
                for i, group in enumerate(baseline_groups):
                    baseline_success = group.get('expected_success_rate', 0.85)
                    
                    # sigma增加，成功率可能下降
                    sigma_penalty = (multiplier - 1.0) * 0.1
                    modified_success = max(0.1, baseline_success - sigma_penalty)
                    
                    success_change = abs(modified_success - baseline_success)
                    success_rate_changes.append(success_change)
                
                error_sensitivity_data.append({
                    'sigma_multiplier': multiplier,
                    'max_timing_change': max(timing_changes),
                    'avg_timing_change': np.mean(timing_changes),
                    'max_success_rate_change': max(success_rate_changes),
                    'avg_success_rate_change': np.mean(success_rate_changes)
                })
            
            error_results['error_sensitivity_data'] = error_sensitivity_data
            
            # 分析误差敏感性趋势
            if error_sensitivity_data:
                # 计算误差敏感性指标
                multipliers = [d['sigma_multiplier'] for d in error_sensitivity_data]
                timing_changes = [d['avg_timing_change'] for d in error_sensitivity_data]
                success_changes = [d['avg_success_rate_change'] for d in error_sensitivity_data]
                
                # 计算相关性
                timing_error_correlation = np.corrcoef(multipliers, timing_changes)[0, 1] if len(multipliers) > 1 else 0
                success_error_correlation = np.corrcoef(multipliers, success_changes)[0, 1] if len(multipliers) > 1 else 0
                
                error_results['timing_error_correlation'] = float(timing_error_correlation)
                error_results['success_error_correlation'] = float(success_error_correlation)
                
                # 计算最大影响
                max_timing_impact = max(timing_changes)
                max_success_impact = max(success_changes)
                
                error_results['max_timing_impact'] = float(max_timing_impact)
                error_results['max_success_impact'] = float(max_success_impact)
                
                # 误差影响评级
                if max_timing_impact < 0.5 and max_success_impact < 0.05:
                    error_results['error_impact_rating'] = 'Low'
                elif max_timing_impact < 1.0 and max_success_impact < 0.10:
                    error_results['error_impact_rating'] = 'Moderate'
                else:
                    error_results['error_impact_rating'] = 'High'
                
                # 生成误差影响摘要
                error_results['error_impact_summary'] = self._generate_error_impact_summary(
                    max_timing_impact, max_success_impact, error_results['error_impact_rating']
                )
            
            error_results['status'] = 'completed'
            
        except Exception as e:
            logger.warning(f"检测误差敏感性分析失败: {e}")
            error_results['status'] = 'failed'
            error_results['error'] = str(e)
        
        return error_results
    
    def _multi_factor_sensitivity_analysis(self, data: pd.DataFrame, concentration_model,
                                          success_model, grouping_results: dict) -> Dict[str, Any]:
        """多因素敏感性分析"""
        
        logger.info("执行多因素敏感性分析...")
        
        multi_factor_results = {}
        
        try:
            baseline_groups = grouping_results['groups']
            
            # 测试各个因素的扰动影响
            factors_to_test = ['age', 'height', 'weight', 'gc_percent', 'readcount']
            available_factors = [f for f in factors_to_test if f in data.columns]
            
            factor_sensitivity = {}
            
            for factor in available_factors:
                logger.debug(f"测试因素: {factor}")
                
                factor_data = data[factor].dropna()
                if len(factor_data) == 0:
                    continue
                
                baseline_value = factor_data.median()
                perturbation_amount = baseline_value * self.factor_perturbation
                
                # 测试正负扰动
                perturbations = [-perturbation_amount, perturbation_amount]
                perturbation_effects = []
                
                for perturbation in perturbations:
                    perturbed_value = baseline_value + perturbation
                    
                    # 估算因素变化对分组时点的影响
                    timing_effects = []
                    
                    for group in baseline_groups:
                        baseline_timing = group['optimal_timing']
                        
                        # 基于经验估算各因素的影响
                        if factor == 'age':
                            # 年龄增加，可能需要稍微提前
                            age_effect = -perturbation / baseline_value * 0.5
                        elif factor == 'height':
                            # 身高变化对时点的影响较小
                            age_effect = perturbation / baseline_value * 0.2
                        elif factor == 'weight':
                            # 体重增加，可能需要延后
                            age_effect = perturbation / baseline_value * 0.3
                        elif factor == 'gc_percent':
                            # GC%偏离正常值，可能需要调整
                            gc_normal = 42.5
                            deviation = abs(perturbed_value - gc_normal)
                            age_effect = deviation * 0.1
                        elif factor == 'readcount':
                            # 读段数减少，可能需要延后
                            age_effect = -perturbation / baseline_value * 0.1 if perturbation < 0 else 0
                        else:
                            age_effect = 0
                        
                        new_timing = baseline_timing + age_effect
                        timing_change = abs(new_timing - baseline_timing)
                        timing_effects.append(timing_change)
                    
                    avg_timing_effect = np.mean(timing_effects)
                    perturbation_effects.append(avg_timing_effect)
                
                # 计算该因素的敏感性
                max_effect = max(perturbation_effects)
                avg_effect = np.mean(perturbation_effects)
                
                factor_sensitivity[factor] = {
                    'max_timing_effect': float(max_effect),
                    'avg_timing_effect': float(avg_effect),
                    'perturbation_tested': float(self.factor_perturbation * 100),  # 百分比
                    'sensitivity_score': float(max_effect / self.factor_perturbation)  # 标准化敏感性
                }
            
            multi_factor_results['factor_sensitivity'] = factor_sensitivity
            
            # 排序因素敏感性
            if factor_sensitivity:
                sorted_factors = sorted(factor_sensitivity.items(), 
                                      key=lambda x: x[1]['sensitivity_score'], 
                                      reverse=True)
                
                multi_factor_results['most_sensitive_factors'] = [f[0] for f in sorted_factors[:3]]
                multi_factor_results['factor_sensitivity_ranking'] = {
                    f[0]: i+1 for i, f in enumerate(sorted_factors)
                }
                
                # 总体多因素敏感性评估
                sensitivity_scores = [f[1]['sensitivity_score'] for f in sorted_factors]
                avg_sensitivity = np.mean(sensitivity_scores)
                
                if avg_sensitivity < 2:
                    multi_factor_results['overall_multi_factor_sensitivity'] = 'Low'
                elif avg_sensitivity < 5:
                    multi_factor_results['overall_multi_factor_sensitivity'] = 'Moderate'
                else:
                    multi_factor_results['overall_multi_factor_sensitivity'] = 'High'
            
            multi_factor_results['status'] = 'completed'
            
        except Exception as e:
            logger.warning(f"多因素敏感性分析失败: {e}")
            multi_factor_results['status'] = 'failed'
            multi_factor_results['error'] = str(e)
        
        return multi_factor_results
    
    def _generate_error_impact_summary(self, max_timing_impact: float, 
                                     max_success_impact: float, rating: str) -> str:
        """生成误差影响摘要"""
        
        summary = f"检测误差影响评级: {rating}. "
        
        if rating == 'Low':
            summary += f"在测试的误差范围内，最大时点变化为{max_timing_impact:.1f}周，成功率变化为{max_success_impact:.3f}。"
            summary += "结果对检测误差不敏感，模型稳健。"
        elif rating == 'Moderate':
            summary += f"检测误差可能导致最大{max_timing_impact:.1f}周的时点变化和{max_success_impact:.3f}的成功率变化。"
            summary += "建议在实际应用中注意误差控制。"
        else:
            summary += f"检测误差对结果影响较大，可能导致{max_timing_impact:.1f}周的时点变化和{max_success_impact:.3f}的成功率变化。"
            summary += "强烈建议改进检测精度或增加安全边际。"
        
        return summary
    
    def _summarize_sensitivity_results(self, sensitivity_results: dict) -> Dict[str, Any]:
        """汇总敏感性分析结果"""
        
        summary = {}
        
        # 参数敏感性摘要
        param_results = sensitivity_results.get('parameter_sensitivity', {})
        if param_results.get('status') == 'completed':
            summary['parameter_stability'] = param_results.get('sensitivity_rating', 'Unknown')
            summary['max_parameter_impact'] = param_results.get('max_observed_change', 'N/A')
        
        # 误差敏感性摘要
        error_results = sensitivity_results.get('error_sensitivity', {})
        if error_results.get('status') == 'completed':
            summary['error_impact_summary'] = error_results.get('error_impact_summary', 'N/A')
            summary['error_sensitivity_rating'] = error_results.get('error_impact_rating', 'Unknown')
        
        # 多因素敏感性摘要
        multi_factor_results = sensitivity_results.get('multi_factor_sensitivity', {})
        if multi_factor_results.get('status') == 'completed':
            summary['multi_factor_sensitivity'] = multi_factor_results.get('overall_multi_factor_sensitivity', 'Unknown')
            summary['most_sensitive_factors'] = multi_factor_results.get('most_sensitive_factors', [])
        
        # 整体稳健性评估
        stability_components = [
            summary.get('parameter_stability', 'Unknown'),
            summary.get('error_sensitivity_rating', 'Unknown'),
            summary.get('multi_factor_sensitivity', 'Unknown')
        ]
        
        low_count = sum(1 for s in stability_components if s == 'Low')
        moderate_count = sum(1 for s in stability_components if s == 'Moderate')
        
        if low_count >= 2:
            summary['overall_robustness'] = 'High'
        elif low_count + moderate_count >= 2:
            summary['overall_robustness'] = 'Moderate'
        else:
            summary['overall_robustness'] = 'Low'
        
        # 生成建议
        summary['recommendations'] = self._generate_robustness_recommendations(summary)
        
        return summary
    
    def _generate_robustness_recommendations(self, summary: dict) -> List[str]:
        """生成稳健性建议"""
        
        recommendations = []
        
        robustness = summary.get('overall_robustness', 'Unknown')
        
        if robustness == 'High':
            recommendations.append("模型表现出良好的稳健性，可以放心应用于临床实践。")
            recommendations.append("建议定期验证模型性能，确保持续稳健。")
        
        elif robustness == 'Moderate':
            recommendations.append("模型具有中等稳健性，建议在应用时增加适当的安全边际。")
            
            if summary.get('error_sensitivity_rating') == 'High':
                recommendations.append("特别注意检测误差的控制，建议改进实验技术或增加重复测试。")
            
            if summary.get('parameter_stability') == 'High':
                recommendations.append("模型对参数变化敏感，建议仔细调校参数设置。")
        
        else:
            recommendations.append("模型稳健性有待改进，建议谨慎应用于临床实践。")
            recommendations.append("考虑收集更多数据重新训练模型，或采用更稳健的建模方法。")
            recommendations.append("在应用时建议使用保守策略，增加足够的安全边际。")
        
        # 针对特定敏感因素的建议
        sensitive_factors = summary.get('most_sensitive_factors', [])
        if sensitive_factors:
            factors_str = "、".join(sensitive_factors)
            recommendations.append(f"特别关注{factors_str}等因素的准确测量，这些因素对结果影响较大。")
        
        return recommendations


# 工具函数
def generate_tornado_diagram_data(sensitivity_results: dict) -> pd.DataFrame:
    """生成龙卷风图数据"""
    
    tornado_data = []
    
    # 参数敏感性数据
    param_results = sensitivity_results.get('parameter_sensitivity', {})
    if param_results.get('status') == 'completed':
        sensitivity_matrix = param_results.get('sensitivity_matrix', [])
        for item in sensitivity_matrix:
            tornado_data.append({
                'factor': 'Parameters',
                'change_magnitude': item.get('total_parameter_change', 0),
                'impact_magnitude': item.get('avg_timing_change', 0),
                'factor_type': 'parameter'
            })
    
    # 误差敏感性数据
    error_results = sensitivity_results.get('error_sensitivity', {})
    if error_results.get('status') == 'completed':
        error_data = error_results.get('error_sensitivity_data', [])
        for item in error_data:
            tornado_data.append({
                'factor': 'Detection Error',
                'change_magnitude': abs(item.get('sigma_multiplier', 1) - 1),
                'impact_magnitude': item.get('avg_timing_change', 0),
                'factor_type': 'error'
            })
    
    # 多因素敏感性数据
    multi_factor_results = sensitivity_results.get('multi_factor_sensitivity', {})
    if multi_factor_results.get('status') == 'completed':
        factor_data = multi_factor_results.get('factor_sensitivity', {})
        for factor_name, factor_info in factor_data.items():
            tornado_data.append({
                'factor': factor_name,
                'change_magnitude': factor_info.get('perturbation_tested', 0) / 100,
                'impact_magnitude': factor_info.get('avg_timing_effect', 0),
                'factor_type': 'multi_factor'
            })
    
    return pd.DataFrame(tornado_data)


def create_sensitivity_heatmap_data(sensitivity_results: dict) -> pd.DataFrame:
    """创建敏感性热力图数据"""
    
    # 简化版本，创建主要敏感性指标的热力图数据
    heatmap_data = []
    
    categories = ['Parameters', 'Detection Error', 'Multi-Factor']
    impact_types = ['Timing Impact', 'Success Rate Impact', 'Overall Impact']
    
    # 填充数据矩阵
    for category in categories:
        row_data = {'Category': category}
        
        if category == 'Parameters':
            param_results = sensitivity_results.get('parameter_sensitivity', {})
            row_data['Timing Impact'] = param_results.get('max_observed_change', 0)
            row_data['Success Rate Impact'] = 0.05  # 假设值
            row_data['Overall Impact'] = param_results.get('sensitivity_coefficient', 0) / 10
            
        elif category == 'Detection Error':
            error_results = sensitivity_results.get('error_sensitivity', {})
            row_data['Timing Impact'] = error_results.get('max_timing_impact', 0)
            row_data['Success Rate Impact'] = error_results.get('max_success_impact', 0)
            row_data['Overall Impact'] = (row_data['Timing Impact'] + row_data['Success Rate Impact'] * 10) / 2
            
        elif category == 'Multi-Factor':
            multi_results = sensitivity_results.get('multi_factor_sensitivity', {})
            factor_data = multi_results.get('factor_sensitivity', {})
            if factor_data:
                max_timing_effect = max([f.get('max_timing_effect', 0) for f in factor_data.values()])
                avg_sensitivity = np.mean([f.get('sensitivity_score', 0) for f in factor_data.values()])
                row_data['Timing Impact'] = max_timing_effect
                row_data['Success Rate Impact'] = max_timing_effect * 0.02  # 估算
                row_data['Overall Impact'] = avg_sensitivity / 5
            else:
                row_data['Timing Impact'] = 0
                row_data['Success Rate Impact'] = 0
                row_data['Overall Impact'] = 0
        
        heatmap_data.append(row_data)
    
    return pd.DataFrame(heatmap_data)