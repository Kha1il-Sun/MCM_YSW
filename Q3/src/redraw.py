"""
重采策略模块
实现两阶段重采策略优化
"""

import numpy as np
import pandas as pd
from itertools import product
import logging

logger = logging.getLogger(__name__)


class RedrawStrategy:
    """重采策略优化器"""
    
    def __init__(self, config):
        self.config = config
        self.max_redraw_weeks = config['redraw']['max_redraw_weeks']
        self.redraw_step = config['redraw']['redraw_step']
        self.gamma = config['risk_function']['gamma']
        
    def calculate_expected_risk(self, w1, delta, attain_model, success_model, features):
        """计算期望风险"""
        # 第一次检测
        features_w1 = features.copy()
        features_w1['孕周'] = w1
        
        try:
            p_attain_1 = attain_model.predict_proba(features_w1)[0]
            p_success_1 = success_model.predict_proba(features_w1)[0]
            p_detect_1 = p_attain_1 * p_success_1
            
            # 第一次失败概率
            p_fail_1 = 1 - p_detect_1
            
            # 第二次检测
            w2 = w1 + delta
            features_w2 = features.copy()
            features_w2['孕周'] = w2
            
            p_attain_2 = attain_model.predict_proba(features_w2)[0]
            p_success_2 = success_model.predict_proba(features_w2)[0]
            p_detect_2 = p_attain_2 * p_success_2
            
            # 期望风险
            # E[L] = L(w1) * P(失败于w1) + L(w2) * P(到w2)
            risk_1 = self._calculate_risk(w1, p_detect_1)
            risk_2 = self._calculate_risk(w2, p_detect_2)
            
            expected_risk = risk_1 * p_fail_1 + risk_2 * (1 - p_fail_1)
            
            return expected_risk, p_detect_1, p_detect_2
            
        except Exception as e:
            logger.warning(f"重采策略计算失败 w1={w1}, delta={delta}: {e}")
            return float('inf'), 0, 0
    
    def _calculate_risk(self, week, detect_prob):
        """计算单次检测风险"""
        # 时间段风险
        if week <= 12:
            time_risk = 0.1
        elif week <= 27:
            time_risk = 1.0
        else:
            time_risk = 2.0
        
        # 失败风险
        failure_risk = self.gamma * (1 - detect_prob)
        
        return time_risk + failure_risk
    
    def optimize_redraw_strategy(self, attain_model, success_model, features, 
                                w1_range=(10, 20), delta_range=(1, 4)):
        """优化重采策略"""
        logger.info("开始优化重采策略...")
        
        # 创建搜索网格
        w1_candidates = np.arange(w1_range[0], w1_range[1] + self.redraw_step, self.redraw_step)
        delta_candidates = np.arange(delta_range[0], delta_range[1] + self.redraw_step, self.redraw_step)
        
        best_strategy = None
        best_expected_risk = float('inf')
        all_results = []
        
        for w1, delta in product(w1_candidates, delta_candidates):
            if w1 + delta > 25:  # 确保第二次检测不超过25周
                continue
                
            expected_risk, p_detect_1, p_detect_2 = self.calculate_expected_risk(
                w1, delta, attain_model, success_model, features
            )
            
            all_results.append({
                'w1': w1,
                'delta': delta,
                'w2': w1 + delta,
                'expected_risk': expected_risk,
                'p_detect_1': p_detect_1,
                'p_detect_2': p_detect_2,
                'p_fail_1': 1 - p_detect_1
            })
            
            if expected_risk < best_expected_risk:
                best_expected_risk = expected_risk
                best_strategy = {
                    'w1': w1,
                    'delta': delta,
                    'w2': w1 + delta,
                    'expected_risk': expected_risk,
                    'p_detect_1': p_detect_1,
                    'p_detect_2': p_detect_2,
                    'p_fail_1': 1 - p_detect_1
                }
        
        logger.info(f"重采策略优化完成，最优策略: w1={best_strategy['w1']:.1f}, "
                   f"delta={best_strategy['delta']:.1f}, 期望风险={best_strategy['expected_risk']:.4f}")
        
        return best_strategy, all_results
    
    def optimize_group_redraw_strategies(self, attain_model, success_model, X, groups, 
                                       group_stats, w1_range=(10, 20), delta_range=(1, 4)):
        """为每个BMI组优化重采策略"""
        logger.info("开始为各BMI组优化重采策略...")
        
        group_strategies = {}
        
        for _, group_row in group_stats.iterrows():
            group_name = group_row['group']
            bmi_min = group_row['bmi_min']
            bmi_max = group_row['bmi_max']
            
            # 选择该组的代表性样本
            group_mask = (X['BMI'] >= bmi_min) & (X['BMI'] <= bmi_max)
            group_samples = X[group_mask]
            
            if len(group_samples) == 0:
                logger.warning(f"组 {group_name} 没有样本")
                continue
            
            # 使用组内中位数BMI的样本作为代表
            median_bmi_idx = group_samples['BMI'].sub(group_row['bmi_mean']).abs().idxmin()
            representative_sample = group_samples.loc[median_bmi_idx]
            
            # 优化该组的重采策略
            best_strategy, all_results = self.optimize_redraw_strategy(
                attain_model, success_model, representative_sample,
                w1_range, delta_range
            )
            
            group_strategies[group_name] = {
                'strategy': best_strategy,
                'all_results': all_results,
                'n_samples': len(group_samples),
                'bmi_range': f"{bmi_min:.1f}-{bmi_max:.1f}"
            }
            
            logger.info(f"组 {group_name} 重采策略: w1={best_strategy['w1']:.1f}, "
                       f"delta={best_strategy['delta']:.1f}, "
                       f"期望风险={best_strategy['expected_risk']:.4f}")
        
        return group_strategies
    
    def compare_strategies(self, single_strategy, group_strategies):
        """比较单次检测和重采策略"""
        logger.info("比较单次检测和重采策略...")
        
        comparison = {
            'single_detection': single_strategy,
            'redraw_strategies': group_strategies
        }
        
        # 计算重采策略的平均期望风险
        redraw_risks = [strategy['strategy']['expected_risk'] 
                       for strategy in group_strategies.values()]
        avg_redraw_risk = np.mean(redraw_risks) if redraw_risks else float('inf')
        
        comparison['avg_redraw_risk'] = avg_redraw_risk
        comparison['risk_reduction'] = single_strategy['optimal_risk'] - avg_redraw_risk
        comparison['risk_reduction_pct'] = (comparison['risk_reduction'] / 
                                          single_strategy['optimal_risk'] * 100)
        
        logger.info(f"策略比较结果:")
        logger.info(f"  单次检测平均风险: {single_strategy['optimal_risk']:.4f}")
        logger.info(f"  重采策略平均风险: {avg_redraw_risk:.4f}")
        logger.info(f"  风险降低: {comparison['risk_reduction']:.4f} ({comparison['risk_reduction_pct']:.1f}%)")
        
        return comparison
