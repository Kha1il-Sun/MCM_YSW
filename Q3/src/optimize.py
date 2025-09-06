"""
个体最优时点求解模块
在10-25周范围内搜索最优检测时点
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
import logging

logger = logging.getLogger(__name__)


class OptimalTimeOptimizer:
    """个体最优时点优化器"""
    
    def __init__(self, config):
        self.config = config
        self.week_range = config['optimization']['week_range']
        self.week_step = config['optimization']['week_step']
        self.gamma = config['risk_function']['gamma']
        self.time_weights = config['risk_function']['time_weights']
        
    def calculate_time_risk(self, week):
        """计算时间段风险"""
        if week <= 12:
            return self.time_weights['early_weeks']
        elif week <= 27:
            return self.time_weights['mid_weeks']
        else:
            return self.time_weights['late_weeks']
    
    def calculate_risk_function(self, week, attain_prob, success_prob):
        """计算风险函数"""
        # 时间段风险
        time_risk = self.calculate_time_risk(week)
        
        # 检测失败风险
        detect_prob = attain_prob * success_prob
        failure_risk = self.gamma * (1 - detect_prob)
        
        # 总风险
        total_risk = time_risk + failure_risk
        
        return total_risk, detect_prob
    
    def optimize_individual(self, attain_model, success_model, features, week_range=None):
        """优化单个样本的最优时点"""
        if week_range is None:
            week_range = self.week_range
        
        # 使用更精细的搜索网格
        weeks = np.arange(week_range[0], week_range[1] + self.week_step, self.week_step)
        
        best_week = None
        best_risk = float('inf')
        best_detect_prob = 0
        results = []
        
        for week in weeks:
            # 更新孕周特征
            features_week = features.copy()
            features_week['孕周'] = week
            
            # 预测达标概率和成功概率
            try:
                attain_prob = attain_model.predict_proba(features_week)[0]
                success_prob = success_model.predict_proba(features_week)[0]
                
                # 计算风险函数
                risk, detect_prob = self.calculate_risk_function(week, attain_prob, success_prob)
                
                results.append({
                    'week': week,
                    'attain_prob': attain_prob,
                    'success_prob': success_prob,
                    'detect_prob': detect_prob,
                    'risk': risk
                })
                
                # 更新最优解
                if risk < best_risk:
                    best_risk = risk
                    best_week = week
                    best_detect_prob = detect_prob
                    
            except Exception as e:
                logger.warning(f"周次 {week} 预测失败: {e}")
                continue
        
        return {
            'optimal_week': best_week,
            'optimal_risk': best_risk,
            'optimal_detect_prob': best_detect_prob,
            'all_results': results
        }
    
    def optimize_batch(self, attain_model, success_model, X, y_attain, y_success, groups):
        """批量优化所有样本"""
        logger.info("开始批量优化个体最优时点...")
        
        results = []
        
        for idx, (_, row) in enumerate(X.iterrows()):
            if idx % 100 == 0:
                logger.info(f"处理进度: {idx}/{len(X)}")
            
            # 优化单个样本
            result = self.optimize_individual(attain_model, success_model, row)
            
            # 添加样本信息
            result['sample_idx'] = idx
            result['group_id'] = groups[idx] if idx < len(groups) else None
            result['actual_attain'] = y_attain[idx] if idx < len(y_attain) else None
            result['actual_success'] = y_success[idx] if idx < len(y_success) else None
            
            results.append(result)
        
        logger.info("批量优化完成")
        return results
    
    def analyze_optimization_results(self, results):
        """分析优化结果"""
        logger.info("分析优化结果...")
        
        # 提取最优时点
        optimal_weeks = [r['optimal_week'] for r in results if r['optimal_week'] is not None]
        optimal_risks = [r['optimal_risk'] for r in results if r['optimal_risk'] is not None]
        optimal_detect_probs = [r['optimal_detect_prob'] for r in results if r['optimal_detect_prob'] is not None]
        
        # 统计信息
        stats = {
            'n_samples': len(results),
            'n_optimized': len(optimal_weeks),
            'optimal_week_mean': np.mean(optimal_weeks) if optimal_weeks else None,
            'optimal_week_std': np.std(optimal_weeks) if optimal_weeks else None,
            'optimal_week_median': np.median(optimal_weeks) if optimal_weeks else None,
            'optimal_week_min': np.min(optimal_weeks) if optimal_weeks else None,
            'optimal_week_max': np.max(optimal_weeks) if optimal_weeks else None,
            'optimal_risk_mean': np.mean(optimal_risks) if optimal_risks else None,
            'optimal_detect_prob_mean': np.mean(optimal_detect_probs) if optimal_detect_probs else None,
        }
        
        logger.info(f"优化结果统计:")
        logger.info(f"  样本数: {stats['n_samples']}")
        logger.info(f"  成功优化: {stats['n_optimized']}")
        logger.info(f"  最优时点均值: {stats['optimal_week_mean']:.2f} 周")
        logger.info(f"  最优时点中位数: {stats['optimal_week_median']:.2f} 周")
        logger.info(f"  最优时点范围: {stats['optimal_week_min']:.2f} - {stats['optimal_week_max']:.2f} 周")
        logger.info(f"  平均检测概率: {stats['optimal_detect_prob_mean']:.4f}")
        
        return stats, results
