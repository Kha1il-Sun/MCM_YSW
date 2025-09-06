"""
增强的BMI分组模块
支持多因素考虑的BMI分组策略
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeRegressor
from scipy.optimize import minimize_scalar
from typing import Dict, List, Tuple, Optional, Any
import logging

logger = logging.getLogger(__name__)


class EnhancedBMIGrouper:
    """增强的BMI分组器"""
    
    def __init__(self, config: dict):
        """
        初始化分组器
        
        Parameters:
        -----------
        config : dict
            分组配置
        """
        self.config = config
        self.method = config.get('method', 'hybrid')
        self.who_cuts = config.get('who_cuts', [18.5, 25.0, 30.0, 35.0])
        
        # 约束条件
        self.constraints = config.get('constraints', {})
        self.min_group_size = self.constraints.get('min_group_size', 20)
        self.min_cut_distance = self.constraints.get('min_cut_distance', 2.0)
        self.max_groups = self.constraints.get('max_groups', 5)
        
        # 多因素分组
        self.multi_factor_config = config.get('multi_factor_grouping', {})
        self.secondary_factors = self.multi_factor_config.get('secondary_factors', [])
        
        logger.info(f"增强BMI分组器初始化完成")
        logger.info(f"  - 分组方法: {self.method}")
        logger.info(f"  - WHO切点: {self.who_cuts}")
        logger.info(f"  - 最小组大小: {self.min_group_size}")
        
    def find_optimal_groups(self, wstar_results: dict, data: pd.DataFrame) -> Dict[str, Any]:
        """
        找到最优BMI分组
        
        Parameters:
        -----------
        wstar_results : dict
            最优时点结果
        data : pd.DataFrame
            原始数据
            
        Returns:
        --------
        dict
            分组结果
        """
        logger.info("开始增强BMI分组...")
        
        wstar_curve = wstar_results['curve_data']
        
        # 1. 初始分组尝试
        if self.method == 'hybrid':
            cuts, groups = self._hybrid_grouping(wstar_curve, data)
        elif self.method == 'tree':
            cuts, groups = self._tree_based_grouping(wstar_curve, data)
        elif self.method == 'dp':
            cuts, groups = self._dp_grouping(wstar_curve, data)
        else:
            cuts, groups = self._who_standard_grouping(wstar_curve, data)
        
        # 2. 多因素调整
        if self.multi_factor_config.get('enabled', False):
            groups = self._adjust_for_multi_factors(groups, data)
        
        # 3. 质量评估和优化
        groups = self._optimize_group_quality(groups, wstar_curve, data)
        
        # 4. 生成最终结果
        grouping_results = {
            'groups': groups,
            'cuts': cuts,
            'n_groups': len(groups),
            'method_used': self.method,
            'quality_metrics': self._calculate_group_quality(groups, data)
        }
        
        logger.info(f"BMI分组完成，共{len(groups)}组")
        
        return grouping_results
    
    def _hybrid_grouping(self, wstar_curve: pd.DataFrame, data: pd.DataFrame) -> Tuple[List[float], List[Dict]]:
        """混合分组方法"""
        
        # 从WHO切点开始
        base_cuts = self.who_cuts.copy()
        
        # 基于w*曲线的变化率调整
        wstar_curve_sorted = wstar_curve.sort_values('BMI')
        bmi_vals = wstar_curve_sorted['BMI'].values
        week_vals = wstar_curve_sorted['optimal_week'].values
        
        # 计算变化率
        if len(week_vals) > 1:
            derivatives = np.gradient(week_vals, bmi_vals)
            high_change_points = bmi_vals[np.abs(derivatives) > np.percentile(np.abs(derivatives), 75)]
            
            # 在变化率高的区域调整切点
            adjusted_cuts = []
            for cut in base_cuts:
                nearby_changes = high_change_points[np.abs(high_change_points - cut) <= 3.0]
                if len(nearby_changes) > 0:
                    # 移动到最近的高变化点
                    new_cut = nearby_changes[np.argmin(np.abs(nearby_changes - cut))]
                    adjusted_cuts.append(new_cut)
                else:
                    adjusted_cuts.append(cut)
            
            final_cuts = self._validate_cuts(adjusted_cuts, data)
        else:
            final_cuts = base_cuts
        
        # 生成分组
        groups = self._create_groups_from_cuts(final_cuts, wstar_curve, data)
        
        return final_cuts, groups
    
    def _tree_based_grouping(self, wstar_curve: pd.DataFrame, data: pd.DataFrame) -> Tuple[List[float], List[Dict]]:
        """基于决策树的分组"""
        
        # 使用决策树找到最佳分割点
        X = wstar_curve[['BMI']].values
        y = wstar_curve['optimal_week'].values
        
        tree = DecisionTreeRegressor(
            max_leaf_nodes=min(self.max_groups + 1, 6),
            min_samples_leaf=max(5, len(wstar_curve) // 10)
        )
        
        tree.fit(X, y)
        
        # 提取分割点
        tree_structure = tree.tree_
        cuts = []
        
        def extract_cuts(node_id):
            if tree_structure.children_left[node_id] != tree_structure.children_right[node_id]:
                # 非叶子节点，有分割
                threshold = tree_structure.threshold[node_id]
                cuts.append(threshold)
                extract_cuts(tree_structure.children_left[node_id])
                extract_cuts(tree_structure.children_right[node_id])
        
        extract_cuts(0)
        cuts = sorted(set(cuts))
        
        # 验证和调整切点
        final_cuts = self._validate_cuts(cuts, data)
        
        # 生成分组
        groups = self._create_groups_from_cuts(final_cuts, wstar_curve, data)
        
        return final_cuts, groups
    
    def _dp_grouping(self, wstar_curve: pd.DataFrame, data: pd.DataFrame) -> Tuple[List[float], List[Dict]]:
        """动态规划分组"""
        
        # 简化的DP实现
        bmi_sorted = np.sort(data['BMI_used'].dropna().unique())
        
        if len(bmi_sorted) < 10:
            # 数据太少，使用WHO标准
            return self._who_standard_grouping(wstar_curve, data)
        
        # DP寻找最优分割
        n = len(bmi_sorted)
        max_k = min(self.max_groups, n // self.min_group_size)
        
        # 简化：使用等频分割作为基础
        cuts = []
        for i in range(1, max_k):
            percentile = i / max_k * 100
            cut_point = np.percentile(bmi_sorted, percentile)
            cuts.append(cut_point)
        
        # 验证切点
        final_cuts = self._validate_cuts(cuts, data)
        
        # 生成分组
        groups = self._create_groups_from_cuts(final_cuts, wstar_curve, data)
        
        return final_cuts, groups
    
    def _who_standard_grouping(self, wstar_curve: pd.DataFrame, data: pd.DataFrame) -> Tuple[List[float], List[Dict]]:
        """WHO标准分组"""
        
        cuts = self.who_cuts.copy()
        
        # 验证切点
        final_cuts = self._validate_cuts(cuts, data)
        
        # 生成分组
        groups = self._create_groups_from_cuts(final_cuts, wstar_curve, data)
        
        return final_cuts, groups
    
    def _validate_cuts(self, cuts: List[float], data: pd.DataFrame) -> List[float]:
        """验证和调整切点"""
        
        bmi_data = data['BMI_used'].dropna()
        bmi_min, bmi_max = bmi_data.min(), bmi_data.max()
        
        # 过滤有效范围内的切点
        valid_cuts = [cut for cut in cuts if bmi_min < cut < bmi_max]
        valid_cuts = sorted(set(valid_cuts))
        
        # 检查最小间距
        filtered_cuts = []
        for i, cut in enumerate(valid_cuts):
            if i == 0 or (cut - filtered_cuts[-1]) >= self.min_cut_distance:
                filtered_cuts.append(cut)
        
        # 检查组大小
        final_cuts = []
        prev_cut = bmi_min
        
        for cut in filtered_cuts:
            group_size = len(bmi_data[(bmi_data >= prev_cut) & (bmi_data < cut)])
            if group_size >= self.min_group_size:
                final_cuts.append(cut)
                prev_cut = cut
            
        # 检查最后一组
        last_group_size = len(bmi_data[bmi_data >= prev_cut])
        if last_group_size < self.min_group_size and final_cuts:
            # 合并到前一组
            final_cuts.pop()
        
        return final_cuts
    
    def _create_groups_from_cuts(self, cuts: List[float], wstar_curve: pd.DataFrame, data: pd.DataFrame) -> List[Dict]:
        """从切点创建分组"""
        
        groups = []
        bmi_data = data['BMI_used'].dropna()
        bmi_min, bmi_max = bmi_data.min(), bmi_data.max()
        
        # 创建区间
        boundaries = [bmi_min] + cuts + [bmi_max]
        
        for i in range(len(boundaries) - 1):
            lower_bound = boundaries[i]
            upper_bound = boundaries[i + 1]
            
            # 找到该组的数据
            if i == len(boundaries) - 2:  # 最后一组，包含右边界
                group_mask = (data['BMI_used'] >= lower_bound) & (data['BMI_used'] <= upper_bound)
            else:
                group_mask = (data['BMI_used'] >= lower_bound) & (data['BMI_used'] < upper_bound)
            
            group_data = data[group_mask]
            
            if len(group_data) == 0:
                continue
            
            # 找到该组的最优时点
            group_bmi_median = group_data['BMI_used'].median()
            
            # 在wstar曲线中找到最接近的点
            closest_idx = np.argmin(np.abs(wstar_curve['BMI'] - group_bmi_median))
            optimal_timing = wstar_curve.iloc[closest_idx]['optimal_week']
            
            # 计算组内统计
            group_info = {
                'group_id': i + 1,
                'bmi_range': [float(lower_bound), float(upper_bound)],
                'optimal_timing': float(optimal_timing),
                'n_samples': len(group_data),
                'median_bmi': float(group_data['BMI_used'].median()),
                'mean_bmi': float(group_data['BMI_used'].mean()),
                'bmi_std': float(group_data['BMI_used'].std()),
                'expected_success_rate': wstar_curve.iloc[closest_idx]['success_prob'],
                'expected_attain_rate': wstar_curve.iloc[closest_idx]['attain_prob'],
                'risk_level': self._classify_risk_level(wstar_curve.iloc[closest_idx]['risk_value'])
            }
            
            groups.append(group_info)
        
        return groups
    
    def _adjust_for_multi_factors(self, groups: List[Dict], data: pd.DataFrame) -> List[Dict]:
        """基于多因素调整分组"""
        
        if not self.secondary_factors:
            return groups
        
        logger.info("应用多因素调整...")
        
        adjusted_groups = []
        
        for group in groups:
            bmi_lower, bmi_upper = group['bmi_range']
            
            if bmi_lower == bmi_upper:
                group_mask = data['BMI_used'] == bmi_lower
            else:
                group_mask = (data['BMI_used'] >= bmi_lower) & (data['BMI_used'] < bmi_upper)
            
            group_data = data[group_mask]
            
            if len(group_data) == 0:
                continue
            
            # 分析次要因素的分布
            subgroup_analysis = {}
            for factor in self.secondary_factors:
                if factor in group_data.columns:
                    factor_median = group_data[factor].median()
                    factor_std = group_data[factor].std()
                    
                    # 检查是否需要进一步细分
                    if factor_std > factor_median * 0.2:  # 变异系数 > 20%
                        subgroup_analysis[factor] = {
                            'high_variation': True,
                            'median': factor_median,
                            'std': factor_std
                        }
            
            # 如果次要因素变异很大，调整推荐时点
            timing_adjustment = 0.0
            
            if 'age' in subgroup_analysis and subgroup_analysis['age']['high_variation']:
                # 高年龄组可能需要稍微提前
                high_age_ratio = (group_data['age'] > 35).mean() if 'age' in group_data.columns else 0
                if high_age_ratio > 0.3:
                    timing_adjustment -= 0.5
            
            if 'height' in subgroup_analysis and subgroup_analysis['height']['high_variation']:
                # 极端身高可能需要调整
                extreme_height_ratio = ((group_data['height'] < 150) | (group_data['height'] > 175)).mean() if 'height' in group_data.columns else 0
                if extreme_height_ratio > 0.2:
                    timing_adjustment += 0.3
            
            # 应用调整
            adjusted_group = group.copy()
            adjusted_group['optimal_timing'] = max(8.0, min(22.0, group['optimal_timing'] + timing_adjustment))
            adjusted_group['multi_factor_adjustment'] = timing_adjustment
            adjusted_group['subgroup_analysis'] = subgroup_analysis
            
            adjusted_groups.append(adjusted_group)
        
        return adjusted_groups
    
    def _optimize_group_quality(self, groups: List[Dict], wstar_curve: pd.DataFrame, data: pd.DataFrame) -> List[Dict]:
        """优化分组质量"""
        
        # 检查组间差异
        if len(groups) <= 1:
            return groups
        
        timings = [g['optimal_timing'] for g in groups]
        
        # 如果组间时点差异太小，考虑合并
        min_timing_diff = min(timings[i+1] - timings[i] for i in range(len(timings)-1))
        
        if min_timing_diff < 1.0:
            logger.info("检测到组间时点差异较小，尝试优化...")
            
            # 找到差异最小的相邻组进行合并
            optimized_groups = []
            skip_next = False
            
            for i in range(len(groups)):
                if skip_next:
                    skip_next = False
                    continue
                
                current_group = groups[i]
                
                # 检查是否需要与下一组合并
                if i < len(groups) - 1:
                    next_group = groups[i + 1]
                    timing_diff = next_group['optimal_timing'] - current_group['optimal_timing']
                    
                    if timing_diff < 1.0 and (current_group['n_samples'] + next_group['n_samples']) >= self.min_group_size:
                        # 合并组
                        merged_group = self._merge_groups(current_group, next_group)
                        optimized_groups.append(merged_group)
                        skip_next = True
                        continue
                
                optimized_groups.append(current_group)
            
            return optimized_groups
        
        return groups
    
    def _merge_groups(self, group1: Dict, group2: Dict) -> Dict:
        """合并两个组"""
        
        merged_group = {
            'group_id': group1['group_id'],
            'bmi_range': [group1['bmi_range'][0], group2['bmi_range'][1]],
            'optimal_timing': (group1['optimal_timing'] + group2['optimal_timing']) / 2,
            'n_samples': group1['n_samples'] + group2['n_samples'],
            'median_bmi': (group1['median_bmi'] + group2['median_bmi']) / 2,
            'mean_bmi': (group1['mean_bmi'] * group1['n_samples'] + group2['mean_bmi'] * group2['n_samples']) / (group1['n_samples'] + group2['n_samples']),
            'expected_success_rate': (group1['expected_success_rate'] + group2['expected_success_rate']) / 2,
            'expected_attain_rate': (group1['expected_attain_rate'] + group2['expected_attain_rate']) / 2,
            'risk_level': self._classify_risk_level((group1.get('risk_level', 1) + group2.get('risk_level', 1)) / 2),
            'merged_from': [group1['group_id'], group2['group_id']]
        }
        
        return merged_group
    
    def _classify_risk_level(self, risk_value: float) -> str:
        """分类风险水平"""
        
        if risk_value < 1.0:
            return 'Low'
        elif risk_value < 2.0:
            return 'Medium'
        else:
            return 'High'
    
    def _calculate_group_quality(self, groups: List[Dict], data: pd.DataFrame) -> Dict[str, Any]:
        """计算分组质量指标"""
        
        quality = {}
        
        try:
            # 1. 组数合理性
            quality['n_groups'] = len(groups)
            quality['groups_in_range'] = 2 <= len(groups) <= self.max_groups
            
            # 2. 组大小分布
            group_sizes = [g['n_samples'] for g in groups]
            quality['min_group_size'] = min(group_sizes)
            quality['max_group_size'] = max(group_sizes)
            quality['group_size_cv'] = np.std(group_sizes) / np.mean(group_sizes) if group_sizes else 0
            quality['all_groups_sufficient'] = all(size >= self.min_group_size for size in group_sizes)
            
            # 3. 时点差异
            timings = [g['optimal_timing'] for g in groups]
            if len(timings) > 1:
                timing_diffs = [timings[i+1] - timings[i] for i in range(len(timings)-1)]
                quality['min_timing_diff'] = min(timing_diffs)
                quality['max_timing_diff'] = max(timing_diffs)
                quality['avg_timing_diff'] = np.mean(timing_diffs)
                quality['timing_progression_good'] = all(diff >= 0.5 for diff in timing_diffs)
            
            # 4. BMI覆盖率
            total_samples = sum(group_sizes)
            coverage_rate = total_samples / len(data) if len(data) > 0 else 0
            quality['bmi_coverage_rate'] = coverage_rate
            quality['good_coverage'] = coverage_rate >= 0.95
            
            # 5. 整体质量评分
            quality_score = 0
            if quality['groups_in_range']: quality_score += 25
            if quality['all_groups_sufficient']: quality_score += 25
            if quality.get('timing_progression_good', True): quality_score += 25
            if quality['good_coverage']: quality_score += 25
            
            quality['overall_quality_score'] = quality_score
            quality['quality_rating'] = 'Excellent' if quality_score >= 90 else ('Good' if quality_score >= 70 else ('Fair' if quality_score >= 50 else 'Poor'))
            
        except Exception as e:
            logger.warning(f"质量指标计算失败: {e}")
            quality = {'error': str(e)}
        
        return quality