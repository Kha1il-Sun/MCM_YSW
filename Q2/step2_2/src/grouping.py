"""
BMI分组模块
WHO起点 + 决策树/DP微调 + 约束（间距/样本量）
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Any, Optional
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score
import logging

logger = logging.getLogger(__name__)


def find_bmi_cuts(wstar_curve: pd.DataFrame, who_cuts: List[float] = [18.5, 25.0, 30.0],
                 method: str = "hybrid", delta: float = 2.0,
                 min_group_n: int = 30, min_cut_distance: float = 1.0,
                 search: str = "tree", tree_params: Dict[str, Any] = None,
                 dp_params: Dict[str, Any] = None, custom_cuts: List[float] = None) -> Tuple[List[float], pd.DataFrame]:
    """
    找到BMI切点
    
    Parameters:
    -----------
    wstar_curve : pd.DataFrame
        w*(b) 曲线数据
    who_cuts : list
        WHO标准切点
    method : str
        方法: "custom", "hybrid", "tree", "dp"
    delta : float
        微调窗口
    min_group_n : int
        最小组内样本量
    min_cut_distance : float
        最小切点间距
    search : str
        搜索方法: "tree", "dp"
    tree_params : dict, optional
        决策树参数
    dp_params : dict, optional
        DP参数
    custom_cuts : list, optional
        自定义切点
        
    Returns:
    --------
    tuple
        (切点列表, 分组结果表)
    """
    logger.info(f"寻找BMI切点: 方法={method}, WHO切点={who_cuts}")
    
    if method == "custom":
        # 自定义分组方法
        cuts = _custom_method(wstar_curve, custom_cuts or who_cuts)
    elif method == "hybrid":
        # 混合方法：WHO起点 + 微调
        cuts = _hybrid_method(wstar_curve, who_cuts, delta, min_group_n, 
                            min_cut_distance, search, tree_params, dp_params)
    elif method == "tree":
        # 纯决策树方法
        cuts = _tree_method(wstar_curve, tree_params or {})
    elif method == "dp":
        # 纯DP方法
        cuts = _dp_method(wstar_curve, dp_params or {})
    else:
        raise ValueError(f"不支持的方法: {method}")
    
    # 应用约束（自定义方法跳过约束检查）
    if method != "custom":
        cuts = _apply_constraints(cuts, wstar_curve, min_group_n, min_cut_distance)
    
    # 创建分组结果表
    groups_df = _create_groups_table(wstar_curve, cuts)
    
    logger.info(f"BMI切点确定: {cuts}")
    
    return cuts, groups_df


def _custom_method(wstar_curve: pd.DataFrame, custom_cuts: List[float]) -> List[float]:
    """
    自定义分组方法：直接使用指定的切点
    """
    logger.info("使用自定义分组方法...")
    logger.info(f"自定义切点: {custom_cuts}")
    
    # 过滤掉超出数据范围的切点
    bmi_min = wstar_curve['BMI'].min()
    bmi_max = wstar_curve['BMI'].max()
    
    valid_cuts = [cut for cut in custom_cuts if bmi_min <= cut <= bmi_max]
    
    if not valid_cuts:
        logger.warning("所有自定义切点都超出数据范围，使用数据范围边界")
        valid_cuts = [bmi_min, bmi_max]
    
    logger.info(f"有效切点: {valid_cuts}")
    
    return valid_cuts


def _hybrid_method(wstar_curve: pd.DataFrame, who_cuts: List[float], delta: float,
                  min_group_n: int, min_cut_distance: float, search: str,
                  tree_params: Dict[str, Any], dp_params: Dict[str, Any]) -> List[float]:
    """
    混合方法：WHO起点 + 微调
    """
    logger.info("使用混合方法寻找BMI切点...")
    
    # 以WHO切点为起点
    initial_cuts = who_cuts.copy()
    
    # 在WHO切点附近微调
    refined_cuts = []
    
    for cut in initial_cuts:
        # 定义微调范围
        search_min = max(wstar_curve['BMI'].min(), cut - delta)
        search_max = min(wstar_curve['BMI'].max(), cut + delta)
        
        # 在范围内搜索最优切点
        if search == "tree":
            optimal_cut = _find_optimal_cut_tree(wstar_curve, search_min, search_max, tree_params)
        elif search == "dp":
            optimal_cut = _find_optimal_cut_dp(wstar_curve, search_min, search_max, dp_params)
        else:
            optimal_cut = cut  # 不微调
        
        refined_cuts.append(optimal_cut)
    
    return refined_cuts


def _tree_method(wstar_curve: pd.DataFrame, tree_params: Dict[str, Any]) -> List[float]:
    """
    决策树方法
    """
    logger.info("使用决策树方法寻找BMI切点...")
    
    # 设置默认参数
    default_params = {
        'max_depth': 3,
        'min_samples_split': 20,
        'min_samples_leaf': 10
    }
    params = {**default_params, **tree_params}
    
    # 创建决策树
    tree = DecisionTreeRegressor(**params)
    
    # 拟合模型
    X = wstar_curve[['BMI']].values
    y = wstar_curve['w_star_smooth'].values if 'w_star_smooth' in wstar_curve.columns else wstar_curve['w_star'].values
    
    tree.fit(X, y)
    
    # 提取切点
    cuts = _extract_tree_cuts(tree, wstar_curve['BMI'].min(), wstar_curve['BMI'].max())
    
    return cuts


def _dp_method(wstar_curve: pd.DataFrame, dp_params: Dict[str, Any]) -> List[float]:
    """
    动态规划方法
    """
    logger.info("使用动态规划方法寻找BMI切点...")
    
    # 设置默认参数
    default_params = {
        'penalty_weight': 0.1,
        'max_cuts': 5
    }
    params = {**default_params, **dp_params}
    
    # 实现DP算法
    cuts = _dp_optimization(wstar_curve, params)
    
    return cuts


def _find_optimal_cut_tree(wstar_curve: pd.DataFrame, search_min: float, search_max: float,
                          tree_params: Dict[str, Any]) -> float:
    """
    使用决策树在范围内寻找最优切点
    """
    # 筛选范围内的数据
    mask = (wstar_curve['BMI'] >= search_min) & (wstar_curve['BMI'] <= search_max)
    range_data = wstar_curve[mask].copy()
    
    if len(range_data) < 10:
        return (search_min + search_max) / 2
    
    # 创建决策树
    tree = DecisionTreeRegressor(max_depth=2, min_samples_leaf=5)
    
    X = range_data[['BMI']].values
    y = range_data['w_star_smooth'].values if 'w_star_smooth' in range_data.columns else range_data['w_star'].values
    
    tree.fit(X, y)
    
    # 提取第一个切点
    cuts = _extract_tree_cuts(tree, search_min, search_max)
    
    if cuts:
        return cuts[0]
    else:
        return (search_min + search_max) / 2


def _find_optimal_cut_dp(wstar_curve: pd.DataFrame, search_min: float, search_max: float,
                        dp_params: Dict[str, Any]) -> float:
    """
    使用DP在范围内寻找最优切点
    """
    # 筛选范围内的数据
    mask = (wstar_curve['BMI'] >= search_min) & (wstar_curve['BMI'] <= search_max)
    range_data = wstar_curve[mask].copy()
    
    if len(range_data) < 10:
        return (search_min + search_max) / 2
    
    # 简化的DP：寻找使组内方差最小的切点
    best_cut = search_min
    best_score = float('inf')
    
    # 在范围内搜索
    search_points = np.linspace(search_min, search_max, 20)
    
    for cut in search_points:
        # 计算切点两侧的方差
        left_data = range_data[range_data['BMI'] < cut]
        right_data = range_data[range_data['BMI'] >= cut]
        
        if len(left_data) < 5 or len(right_data) < 5:
            continue
        
        # 计算组内方差
        left_var = left_data['w_star_smooth'].var() if 'w_star_smooth' in left_data.columns else left_data['w_star'].var()
        right_var = right_data['w_star_smooth'].var() if 'w_star_smooth' in right_data.columns else right_data['w_star'].var()
        
        # 加权平均方差
        total_n = len(left_data) + len(right_data)
        weighted_var = (len(left_data) * left_var + len(right_data) * right_var) / total_n
        
        if weighted_var < best_score:
            best_score = weighted_var
            best_cut = cut
    
    return best_cut


def _extract_tree_cuts(tree: DecisionTreeRegressor, min_bmi: float, max_bmi: float) -> List[float]:
    """
    从决策树中提取切点
    """
    cuts = []
    
    def extract_cuts_recursive(node, depth=0):
        if tree.tree_.children_left[node] != tree.tree_.children_right[node]:  # 不是叶子节点
            feature = tree.tree_.feature[node]
            threshold = tree.tree_.threshold[node]
            
            if feature == 0:  # BMI特征
                cuts.append(threshold)
            
            extract_cuts_recursive(tree.tree_.children_left[node], depth + 1)
            extract_cuts_recursive(tree.tree_.children_right[node], depth + 1)
    
    extract_cuts_recursive(0)
    
    # 过滤在范围内的切点
    cuts = [cut for cut in cuts if min_bmi < cut < max_bmi]
    cuts.sort()
    
    return cuts


def _dp_optimization(wstar_curve: pd.DataFrame, params: Dict[str, Any]) -> List[float]:
    """
    动态规划优化
    """
    penalty_weight = params.get('penalty_weight', 0.1)
    max_cuts = params.get('max_cuts', 5)
    
    # 简化的DP实现
    bmi_values = wstar_curve['BMI'].values
    w_star_values = wstar_curve['w_star_smooth'].values if 'w_star_smooth' in wstar_curve.columns else wstar_curve['w_star'].values
    
    n = len(bmi_values)
    
    # 计算所有可能切点的成本
    costs = {}
    for i in range(1, n-1):
        for j in range(i+1, n):
            cut1, cut2 = bmi_values[i], bmi_values[j]
            
            # 计算三个组的方差
            group1 = w_star_values[bmi_values < cut1]
            group2 = w_star_values[(bmi_values >= cut1) & (bmi_values < cut2)]
            group3 = w_star_values[bmi_values >= cut2]
            
            if len(group1) < 5 or len(group2) < 5 or len(group3) < 5:
                continue
            
            # 计算总成本
            var1, var2, var3 = group1.var(), group2.var(), group3.var()
            total_var = (len(group1) * var1 + len(group2) * var2 + len(group3) * var3) / n
            
            # 添加惩罚项
            penalty = penalty_weight * (cut2 - cut1)
            cost = total_var + penalty
            
            costs[(cut1, cut2)] = cost
    
    # 找到最小成本的切点
    if costs:
        best_cuts = min(costs.keys(), key=lambda x: costs[x])
        return list(best_cuts)
    else:
        return []


def _apply_constraints(cuts: List[float], wstar_curve: pd.DataFrame,
                      min_group_n: int, min_cut_distance: float) -> List[float]:
    """
    应用约束条件
    """
    logger.info("应用约束条件...")
    
    if not cuts:
        return cuts
    
    # 排序
    cuts = sorted(cuts)
    
    # 应用最小间距约束
    constrained_cuts = [cuts[0]]
    for cut in cuts[1:]:
        if cut - constrained_cuts[-1] >= min_cut_distance:
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
        if left_count >= min_group_n and right_count >= min_group_n:
            final_cuts.append(cut)
        else:
            logger.warning(f"切点 {cut:.2f} 不满足样本量约束: 左={left_count}, 右={right_count}")
    
    logger.info(f"约束后切点: {final_cuts}")
    
    return final_cuts


def _create_groups_table(wstar_curve: pd.DataFrame, cuts: List[float]) -> pd.DataFrame:
    """
    创建分组结果表
    """
    logger.info("创建分组结果表...")
    
    if not cuts:
        # 无切点，所有数据为一组
        groups_df = pd.DataFrame({
            'group_id': [1],
            'bmi_min': [wstar_curve['BMI'].min()],
            'bmi_max': [wstar_curve['BMI'].max()],
            'bmi_mean': [wstar_curve['BMI'].mean()],
            'n_points': [len(wstar_curve)],
            'optimal_time': [wstar_curve['w_star_smooth'].mean() if 'w_star_smooth' in wstar_curve.columns else wstar_curve['w_star'].mean()],
            'time_std': [wstar_curve['w_star_smooth'].std() if 'w_star_smooth' in wstar_curve.columns else wstar_curve['w_star'].std()],
            'mean_risk': [wstar_curve['min_risk'].mean()]
        })
        return groups_df
    
    # 创建分组
    groups = []
    bmi_values = wstar_curve['BMI'].values
    w_star_values = wstar_curve['w_star_smooth'].values if 'w_star_smooth' in wstar_curve.columns else wstar_curve['w_star'].values
    risk_values = wstar_curve['min_risk'].values
    
    # 第一组
    mask = bmi_values < cuts[0]
    if mask.sum() > 0:
        groups.append({
            'group_id': 1,
            'bmi_min': bmi_values[mask].min(),
            'bmi_max': bmi_values[mask].max(),
            'bmi_mean': bmi_values[mask].mean(),
            'n_points': mask.sum(),
            'optimal_time': w_star_values[mask].mean(),
            'time_std': w_star_values[mask].std(),
            'mean_risk': risk_values[mask].mean()
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
                'optimal_time': w_star_values[mask].mean(),
                'time_std': w_star_values[mask].std(),
                'mean_risk': risk_values[mask].mean()
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
            'optimal_time': w_star_values[mask].mean(),
            'time_std': w_star_values[mask].std(),
            'mean_risk': risk_values[mask].mean()
        })
    
    groups_df = pd.DataFrame(groups)
    
    logger.info(f"创建了 {len(groups_df)} 个分组")
    
    return groups_df


def evaluate_grouping_quality(groups_df: pd.DataFrame, wstar_curve: pd.DataFrame) -> Dict[str, Any]:
    """
    评估分组质量
    """
    logger.info("评估分组质量...")
    
    evaluation = {
        'n_groups': len(groups_df),
        'total_points': len(wstar_curve),
        'group_sizes': groups_df['n_points'].tolist(),
        'min_group_size': groups_df['n_points'].min(),
        'max_group_size': groups_df['n_points'].max(),
        'mean_group_size': groups_df['n_points'].mean(),
        'group_size_std': groups_df['n_points'].std()
    }
    
    # 计算组内方差
    group_variances = []
    for _, group in groups_df.iterrows():
        mask = (wstar_curve['BMI'] >= group['bmi_min']) & (wstar_curve['BMI'] <= group['bmi_max'])
        group_data = wstar_curve[mask]
        if len(group_data) > 1:
            var = group_data['w_star_smooth'].var() if 'w_star_smooth' in group_data.columns else group_data['w_star'].var()
            group_variances.append(var)
    
    evaluation['group_variances'] = group_variances
    evaluation['mean_group_variance'] = np.mean(group_variances) if group_variances else 0
    evaluation['total_variance'] = wstar_curve['w_star_smooth'].var() if 'w_star_smooth' in wstar_curve.columns else wstar_curve['w_star'].var()
    
    # 计算组间差异
    if len(groups_df) > 1:
        group_means = groups_df['optimal_time'].values
        evaluation['group_mean_std'] = np.std(group_means)
        evaluation['group_mean_range'] = np.max(group_means) - np.min(group_means)
    
    logger.info(f"分组质量评估: {evaluation['n_groups']} 组, "
               f"平均组大小={evaluation['mean_group_size']:.1f}, "
               f"组内方差={evaluation['mean_group_variance']:.4f}")
    
    return evaluation


def create_grouping_report(groups_df: pd.DataFrame, evaluation: Dict[str, Any]) -> str:
    """
    创建分组报告
    """
    report = []
    report.append("=== BMI分组报告 ===")
    report.append("")
    
    # 基本信息
    report.append("1. 基本信息")
    report.append(f"   - 分组数量: {evaluation['n_groups']}")
    report.append(f"   - 总数据点: {evaluation['total_points']}")
    report.append(f"   - 平均组大小: {evaluation['mean_group_size']:.1f}")
    report.append(f"   - 组大小范围: [{evaluation['min_group_size']}, {evaluation['max_group_size']}]")
    report.append("")
    
    # 分组详情
    report.append("2. 分组详情")
    for _, group in groups_df.iterrows():
        report.append(f"   组 {group['group_id']}: BMI [{group['bmi_min']:.1f}, {group['bmi_max']:.1f}], "
                     f"均值 {group['bmi_mean']:.1f}, 样本数 {group['n_points']}, "
                     f"最优时间 {group['optimal_time']:.2f}±{group['time_std']:.2f}")
    report.append("")
    
    # 质量指标
    report.append("3. 质量指标")
    report.append(f"   - 平均组内方差: {evaluation['mean_group_variance']:.4f}")
    report.append(f"   - 总方差: {evaluation['total_variance']:.4f}")
    if 'group_mean_std' in evaluation:
        report.append(f"   - 组间时间标准差: {evaluation['group_mean_std']:.2f}")
        report.append(f"   - 组间时间范围: {evaluation['group_mean_range']:.2f}")
    
    return "\n".join(report)
