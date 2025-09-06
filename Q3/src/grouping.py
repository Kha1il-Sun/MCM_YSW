"""
BMI分段优化模块

实现基于动态规划(DP)和CART的BMI分段方法，
用于将患者按BMI分组并为每组优化推荐时点。
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from joblib import Parallel, delayed
import logging

from .exceptions import OptimizationError, ComputationError
from .risk_objective import RiskObjective, create_risk_objective
from .surv_predict import SurvivalPredictor
from .utils import ensure_numpy, timer

logger = logging.getLogger(__name__)


class BMISegmentation:
    """BMI分段基类"""
    
    def __init__(self, method: str):
        self.method = method
        self.is_fitted = False
        self.breakpoints_ = None
        self.group_labels_ = None
        self.segment_costs_ = None
        self.optimal_w_by_group_ = None
    
    def fit(self, BMI: np.ndarray, X: np.ndarray, 
            survival_predictor: SurvivalPredictor,
            risk_config: Dict[str, Any]) -> None:
        """拟合分段模型"""
        raise NotImplementedError
    
    def predict_groups(self, BMI: np.ndarray) -> np.ndarray:
        """预测BMI分组"""
        if not self.is_fitted:
            raise OptimizationError("分段模型尚未拟合")
        
        BMI = ensure_numpy(BMI)
        groups = np.zeros(len(BMI), dtype=int)
        
        for i, bmi_val in enumerate(BMI):
            # 找到BMI所属的区间
            group = 0
            for j, bp in enumerate(self.breakpoints_[1:]):  # 跳过第一个断点（最小值）
                if bmi_val <= bp:
                    group = j
                    break
                group = j + 1
            groups[i] = group
        
        return groups
    
    def get_segment_info(self) -> Dict[str, Any]:
        """获取分段信息"""
        if not self.is_fitted:
            return {"is_fitted": False}
        
        segments = []
        for i in range(len(self.breakpoints_) - 1):
            segments.append({
                "segment_id": i,
                "bmi_range": (self.breakpoints_[i], self.breakpoints_[i+1]),
                "optimal_w": self.optimal_w_by_group_.get(i, None),
                "segment_cost": self.segment_costs_.get(i, None) if self.segment_costs_ else None
            })
        
        return {
            "method": self.method,
            "is_fitted": True,
            "n_segments": len(self.breakpoints_) - 1,
            "breakpoints": self.breakpoints_,
            "segments": segments
        }


class DynamicProgrammingSegmentation(BMISegmentation):
    """基于动态规划的BMI分段"""
    
    def __init__(self, K: int = 4, min_group_size: int = 40):
        """
        初始化DP分段
        
        Args:
            K: 分段数量
            min_group_size: 最小组大小
        """
        super().__init__("dp")
        self.K = K
        self.min_group_size = min_group_size
        self.segment_cost_matrix_ = None
    
    def fit(self, BMI: np.ndarray, X: np.ndarray, 
            survival_predictor: SurvivalPredictor,
            risk_config: Dict[str, Any]) -> None:
        """
        拟合DP分段模型
        
        Args:
            BMI: BMI数组
            X: 协变量矩阵
            survival_predictor: 生存预测器
            risk_config: 风险配置
        """
        logger.info(f"拟合DP分段模型: K={self.K}, 最小组大小={self.min_group_size}")
        
        BMI = ensure_numpy(BMI)
        X = ensure_numpy(X)
        
        if len(BMI) != len(X):
            raise ValueError("BMI和X长度不匹配")
        
        with timer("DP分段拟合", logger):
            # 按BMI排序
            sort_indices = np.argsort(BMI)
            BMI_sorted = BMI[sort_indices]
            X_sorted = X[sort_indices]
            
            # 获取候选断点
            candidate_points = self._get_candidate_breakpoints(BMI_sorted)
            
            # 计算段成本矩阵
            self.segment_cost_matrix_ = self._compute_segment_costs(
                BMI_sorted, X_sorted, candidate_points, 
                survival_predictor, risk_config
            )
            
            # 动态规划求解
            breakpoints = self._dp_segmentation(candidate_points)
            
            # 存储结果
            self.breakpoints_ = breakpoints
            self.is_fitted = True
            
            # 为每个分段优化w*
            self._optimize_segment_recommendations(
                BMI_sorted, X_sorted, survival_predictor, risk_config
            )
            
            logger.info(f"DP分段完成: 断点 {self.breakpoints_}")
    
    def _get_candidate_breakpoints(self, BMI_sorted: np.ndarray) -> np.ndarray:
        """获取候选断点"""
        n = len(BMI_sorted)
        
        # 确保最小组大小
        min_idx = self.min_group_size
        max_idx = n - self.min_group_size
        
        if max_idx <= min_idx:
            logger.warning("数据量太小，无法满足最小组大小约束")
            return np.array([BMI_sorted[0], BMI_sorted[-1]])
        
        # 候选断点：在满足最小组大小的范围内等间距选择
        candidate_indices = np.linspace(min_idx, max_idx, 
                                       min(50, max_idx - min_idx + 1), 
                                       dtype=int)
        
        candidate_points = BMI_sorted[candidate_indices]
        
        # 添加边界点
        candidate_points = np.unique(np.concatenate([
            [BMI_sorted[0]], candidate_points, [BMI_sorted[-1]]
        ]))
        
        return candidate_points
    
    def _compute_segment_costs(self, BMI_sorted: np.ndarray, X_sorted: np.ndarray,
                             candidate_points: np.ndarray,
                             survival_predictor: SurvivalPredictor,
                             risk_config: Dict[str, Any]) -> Dict[Tuple[int, int], float]:
        """
        计算段成本矩阵 C([bs, be])
        
        返回字典，键为(start_idx, end_idx)，值为该段的最小风险
        """
        logger.info("计算段成本矩阵")
        
        risk_objective = create_risk_objective(survival_predictor, risk_config)
        segment_costs = {}
        
        n_candidates = len(candidate_points)
        
        for i in range(n_candidates - 1):
            for j in range(i + 1, n_candidates):
                start_bmi = candidate_points[i]
                end_bmi = candidate_points[j]
                
                # 找到对应的样本索引
                start_idx = np.searchsorted(BMI_sorted, start_bmi, side='left')
                end_idx = np.searchsorted(BMI_sorted, end_bmi, side='right')
                
                # 检查最小组大小
                if end_idx - start_idx < self.min_group_size:
                    segment_costs[(i, j)] = np.inf
                    continue
                
                # 提取该段的数据
                X_segment = X_sorted[start_idx:end_idx]
                
                try:
                    # 为该段优化w*
                    opt_result = risk_objective.find_optimal_w(
                        X_segment, 
                        optimization_method="grid_search"
                    )
                    
                    segment_costs[(i, j)] = opt_result["optimal_risk"]
                    
                except Exception as e:
                    logger.warning(f"计算段({i},{j})成本失败: {e}")
                    segment_costs[(i, j)] = np.inf
        
        logger.info(f"计算了 {len(segment_costs)} 个段成本")
        return segment_costs
    
    def _dp_segmentation(self, candidate_points: np.ndarray) -> np.ndarray:
        """
        动态规划求解最优分段
        
        DP[k][e] = min_s<e {DP[k-1][s] + C([s+1, e])}
        """
        logger.info("执行动态规划分段")
        
        n_candidates = len(candidate_points)
        
        # 初始化DP表
        # DP[k][e] 表示用k段分割到第e个候选点的最小成本
        DP = np.full((self.K + 1, n_candidates), np.inf)
        parent = np.full((self.K + 1, n_candidates), -1, dtype=int)
        
        # 边界条件：0段的成本为0（只在起始点）
        DP[0][0] = 0
        
        # 填充DP表
        for k in range(1, self.K + 1):
            for e in range(k, n_candidates):  # 至少需要k个点来分k段
                for s in range(k - 1, e):
                    # 检查段成本是否存在
                    if (s, e) in self.segment_cost_matrix_:
                        cost = DP[k-1][s] + self.segment_cost_matrix_[(s, e)]
                        
                        if cost < DP[k][e]:
                            DP[k][e] = cost
                            parent[k][e] = s
        
        # 回溯找到最优分割点
        if DP[self.K][-1] == np.inf:
            raise OptimizationError("无法找到满足约束的分段方案")
        
        # 回溯路径
        breakpoint_indices = []
        k, e = self.K, n_candidates - 1
        
        while k > 0 and parent[k][e] != -1:
            breakpoint_indices.append(e)
            e = parent[k][e]
            k -= 1
        breakpoint_indices.append(0)  # 起始点
        
        breakpoint_indices.reverse()
        breakpoints = candidate_points[breakpoint_indices]
        
        # 存储段成本
        self.segment_costs_ = {}
        for i in range(len(breakpoint_indices) - 1):
            start_idx = breakpoint_indices[i]
            end_idx = breakpoint_indices[i + 1]
            if (start_idx, end_idx) in self.segment_cost_matrix_:
                self.segment_costs_[i] = self.segment_cost_matrix_[(start_idx, end_idx)]
        
        return breakpoints
    
    def _optimize_segment_recommendations(self, BMI_sorted: np.ndarray, 
                                        X_sorted: np.ndarray,
                                        survival_predictor: SurvivalPredictor,
                                        risk_config: Dict[str, Any]) -> None:
        """为每个分段优化推荐时点"""
        logger.info("为各分段优化推荐时点")
        
        risk_objective = create_risk_objective(survival_predictor, risk_config)
        self.optimal_w_by_group_ = {}
        
        for i in range(len(self.breakpoints_) - 1):
            start_bmi = self.breakpoints_[i]
            end_bmi = self.breakpoints_[i + 1]
            
            # 找到对应的样本
            start_idx = np.searchsorted(BMI_sorted, start_bmi, side='left')
            end_idx = np.searchsorted(BMI_sorted, end_bmi, side='right')
            
            X_segment = X_sorted[start_idx:end_idx]
            
            try:
                opt_result = risk_objective.find_optimal_w(
                    X_segment,
                    optimization_method="grid_search"
                )
                
                self.optimal_w_by_group_[i] = opt_result["optimal_w"]
                
                logger.debug(f"分段 {i} BMI[{start_bmi:.1f}, {end_bmi:.1f}]: "
                           f"w*={opt_result['optimal_w']:.3f}")
                
            except Exception as e:
                logger.warning(f"分段 {i} 优化失败: {e}")
                self.optimal_w_by_group_[i] = None


class CARTSegmentation(BMISegmentation):
    """基于CART的BMI分段"""
    
    def __init__(self, K: int = 4, min_group_size: int = 40):
        """
        初始化CART分段
        
        Args:
            K: 最大分段数量（叶子节点数）
            min_group_size: 最小组大小
        """
        super().__init__("cart")
        self.K = K
        self.min_group_size = min_group_size
        self.tree_model_ = None
    
    def fit(self, BMI: np.ndarray, X: np.ndarray, 
            survival_predictor: SurvivalPredictor,
            risk_config: Dict[str, Any]) -> None:
        """
        拟合CART分段模型
        
        Args:
            BMI: BMI数组
            X: 协变量矩阵
            survival_predictor: 生存预测器
            risk_config: 风险配置
        """
        logger.info(f"拟合CART分段模型: 最大叶子数={self.K}")
        
        BMI = ensure_numpy(BMI)
        X = ensure_numpy(X)
        
        with timer("CART分段拟合", logger):
            # 计算每个样本的最优w*（作为目标变量）
            target_w = self._compute_individual_optimal_w(
                BMI, X, survival_predictor, risk_config
            )
            
            # 构建CART模型
            self.tree_model_ = DecisionTreeRegressor(
                max_leaf_nodes=self.K,
                min_samples_leaf=self.min_group_size,
                random_state=42
            )
            
            # 只使用BMI作为特征
            BMI_feature = BMI.reshape(-1, 1)
            self.tree_model_.fit(BMI_feature, target_w)
            
            # 提取分割点
            self.breakpoints_ = self._extract_breakpoints_from_tree(BMI)
            self.is_fitted = True
            
            # 为每个分段重新优化w*
            self._optimize_segment_recommendations(
                BMI, X, survival_predictor, risk_config
            )
            
            logger.info(f"CART分段完成: 断点 {self.breakpoints_}")
    
    def _compute_individual_optimal_w(self, BMI: np.ndarray, X: np.ndarray,
                                    survival_predictor: SurvivalPredictor,
                                    risk_config: Dict[str, Any]) -> np.ndarray:
        """计算每个个体的最优w*"""
        logger.info("计算个体最优w*")
        
        risk_objective = create_risk_objective(survival_predictor, risk_config)
        individual_w = np.zeros(len(BMI))
        
        # 可以并行计算，但为简化这里使用串行
        for i in range(len(BMI)):
            X_i = X[i:i+1]  # 保持二维
            
            try:
                opt_result = risk_objective.find_optimal_w(
                    X_i,
                    optimization_method="grid_search"
                )
                individual_w[i] = opt_result["optimal_w"]
                
            except Exception as e:
                logger.warning(f"个体 {i} 优化失败: {e}")
                # 使用默认值
                individual_w[i] = (risk_config.get('clinical_bounds', [12, 25])[0] + 
                                 risk_config.get('clinical_bounds', [12, 25])[1]) / 2
        
        return individual_w
    
    def _extract_breakpoints_from_tree(self, BMI: np.ndarray) -> np.ndarray:
        """从决策树中提取分割点"""
        tree = self.tree_model_.tree_
        
        # 收集所有BMI分割阈值
        thresholds = []
        
        def extract_thresholds(node_id):
            if tree.children_left[node_id] != tree.children_right[node_id]:
                # 内部节点
                threshold = tree.threshold[node_id]
                thresholds.append(threshold)
                
                # 递归处理子节点
                extract_thresholds(tree.children_left[node_id])
                extract_thresholds(tree.children_right[node_id])
        
        extract_thresholds(0)  # 从根节点开始
        
        # 排序并添加边界
        thresholds = sorted(set(thresholds))
        breakpoints = [np.min(BMI)] + thresholds + [np.max(BMI)]
        
        return np.array(sorted(set(breakpoints)))
    
    def _optimize_segment_recommendations(self, BMI: np.ndarray, X: np.ndarray,
                                        survival_predictor: SurvivalPredictor,
                                        risk_config: Dict[str, Any]) -> None:
        """为每个分段优化推荐时点"""
        logger.info("为各CART分段优化推荐时点")
        
        risk_objective = create_risk_objective(survival_predictor, risk_config)
        self.optimal_w_by_group_ = {}
        
        for i in range(len(self.breakpoints_) - 1):
            start_bmi = self.breakpoints_[i]
            end_bmi = self.breakpoints_[i + 1]
            
            # 找到该分段的样本
            mask = (BMI >= start_bmi) & (BMI <= end_bmi)
            if i < len(self.breakpoints_) - 2:  # 不是最后一段
                mask = (BMI >= start_bmi) & (BMI < end_bmi)
            
            X_segment = X[mask]
            
            if len(X_segment) < self.min_group_size:
                logger.warning(f"分段 {i} 样本数不足: {len(X_segment)}")
                self.optimal_w_by_group_[i] = None
                continue
            
            try:
                opt_result = risk_objective.find_optimal_w(
                    X_segment,
                    optimization_method="grid_search"
                )
                
                self.optimal_w_by_group_[i] = opt_result["optimal_w"]
                
                logger.debug(f"CART分段 {i} BMI[{start_bmi:.1f}, {end_bmi:.1f}]: "
                           f"w*={opt_result['optimal_w']:.3f}, n={len(X_segment)}")
                
            except Exception as e:
                logger.warning(f"CART分段 {i} 优化失败: {e}")
                self.optimal_w_by_group_[i] = None


def create_bmi_segmentation(method: str = "dp", **kwargs) -> BMISegmentation:
    """
    创建BMI分段器的工厂函数
    
    Args:
        method: 分段方法 ("dp", "cart")
        **kwargs: 方法参数
        
    Returns:
        BMI分段器实例
    """
    if method == "dp":
        return DynamicProgrammingSegmentation(**kwargs)
    elif method == "cart":
        return CARTSegmentation(**kwargs)
    else:
        raise ValueError(f"不支持的分段方法: {method}")


def evaluate_segmentation(segmentation: BMISegmentation,
                         BMI_test: np.ndarray, 
                         X_test: np.ndarray,
                         survival_predictor: SurvivalPredictor,
                         risk_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    评估分段效果
    
    Args:
        segmentation: 训练好的分段器
        BMI_test: 测试BMI数据
        X_test: 测试协变量数据
        survival_predictor: 生存预测器
        risk_config: 风险配置
        
    Returns:
        评估结果字典
    """
    logger.info("评估BMI分段效果")
    
    if not segmentation.is_fitted:
        raise OptimizationError("分段器尚未拟合")
    
    # 预测分组
    groups = segmentation.predict_groups(BMI_test)
    
    # 计算各组的风险
    risk_objective = create_risk_objective(survival_predictor, risk_config)
    group_results = {}
    
    unique_groups = np.unique(groups)
    total_risk = 0.0
    total_samples = 0
    
    for group_id in unique_groups:
        group_mask = groups == group_id
        X_group = X_test[group_mask]
        BMI_group = BMI_test[group_mask]
        
        if len(X_group) == 0:
            continue
        
        # 使用该组的最优w*
        optimal_w = segmentation.optimal_w_by_group_.get(group_id)
        
        if optimal_w is None:
            logger.warning(f"组 {group_id} 没有最优w*")
            continue
        
        # 计算组风险
        risk_result = risk_objective.compute_risk(optimal_w, X_group)
        
        group_results[group_id] = {
            "n_samples": len(X_group),
            "bmi_range": (np.min(BMI_group), np.max(BMI_group)),
            "optimal_w": optimal_w,
            "total_risk": risk_result["total_risk"],
            "success_probability": risk_result["mean_success_prob"],
            "meets_threshold": risk_result["meets_threshold"]
        }
        
        # 累积总风险（加权）
        total_risk += risk_result["total_risk"] * len(X_group)
        total_samples += len(X_group)
    
    # 计算整体指标
    avg_risk = total_risk / total_samples if total_samples > 0 else np.inf
    
    # 计算组间差异
    w_values = [info["optimal_w"] for info in group_results.values() 
                if info["optimal_w"] is not None]
    w_diversity = np.std(w_values) if len(w_values) > 1 else 0.0
    
    # 满足门槛的组数
    n_groups_meeting_threshold = sum(
        1 for info in group_results.values() 
        if info["meets_threshold"]
    )
    
    evaluation_result = {
        "method": segmentation.method,
        "n_groups": len(unique_groups),
        "total_samples": total_samples,
        "average_risk": avg_risk,
        "w_diversity": w_diversity,
        "n_groups_meeting_threshold": n_groups_meeting_threshold,
        "group_results": group_results,
        "segmentation_info": segmentation.get_segment_info()
    }
    
    return evaluation_result


def compare_segmentation_methods(BMI: np.ndarray, X: np.ndarray,
                               survival_predictor: SurvivalPredictor,
                               risk_config: Dict[str, Any],
                               methods: List[str] = ["dp", "cart"],
                               K_values: List[int] = [3, 4, 5]) -> pd.DataFrame:
    """
    比较不同分段方法
    
    Args:
        BMI: BMI数组
        X: 协变量矩阵
        survival_predictor: 生存预测器
        risk_config: 风险配置
        methods: 方法列表
        K_values: K值列表
        
    Returns:
        比较结果DataFrame
    """
    logger.info(f"比较分段方法: {methods}, K值: {K_values}")
    
    comparison_results = []
    
    for method in methods:
        for K in K_values:
            logger.info(f"评估方法: {method}, K={K}")
            
            try:
                # 创建分段器
                segmentation = create_bmi_segmentation(
                    method=method, 
                    K=K, 
                    min_group_size=max(20, len(BMI) // (K * 3))
                )
                
                # 拟合
                segmentation.fit(BMI, X, survival_predictor, risk_config)
                
                # 评估（使用相同数据，实际应该用测试集）
                eval_result = evaluate_segmentation(
                    segmentation, BMI, X, survival_predictor, risk_config
                )
                
                comparison_results.append({
                    "method": method,
                    "K": K,
                    "average_risk": eval_result["average_risk"],
                    "w_diversity": eval_result["w_diversity"],
                    "n_groups_meeting_threshold": eval_result["n_groups_meeting_threshold"],
                    "total_samples": eval_result["total_samples"]
                })
                
            except Exception as e:
                logger.error(f"方法 {method} K={K} 失败: {e}")
                comparison_results.append({
                    "method": method,
                    "K": K,
                    "average_risk": np.inf,
                    "w_diversity": 0.0,
                    "n_groups_meeting_threshold": 0,
                    "total_samples": 0,
                    "error": str(e)
                })
    
    return pd.DataFrame(comparison_results)