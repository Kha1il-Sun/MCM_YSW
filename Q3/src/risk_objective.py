"""
风险函数和w*优化模块

实现风险最小化目标函数和最优推荐时点w*的求解。
包含未达标惩罚、延迟惩罚和不确定性惩罚的综合风险函数。
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from scipy import optimize
from scipy.stats import norm
import warnings
import logging

from .exceptions import OptimizationError, ComputationError
from .surv_predict import SurvivalPredictor
from .utils import ensure_numpy, timer, create_grid, clip_to_bounds

logger = logging.getLogger(__name__)


class RiskObjective:
    """风险目标函数类"""
    
    def __init__(self, 
                 survival_predictor: SurvivalPredictor,
                 weights: Dict[str, float],
                 delay_config: Dict[str, float],
                 tau: float = 0.90,
                 clinical_bounds: Tuple[float, float] = (12.0, 25.0)):
        """
        初始化风险目标函数
        
        Args:
            survival_predictor: 生存预测器
            weights: 权重配置 {w1, w2, w3}
            delay_config: 延迟惩罚配置 {pref_week, scale}
            tau: 概率门槛
            clinical_bounds: 临床可行范围 (min_w, max_w)
        """
        self.survival_predictor = survival_predictor
        self.w1 = weights['w1']  # 未达标惩罚权重
        self.w2 = weights['w2']  # 延迟惩罚权重
        self.w3 = weights['w3']  # 不确定性惩罚权重
        
        self.pref_week = delay_config['pref_week']  # 偏好时间点
        self.delay_scale = delay_config['scale']    # 延迟惩罚尺度
        
        self.tau = tau
        self.clinical_bounds = clinical_bounds
        
        logger.info(f"初始化风险目标函数: w1={self.w1}, w2={self.w2}, w3={self.w3}")
        logger.info(f"延迟参数: pref_week={self.pref_week}, scale={self.delay_scale}")
        logger.info(f"概率门槛: τ={self.tau}, 临床范围: {self.clinical_bounds}")
    
    def compute_risk(self, w: float, X: np.ndarray, 
                    group_weights: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        计算在推荐时点w的总风险
        
        R(w) = w1 * [1 - F̄(w)] + w2 * Delay(w) + w3 * Unc(w)
        
        Args:
            w: 推荐时间点
            X: 协变量矩阵
            group_weights: 样本权重（用于加权平均）
            
        Returns:
            风险组件字典
        """
        if not (self.clinical_bounds[0] <= w <= self.clinical_bounds[1]):
            return {
                "total_risk": np.inf,
                "failure_risk": np.inf,
                "delay_penalty": np.inf,
                "uncertainty_penalty": np.inf,
                "mean_success_prob": 0.0,
                "meets_threshold": False
            }
        
        # 预测在w时刻的达标概率
        success_probs = self.survival_predictor.predict_success_probability(X, w)
        
        # 加权平均（如果提供权重）
        if group_weights is not None:
            group_weights = ensure_numpy(group_weights)
            if len(group_weights) != len(success_probs):
                raise ValueError("group_weights长度与样本数不匹配")
            group_weights = group_weights / np.sum(group_weights)  # 归一化
            mean_success_prob = np.sum(success_probs * group_weights)
        else:
            mean_success_prob = np.mean(success_probs)
        
        # 1. 未达标风险: w1 * [1 - F̄(w)]
        failure_risk = self.w1 * (1 - mean_success_prob)
        
        # 2. 延迟惩罚: w2 * Delay(w)
        delay_penalty = self.w2 * self._compute_delay_penalty(w)
        
        # 3. 不确定性惩罚: w3 * Unc(w)
        uncertainty_penalty = self.w3 * self._compute_uncertainty_penalty(
            success_probs, group_weights
        )
        
        # 总风险
        total_risk = failure_risk + delay_penalty + uncertainty_penalty
        
        # 检查是否满足门槛约束
        meets_threshold = mean_success_prob >= self.tau
        
        return {
            "total_risk": total_risk,
            "failure_risk": failure_risk,
            "delay_penalty": delay_penalty,
            "uncertainty_penalty": uncertainty_penalty,
            "mean_success_prob": mean_success_prob,
            "meets_threshold": meets_threshold
        }
    
    def _compute_delay_penalty(self, w: float) -> float:
        """
        计算延迟惩罚
        
        Delay(w) = |w - pref_week| / scale
        """
        return abs(w - self.pref_week) / self.delay_scale
    
    def _compute_uncertainty_penalty(self, success_probs: np.ndarray,
                                   group_weights: Optional[np.ndarray] = None) -> float:
        """
        计算不确定性惩罚
        
        Unc(w) = Var[F(w|x)] 或其他不确定性度量
        """
        if group_weights is not None:
            # 加权方差
            mean_prob = np.sum(success_probs * group_weights)
            weighted_var = np.sum(group_weights * (success_probs - mean_prob)**2)
            return weighted_var
        else:
            # 简单方差
            return np.var(success_probs)
    
    def find_optimal_w(self, X: np.ndarray,
                      group_weights: Optional[np.ndarray] = None,
                      w_grid: Optional[np.ndarray] = None,
                      optimization_method: str = "grid_search") -> Dict[str, Any]:
        """
        寻找最优推荐时点 w*
        
        Args:
            X: 协变量矩阵
            group_weights: 样本权重
            w_grid: 搜索网格
            optimization_method: 优化方法 ("grid_search", "minimize", "constrained")
            
        Returns:
            优化结果字典
        """
        logger.info(f"寻找最优w*: {optimization_method} 方法")
        
        with timer("w*优化", logger):
            if optimization_method == "grid_search":
                return self._grid_search_optimization(X, group_weights, w_grid)
            elif optimization_method == "minimize":
                return self._scipy_optimization(X, group_weights, constrained=False)
            elif optimization_method == "constrained":
                return self._scipy_optimization(X, group_weights, constrained=True)
            else:
                raise ValueError(f"不支持的优化方法: {optimization_method}")
    
    def _grid_search_optimization(self, X: np.ndarray,
                                group_weights: Optional[np.ndarray],
                                w_grid: Optional[np.ndarray]) -> Dict[str, Any]:
        """网格搜索优化"""
        if w_grid is None:
            w_grid = create_grid(self.clinical_bounds, n_points=100, log_scale=False)
        
        logger.debug(f"网格搜索: {len(w_grid)} 个候选点")
        
        best_w = None
        best_risk = np.inf
        best_result = None
        all_results = []
        
        feasible_candidates = []
        
        for w in w_grid:
            try:
                result = self.compute_risk(w, X, group_weights)
                all_results.append({"w": w, **result})
                
                # 检查可行性（满足门槛约束）
                if result["meets_threshold"]:
                    feasible_candidates.append((w, result["total_risk"], result))
                    
                    if result["total_risk"] < best_risk:
                        best_risk = result["total_risk"]
                        best_w = w
                        best_result = result
                        
            except Exception as e:
                logger.warning(f"计算w={w}时出错: {e}")
                continue
        
        if best_w is None:
            logger.warning("未找到满足门槛约束的可行解")
            # 选择门槛约束违反最小的解
            min_violation_idx = np.argmax([r["mean_success_prob"] for r in all_results])
            best_result = all_results[min_violation_idx]
            best_w = best_result["w"]
        
        return {
            "optimal_w": best_w,
            "optimal_risk": best_risk,
            "optimization_method": "grid_search",
            "n_evaluations": len(all_results),
            "n_feasible": len(feasible_candidates),
            "best_result": best_result,
            "all_results": all_results,
            "w_grid": w_grid
        }
    
    def _scipy_optimization(self, X: np.ndarray,
                          group_weights: Optional[np.ndarray],
                          constrained: bool) -> Dict[str, Any]:
        """SciPy优化"""
        def objective(w_array):
            w = w_array[0]  # 标量优化
            try:
                result = self.compute_risk(w, X, group_weights)
                return result["total_risk"]
            except Exception:
                return np.inf
        
        # 边界约束
        bounds = [self.clinical_bounds]
        
        # 门槛约束
        constraints = []
        if constrained:
            def threshold_constraint(w_array):
                w = w_array[0]
                try:
                    result = self.compute_risk(w, X, group_weights)
                    return result["mean_success_prob"] - self.tau  # >= 0
                except Exception:
                    return -np.inf
            
            constraints.append({
                'type': 'ineq',
                'fun': threshold_constraint
            })
        
        # 初始点
        initial_w = (self.clinical_bounds[0] + self.clinical_bounds[1]) / 2
        
        try:
            if constrained and constraints:
                result = optimize.minimize(
                    objective,
                    [initial_w],
                    method='SLSQP',
                    bounds=bounds,
                    constraints=constraints,
                    options={'maxiter': 1000, 'ftol': 1e-9}
                )
            else:
                result = optimize.minimize_scalar(
                    lambda w: objective([w]),
                    bounds=self.clinical_bounds,
                    method='bounded',
                    options={'maxfun': 1000, 'xatol': 1e-6}
                )
                # 转换为统一格式
                result = type('Result', (), {
                    'success': result.success,
                    'x': [result.x],
                    'fun': result.fun,
                    'nfev': result.nfev,
                    'message': getattr(result, 'message', 'Optimization completed')
                })()
            
            if result.success:
                optimal_w = result.x[0]
                optimal_result = self.compute_risk(optimal_w, X, group_weights)
                
                return {
                    "optimal_w": optimal_w,
                    "optimal_risk": result.fun,
                    "optimization_method": "scipy_constrained" if constrained else "scipy_minimize",
                    "success": True,
                    "n_evaluations": result.nfev,
                    "message": result.message,
                    "best_result": optimal_result
                }
            else:
                raise OptimizationError(f"SciPy优化失败: {result.message}")
                
        except Exception as e:
            logger.error(f"SciPy优化异常: {e}")
            # 回退到网格搜索
            logger.info("回退到网格搜索方法")
            return self._grid_search_optimization(X, group_weights, None)
    
    def compute_risk_curve(self, X: np.ndarray,
                          group_weights: Optional[np.ndarray] = None,
                          w_grid: Optional[np.ndarray] = None) -> pd.DataFrame:
        """
        计算风险曲线 R(w)
        
        Args:
            X: 协变量矩阵
            group_weights: 样本权重
            w_grid: 时间网格
            
        Returns:
            包含w和各风险组件的DataFrame
        """
        if w_grid is None:
            w_grid = create_grid(self.clinical_bounds, n_points=200, log_scale=False)
        
        logger.info(f"计算风险曲线: {len(w_grid)} 个时间点")
        
        results = []
        
        with timer("风险曲线计算", logger):
            for w in w_grid:
                try:
                    risk_result = self.compute_risk(w, X, group_weights)
                    results.append({"w": w, **risk_result})
                except Exception as e:
                    logger.warning(f"计算w={w}的风险时出错: {e}")
                    continue
        
        return pd.DataFrame(results)
    
    def sensitivity_analysis(self, X: np.ndarray,
                           parameter_ranges: Dict[str, List[float]],
                           base_w: Optional[float] = None) -> Dict[str, pd.DataFrame]:
        """
        敏感性分析
        
        Args:
            X: 协变量矩阵
            parameter_ranges: 参数范围字典
            base_w: 基准推荐时点，如果为None则先优化求得
            
        Returns:
            敏感性分析结果字典
        """
        logger.info("执行风险函数敏感性分析")
        
        if base_w is None:
            # 先找到基准最优解
            opt_result = self.find_optimal_w(X)
            base_w = opt_result["optimal_w"]
            logger.info(f"基准最优w*: {base_w:.3f}")
        
        sensitivity_results = {}
        
        for param_name, param_values in parameter_ranges.items():
            logger.debug(f"分析参数 {param_name} 的敏感性")
            
            param_results = []
            original_value = getattr(self, param_name, None)
            
            for param_value in param_values:
                # 临时修改参数
                setattr(self, param_name, param_value)
                
                try:
                    # 在基准w点计算风险
                    risk_result = self.compute_risk(base_w, X)
                    
                    # 重新优化找最优w
                    opt_result = self.find_optimal_w(X, optimization_method="grid_search")
                    
                    param_results.append({
                        param_name: param_value,
                        "base_w_risk": risk_result["total_risk"],
                        "base_w_success_prob": risk_result["mean_success_prob"],
                        "optimal_w": opt_result["optimal_w"],
                        "optimal_risk": opt_result["optimal_risk"],
                        "w_shift": opt_result["optimal_w"] - base_w
                    })
                    
                except Exception as e:
                    logger.warning(f"参数{param_name}={param_value}时分析失败: {e}")
                    continue
            
            # 恢复原始参数值
            if original_value is not None:
                setattr(self, param_name, original_value)
            
            sensitivity_results[param_name] = pd.DataFrame(param_results)
        
        return sensitivity_results
    
    def get_risk_decomposition(self, w: float, X: np.ndarray,
                             group_weights: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        获取风险分解详情
        
        Args:
            w: 推荐时间点
            X: 协变量矩阵
            group_weights: 样本权重
            
        Returns:
            详细的风险分解信息
        """
        risk_result = self.compute_risk(w, X, group_weights)
        
        # 计算各组件的相对贡献
        total_risk = risk_result["total_risk"]
        if total_risk > 0:
            failure_pct = 100 * risk_result["failure_risk"] / total_risk
            delay_pct = 100 * risk_result["delay_penalty"] / total_risk
            uncertainty_pct = 100 * risk_result["uncertainty_penalty"] / total_risk
        else:
            failure_pct = delay_pct = uncertainty_pct = 0.0
        
        # 预测详情
        success_probs = self.survival_predictor.predict_success_probability(X, w)
        
        decomposition = {
            **risk_result,
            "risk_components_pct": {
                "failure_risk": failure_pct,
                "delay_penalty": delay_pct,
                "uncertainty_penalty": uncertainty_pct
            },
            "success_prob_stats": {
                "mean": np.mean(success_probs),
                "std": np.std(success_probs),
                "min": np.min(success_probs),
                "max": np.max(success_probs),
                "q25": np.percentile(success_probs, 25),
                "q75": np.percentile(success_probs, 75)
            },
            "delay_from_preference": w - self.pref_week,
            "w": w,
            "n_samples": len(X)
        }
        
        return decomposition


def create_risk_objective(survival_predictor: SurvivalPredictor,
                         config: Dict[str, Any]) -> RiskObjective:
    """
    创建风险目标函数的工厂函数
    
    Args:
        survival_predictor: 生存预测器
        config: 配置字典
        
    Returns:
        风险目标函数实例
    """
    return RiskObjective(
        survival_predictor=survival_predictor,
        weights=config.get('weights', {'w1': 1.0, 'w2': 0.6, 'w3': 0.3}),
        delay_config=config.get('delay', {'pref_week': 15.0, 'scale': 10.0}),
        tau=config.get('tau', 0.90),
        clinical_bounds=tuple(config.get('clinical_bounds', [12.0, 25.0]))
    )


def optimize_multiple_groups(survival_predictor: SurvivalPredictor,
                           X_groups: Dict[Any, np.ndarray],
                           config: Dict[str, Any],
                           optimization_method: str = "grid_search") -> Dict[Any, Dict[str, Any]]:
    """
    为多个组分别优化w*
    
    Args:
        survival_predictor: 生存预测器
        X_groups: 分组协变量字典
        config: 配置字典
        optimization_method: 优化方法
        
    Returns:
        各组优化结果字典
    """
    logger.info(f"为 {len(X_groups)} 个组优化w*")
    
    risk_objective = create_risk_objective(survival_predictor, config)
    group_results = {}
    
    for group_id, X_group in X_groups.items():
        logger.info(f"优化组 {group_id}: {len(X_group)} 个样本")
        
        try:
            result = risk_objective.find_optimal_w(
                X_group, 
                optimization_method=optimization_method
            )
            
            # 添加组信息
            result["group_id"] = group_id
            result["n_samples"] = len(X_group)
            
            group_results[group_id] = result
            
            logger.info(f"组 {group_id} 最优w*: {result['optimal_w']:.3f}")
            
        except Exception as e:
            logger.error(f"组 {group_id} 优化失败: {e}")
            group_results[group_id] = {
                "group_id": group_id,
                "n_samples": len(X_group),
                "success": False,
                "error": str(e)
            }
    
    return group_results


def compare_risk_strategies(survival_predictor: SurvivalPredictor,
                          X: np.ndarray,
                          strategies: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    """
    比较不同风险策略
    
    Args:
        survival_predictor: 生存预测器
        X: 协变量矩阵
        strategies: 策略配置字典
        
    Returns:
        策略比较结果DataFrame
    """
    logger.info(f"比较 {len(strategies)} 个风险策略")
    
    comparison_results = []
    
    for strategy_name, strategy_config in strategies.items():
        logger.debug(f"评估策略: {strategy_name}")
        
        try:
            risk_objective = create_risk_objective(survival_predictor, strategy_config)
            opt_result = risk_objective.find_optimal_w(X)
            
            # 获取详细分解
            decomposition = risk_objective.get_risk_decomposition(
                opt_result["optimal_w"], X
            )
            
            comparison_results.append({
                "strategy": strategy_name,
                "optimal_w": opt_result["optimal_w"],
                "total_risk": opt_result["optimal_risk"],
                "success_probability": decomposition["mean_success_prob"],
                "meets_threshold": decomposition["meets_threshold"],
                "failure_risk": decomposition["failure_risk"],
                "delay_penalty": decomposition["delay_penalty"],
                "uncertainty_penalty": decomposition["uncertainty_penalty"],
                **{f"weight_{k}": v for k, v in strategy_config.get('weights', {}).items()}
            })
            
        except Exception as e:
            logger.error(f"策略 {strategy_name} 评估失败: {e}")
            continue
    
    return pd.DataFrame(comparison_results)