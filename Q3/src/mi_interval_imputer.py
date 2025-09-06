"""
多重插补(MI)模块

实现从观测误差到区间删失数据的多重插补过程。
基于μ/σ先验模型和离散时间首次达标过程。
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from scipy.stats import norm
from joblib import Parallel, delayed
import pickle
from pathlib import Path
import logging

from .exceptions import ComputationError, ModelFittingError, ParallelizationError
from .mu_sigma_models import MuSigmaModel
from .utils import rng_for_id, timer, ProgressTracker, ensure_numpy
from .io_utils import save_data

logger = logging.getLogger(__name__)


class IntervalImputer:
    """区间删失多重插补器"""
    
    def __init__(self, 
                 mu_sigma_model: MuSigmaModel,
                 threshold: float = 0.04,
                 M: int = 20,
                 q: float = 0.02,
                 deterministic_by_id: bool = True,
                 global_seed: int = 42,
                 cache_dir: Optional[str] = None,
                 n_jobs: int = -1):
        """
        初始化多重插补器
        
        Args:
            mu_sigma_model: 拟合好的μ/σ模型
            threshold: 达标阈值
            M: 插补次数
            q: 观测误差率
            deterministic_by_id: 是否按ID确定性播种
            global_seed: 全局随机种子
            cache_dir: 缓存目录
            n_jobs: 并行作业数
        """
        self.mu_sigma_model = mu_sigma_model
        self.threshold = threshold
        self.M = M
        self.q = q
        self.deterministic_by_id = deterministic_by_id
        self.global_seed = global_seed
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.n_jobs = n_jobs
        
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"初始化多重插补器: M={M}, q={q}, threshold={threshold}")
    
    def impute_intervals(self, long_df: pd.DataFrame) -> List[pd.DataFrame]:
        """
        执行多重插补，生成M个插补数据集
        
        Args:
            long_df: 长格式数据，包含 id, t, BMI, Y_frac 和协变量
            
        Returns:
            M个插补后的区间删失数据集列表
        """
        if not self.mu_sigma_model.is_fitted:
            raise ModelFittingError("μ/σ模型尚未拟合")
        
        logger.info(f"开始多重插补: {len(long_df)} 行数据, {long_df['id'].nunique()} 个个体")
        
        with timer("多重插补总耗时", logger):
            # 预处理数据
            processed_data = self._preprocess_data(long_df)
            
            # 并行执行插补
            if self.n_jobs == 1:
                # 单线程执行
                imputed_datasets = []
                tracker = ProgressTracker(self.M, "插补进度")
                
                for m in range(self.M):
                    dataset = self._single_imputation(processed_data, m)
                    imputed_datasets.append(dataset)
                    tracker.update()
                    
                    # 保存中间结果
                    if self.cache_dir:
                        cache_file = self.cache_dir / f"imputed_dataset_{m}.pkl"
                        with open(cache_file, 'wb') as f:
                            pickle.dump(dataset, f)
            else:
                # 并行执行
                try:
                    imputed_datasets = Parallel(n_jobs=self.n_jobs, verbose=1)(
                        delayed(self._single_imputation)(processed_data, m) 
                        for m in range(self.M)
                    )
                except Exception as e:
                    raise ParallelizationError(f"并行插补失败: {e}", n_workers=self.n_jobs)
            
            logger.info(f"多重插补完成: 生成 {len(imputed_datasets)} 个数据集")
            return imputed_datasets
    
    def _preprocess_data(self, long_df: pd.DataFrame) -> Dict[str, Any]:
        """预处理数据，准备插补所需的信息"""
        logger.info("预处理长格式数据")
        
        # 按个体分组
        grouped_data = {}
        
        for patient_id, group in long_df.groupby('id'):
            # 排序确保时间顺序
            group = group.sort_values('t').reset_index(drop=True)
            
            # 提取特征
            t_values = group['t'].values
            BMI_values = group['BMI'].values
            Y_frac_values = group['Y_frac'].values
            
            # 协变量
            Z_cols = [col for col in group.columns if col.startswith('Z')]
            Z_values = group[Z_cols].fillna(0).values if Z_cols else None
            
            # 检测类型
            Assay_cols = [col for col in group.columns if col.startswith('Assay_')]
            if Assay_cols:
                Assay_values = group[Assay_cols].fillna(0).values
                Assay_type = np.argmax(Assay_values, axis=1)
            else:
                Assay_type = None
            
            grouped_data[patient_id] = {
                't': t_values,
                'BMI': BMI_values,
                'Y_frac': Y_frac_values,
                'Z': Z_values,
                'Assay': Assay_type,
                'baseline_BMI': BMI_values[0],  # 基线BMI
                'baseline_Z': Z_values[0] if Z_values is not None else None,
                'baseline_Assay': Assay_type[0] if Assay_type is not None else None
            }
        
        return {
            'grouped_data': grouped_data,
            'patient_ids': list(grouped_data.keys()),
            'n_patients': len(grouped_data)
        }
    
    def _single_imputation(self, processed_data: Dict[str, Any], m: int) -> pd.DataFrame:
        """执行单次插补"""
        logger.debug(f"执行第 {m+1} 次插补")
        
        results = []
        grouped_data = processed_data['grouped_data']
        
        for patient_id in processed_data['patient_ids']:
            patient_data = grouped_data[patient_id]
            
            # 为每个患者生成确定性随机数生成器
            if self.deterministic_by_id:
                rng = rng_for_id(self.global_seed + m, patient_id)
            else:
                rng = np.random.default_rng(self.global_seed + m + patient_id)
            
            # 计算达标概率序列
            p_succ = self._compute_success_probabilities(patient_data)
            
            # 生成首次达标时间区间
            L, R, censor_type = self._generate_interval(p_succ, patient_data['t'], rng)
            
            # 构造结果记录
            result_record = {
                'id': patient_id,
                'L': L,
                'R': R,
                'censor_type': censor_type,
                'BMI_base': patient_data['baseline_BMI']
            }
            
            # 添加基线协变量
            if patient_data['baseline_Z'] is not None:
                for i, z_val in enumerate(patient_data['baseline_Z']):
                    result_record[f'Z{i+1}'] = z_val
            
            # 添加基线检测类型
            if patient_data['baseline_Assay'] is not None:
                # 转换为one-hot编码
                for assay_idx in range(3):  # 假设有3种检测类型
                    result_record[f'Assay_{chr(65+assay_idx)}'] = float(
                        patient_data['baseline_Assay'] == assay_idx
                    )
            
            results.append(result_record)
        
        return pd.DataFrame(results)
    
    def _compute_success_probabilities(self, patient_data: Dict[str, Any]) -> np.ndarray:
        """
        计算每个时间点的达标概率
        
        基于公式：p_succ(t) ≈ (1-q) * Φ((μ(t) - threshold) / σ(t))
        """
        t_values = patient_data['t']
        BMI_values = patient_data['BMI']
        Z_values = patient_data['Z']
        Assay_values = patient_data['Assay']
        
        try:
            # 预测μ和σ
            mu_pred, sigma_pred = self.mu_sigma_model.predict(
                t_values, BMI_values, Z_values, Assay_values
            )
            
            # 计算标准化分数
            z_scores = (mu_pred - self.threshold) / sigma_pred
            
            # 计算达标概率
            p_succ = (1 - self.q) * norm.cdf(z_scores)
            
            # 确保概率在有效范围内
            p_succ = np.clip(p_succ, 1e-10, 1 - 1e-10)
            
            return p_succ
            
        except Exception as e:
            raise ComputationError(f"计算达标概率失败: {e}", patient_id=patient_data.get('id'))
    
    def _generate_interval(self, p_succ: np.ndarray, t_values: np.ndarray, 
                          rng: np.random.Generator) -> Tuple[float, Optional[float], str]:
        """
        基于离散时间首次达标过程生成区间删失数据
        
        Args:
            p_succ: 各时间点的达标概率
            t_values: 时间点数组
            rng: 随机数生成器
            
        Returns:
            (L, R, censor_type) 元组
        """
        n_times = len(p_succ)
        
        if n_times == 0:
            return 0.0, None, "right"
        
        # 计算累积首次达标概率：H_k = 1 - ∏(1 - p_j) for j=1 to k
        cum_prob = np.zeros(n_times)
        running_product = 1.0
        
        for k in range(n_times):
            running_product *= (1 - p_succ[k])
            cum_prob[k] = 1 - running_product
        
        # 生成随机数确定首次达标时间
        u = rng.random()
        
        # 找到首次超过u的时间点
        exceed_indices = np.where(cum_prob >= u)[0]
        
        if len(exceed_indices) == 0:
            # 在观测期内未达标 -> 右删失
            return t_values[-1], None, "right"
        
        first_exceed_idx = exceed_indices[0]
        
        if first_exceed_idx == 0:
            # 在第一个时间点就达标 -> 左删失
            return 0.0, t_values[0], "left"
        else:
            # 在区间内达标 -> 区间删失
            L = t_values[first_exceed_idx - 1]
            R = t_values[first_exceed_idx]
            return L, R, "interval"
    
    def combine_imputations(self, imputed_datasets: List[pd.DataFrame],
                           combination_method: str = "rubin") -> Dict[str, Any]:
        """
        合并多重插补结果
        
        Args:
            imputed_datasets: 插补数据集列表
            combination_method: 合并方法 ("rubin", "simple_average")
            
        Returns:
            合并结果字典
        """
        logger.info(f"使用 {combination_method} 方法合并 {len(imputed_datasets)} 个插补结果")
        
        if combination_method == "rubin":
            return self._rubin_combination(imputed_datasets)
        elif combination_method == "simple_average":
            return self._simple_average_combination(imputed_datasets)
        else:
            raise ValueError(f"不支持的合并方法: {combination_method}")
    
    def _rubin_combination(self, imputed_datasets: List[pd.DataFrame]) -> Dict[str, Any]:
        """
        Rubin规则合并插补结果
        
        计算：
        - β̄ = (1/M) Σ β̂^(m)
        - W̄ = (1/M) Σ Var^(m)  
        - B = (1/(M-1)) Σ (β̂^(m) - β̄)(β̂^(m) - β̄)ᵀ
        - T = W̄ + (1 + 1/M)B
        """
        M = len(imputed_datasets)
        
        # 这里简化处理，主要关注区间端点的统计
        L_values = []
        R_values = []
        
        for dataset in imputed_datasets:
            # 只考虑区间删失的情况
            interval_mask = dataset['censor_type'] == 'interval'
            if interval_mask.any():
                L_values.append(dataset.loc[interval_mask, 'L'].values)
                R_values.append(dataset.loc[interval_mask, 'R'].values)
        
        if not L_values:
            logger.warning("没有区间删失数据用于Rubin合并")
            return {"method": "rubin", "status": "no_interval_data"}
        
        # 计算统计量
        L_mean = np.mean([np.mean(L) for L in L_values])
        R_mean = np.mean([np.mean(R) for R in R_values])
        
        L_within_var = np.mean([np.var(L) for L in L_values])
        R_within_var = np.mean([np.var(R) for R in R_values])
        
        L_between_var = np.var([np.mean(L) for L in L_values])
        R_between_var = np.var([np.mean(R) for R in R_values])
        
        # 总方差
        L_total_var = L_within_var + (1 + 1/M) * L_between_var
        R_total_var = R_within_var + (1 + 1/M) * R_between_var
        
        return {
            "method": "rubin",
            "M": M,
            "L_mean": L_mean,
            "R_mean": R_mean,
            "L_se": np.sqrt(L_total_var),
            "R_se": np.sqrt(R_total_var),
            "L_within_var": L_within_var,
            "L_between_var": L_between_var,
            "R_within_var": R_within_var,
            "R_between_var": R_between_var
        }
    
    def _simple_average_combination(self, imputed_datasets: List[pd.DataFrame]) -> Dict[str, Any]:
        """简单平均合并"""
        # 计算每个数据集的统计摘要
        summaries = []
        
        for i, dataset in enumerate(imputed_datasets):
            summary = {
                "dataset_id": i,
                "n_interval": (dataset['censor_type'] == 'interval').sum(),
                "n_left": (dataset['censor_type'] == 'left').sum(),
                "n_right": (dataset['censor_type'] == 'right').sum(),
                "n_exact": (dataset['censor_type'] == 'exact').sum()
            }
            
            # 区间删失的统计
            interval_mask = dataset['censor_type'] == 'interval'
            if interval_mask.any():
                summary.update({
                    "L_mean": dataset.loc[interval_mask, 'L'].mean(),
                    "R_mean": dataset.loc[interval_mask, 'R'].mean(),
                    "interval_width_mean": (dataset.loc[interval_mask, 'R'] - 
                                          dataset.loc[interval_mask, 'L']).mean()
                })
            
            summaries.append(summary)
        
        # 整体统计
        overall_stats = {
            "method": "simple_average",
            "M": len(imputed_datasets),
            "summaries": summaries,
            "avg_n_interval": np.mean([s["n_interval"] for s in summaries]),
            "avg_n_left": np.mean([s["n_left"] for s in summaries]),
            "avg_n_right": np.mean([s["n_right"] for s in summaries]),
            "avg_n_exact": np.mean([s["n_exact"] for s in summaries])
        }
        
        # 如果有区间数据，计算平均值
        L_means = [s.get("L_mean") for s in summaries if "L_mean" in s]
        R_means = [s.get("R_mean") for s in summaries if "R_mean" in s]
        
        if L_means:
            overall_stats.update({
                "L_mean": np.mean(L_means),
                "R_mean": np.mean(R_means),
                "L_std": np.std(L_means),
                "R_std": np.std(R_means)
            })
        
        return overall_stats
    
    def save_imputed_datasets(self, imputed_datasets: List[pd.DataFrame],
                             output_dir: Union[str, Path],
                             format: str = "csv") -> None:
        """
        保存插补数据集
        
        Args:
            imputed_datasets: 插补数据集列表
            output_dir: 输出目录
            format: 保存格式 ("csv", "parquet", "pkl")
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"保存 {len(imputed_datasets)} 个插补数据集到 {output_dir}")
        
        for m, dataset in enumerate(imputed_datasets):
            if format == "csv":
                filename = f"imputed_dataset_{m:03d}.csv"
            elif format == "parquet":
                filename = f"imputed_dataset_{m:03d}.parquet"
            elif format == "pkl":
                filename = f"imputed_dataset_{m:03d}.pkl"
            else:
                raise ValueError(f"不支持的保存格式: {format}")
            
            filepath = output_dir / filename
            save_data(dataset, filepath)
        
        logger.info("插补数据集保存完成")
    
    def get_imputation_summary(self, imputed_datasets: List[pd.DataFrame]) -> Dict[str, Any]:
        """获取插补结果摘要"""
        M = len(imputed_datasets)
        
        # 收集统计信息
        censor_type_counts = {
            'interval': [],
            'left': [],
            'right': [],
            'exact': []
        }
        
        interval_widths = []
        
        for dataset in imputed_datasets:
            # 删失类型统计
            type_counts = dataset['censor_type'].value_counts()
            for ctype in censor_type_counts.keys():
                censor_type_counts[ctype].append(type_counts.get(ctype, 0))
            
            # 区间宽度
            interval_mask = dataset['censor_type'] == 'interval'
            if interval_mask.any():
                widths = dataset.loc[interval_mask, 'R'] - dataset.loc[interval_mask, 'L']
                interval_widths.extend(widths.tolist())
        
        summary = {
            "M": M,
            "total_patients": len(imputed_datasets[0]) if imputed_datasets else 0,
            "censor_type_stats": {
                ctype: {
                    "mean": np.mean(counts),
                    "std": np.std(counts),
                    "min": np.min(counts),
                    "max": np.max(counts)
                }
                for ctype, counts in censor_type_counts.items()
            }
        }
        
        if interval_widths:
            summary["interval_width_stats"] = {
                "mean": np.mean(interval_widths),
                "std": np.std(interval_widths),
                "median": np.median(interval_widths),
                "q25": np.percentile(interval_widths, 25),
                "q75": np.percentile(interval_widths, 75)
            }
        
        return summary