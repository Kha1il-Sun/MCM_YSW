#!/usr/bin/env python3
"""
Sigma敏感性分析脚本

分析σ参数变化对最优推荐时点w*的影响。
"""

import sys
import argparse
from pathlib import Path
import logging
import numpy as np
import pandas as pd

# 添加源码路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import load_config, setup_logging
from src.io_utils import load_data, save_data
from src.exceptions import MCMError
from src.mu_sigma_models import create_mu_sigma_model
from src.mi_interval_imputer import IntervalImputer
from src.aft_models import fit_aft_models
from src.surv_predict import create_survival_predictor
from src.risk_objective import create_risk_objective
from src.plots import create_plot_manager
from src.utils import set_random_seeds, timer

logger = logging.getLogger(__name__)


def run_sigma_sensitivity_analysis(config_path: str, data_dir: str, output_dir: str):
    """运行σ敏感性分析"""
    
    # 加载配置
    config = load_config(config_path)
    setup_logging(config)
    
    logger.info("开始σ敏感性分析")
    
    # 设置随机种子
    set_random_seeds(config.seed)
    
    # 路径设置
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 加载数据
    long_data_path = data_dir / "step1_long_records.csv"
    long_df = load_data(long_data_path)
    
    logger.info(f"加载数据: {long_df.shape}")
    
    # 获取σ扫描参数
    sigma_factors = config.sigma_sweep.factors
    base_sigma = config.sigma_sweep.get("base_sigma", 1.0)
    
    logger.info(f"σ因子: {sigma_factors}")
    
    # 存储结果
    sensitivity_results = []
    
    for factor in sigma_factors:
        logger.info(f"分析σ因子: {factor}")
        
        with timer(f"σ因子 {factor}", logger):
            try:
                # 调整σ参数
                adjusted_sigma = base_sigma * factor
                
                # 创建μ/σ模型（调整σ参数）
                mu_sigma_model = create_mu_sigma_model(
                    "empirical",
                    sigma_factors={"base": adjusted_sigma}
                )
                mu_sigma_model.fit(long_df)
                
                # 多重插补（减少M以加速）
                imputer = IntervalImputer(
                    mu_sigma_model=mu_sigma_model,
                    threshold=config.threshold,
                    M=config.mi.M,
                    q=config.mi.q,
                    deterministic_by_id=True,
                    global_seed=config.seed,
                    n_jobs=1
                )
                
                imputed_datasets = imputer.impute_intervals(long_df)
                
                # 使用第一个数据集拟合AFT模型
                dataset = imputed_datasets[0]
                
                # 准备特征矩阵
                feature_cols = ["BMI_base"]
                Z_cols = [col for col in dataset.columns if col.startswith("Z")]
                feature_cols.extend(Z_cols)
                
                X = dataset[feature_cols].fillna(0).values
                X = np.column_stack([np.ones(len(X)), X])
                
                L = dataset["L"].values
                R = dataset["R"].fillna(np.inf).values
                censor_type = dataset["censor_type"].values
                
                # 拟合AFT模型（只用log-normal以加速）
                aft_model = fit_aft_models(
                    families=["lognormal"],
                    X=X, L=L, R=R, censor_type=censor_type
                )
                
                # 创建生存预测器和风险目标函数
                survival_predictor = create_survival_predictor(aft_model)
                risk_objective = create_risk_objective(survival_predictor, config)
                
                # 优化w*
                opt_result = risk_objective.find_optimal_w(X, optimization_method="grid_search")
                
                # 记录结果
                sensitivity_results.append({
                    "sigma_factor": factor,
                    "adjusted_sigma": adjusted_sigma,
                    "optimal_w": opt_result["optimal_w"],
                    "optimal_risk": opt_result["optimal_risk"],
                    "success_probability": opt_result["best_result"]["mean_success_prob"],
                    "meets_threshold": opt_result["best_result"]["meets_threshold"],
                    "n_evaluations": opt_result["n_evaluations"]
                })
                
                logger.info(f"σ因子 {factor}: w*={opt_result['optimal_w']:.3f}")
                
            except Exception as e:
                logger.error(f"σ因子 {factor} 分析失败: {e}")
                sensitivity_results.append({
                    "sigma_factor": factor,
                    "adjusted_sigma": adjusted_sigma,
                    "optimal_w": np.nan,
                    "optimal_risk": np.inf,
                    "success_probability": np.nan,
                    "meets_threshold": False,
                    "error": str(e)
                })
    
    # 转换为DataFrame
    results_df = pd.DataFrame(sensitivity_results)
    
    # 保存结果
    results_df.to_csv(output_dir / "sigma_sensitivity_results.csv", index=False)
    
    # 分析结果
    logger.info("σ敏感性分析结果:")
    logger.info(f"w*范围: [{results_df['optimal_w'].min():.3f}, {results_df['optimal_w'].max():.3f}]")
    logger.info(f"w*标准差: {results_df['optimal_w'].std():.3f}")
    
    # 计算最大位移
    baseline_w = results_df[results_df['sigma_factor'] == 1.0]['optimal_w'].iloc[0]
    max_shift = results_df['optimal_w'].sub(baseline_w).abs().max()
    logger.info(f"最大w*位移: {max_shift:.3f} 周")
    
    # 生成可视化
    plot_manager = create_plot_manager(config)
    
    # σ敏感性图
    import matplotlib.pyplot as plt
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # 上图：w*随σ因子变化
    valid_mask = ~results_df['optimal_w'].isna()
    ax1.plot(results_df.loc[valid_mask, 'sigma_factor'], 
            results_df.loc[valid_mask, 'optimal_w'], 
            'o-', linewidth=2, markersize=8)
    ax1.axhline(y=baseline_w, color='red', linestyle='--', alpha=0.7, label=f'基准 (σ=1.0)')
    ax1.set_xlabel("σ倍数因子")
    ax1.set_ylabel("最优推荐时点 w* (周)")
    ax1.set_title("σ敏感性分析 - w*变化")
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # 下图：风险随σ因子变化
    ax2.plot(results_df.loc[valid_mask, 'sigma_factor'], 
            results_df.loc[valid_mask, 'optimal_risk'], 
            's-', linewidth=2, markersize=8, color='orange')
    ax2.set_xlabel("σ倍数因子")
    ax2.set_ylabel("最优风险值")
    ax2.set_title("σ敏感性分析 - 风险变化")
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_manager.save_figure(fig, "sigma_sensitivity_analysis")
    
    # 生成报告
    report_lines = [
        "# σ敏感性分析报告",
        "",
        f"基准σ值: {base_sigma}",
        f"σ因子范围: {min(sigma_factors)} - {max(sigma_factors)}",
        f"基准w* (σ=1.0): {baseline_w:.3f} 周",
        f"w*变化范围: [{results_df['optimal_w'].min():.3f}, {results_df['optimal_w'].max():.3f}] 周",
        f"最大w*位移: {max_shift:.3f} 周",
        f"w*标准差: {results_df['optimal_w'].std():.3f} 周",
        "",
        "## 详细结果",
        ""
    ]
    
    for _, row in results_df.iterrows():
        if not pd.isna(row['optimal_w']):
            shift = row['optimal_w'] - baseline_w
            report_lines.append(
                f"- σ因子 {row['sigma_factor']}: w*={row['optimal_w']:.3f} "
                f"(位移: {shift:+.3f}), 风险={row['optimal_risk']:.4f}"
            )
    
    report_content = "\n".join(report_lines)
    
    with open(output_dir / "sigma_sensitivity_report.md", "w", encoding="utf-8") as f:
        f.write(report_content)
    
    logger.info("σ敏感性分析完成")
    
    return results_df


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="σ敏感性分析")
    
    parser.add_argument("--config", "-c", default="configs/mi_sigma_sweep.yaml", 
                       help="配置文件路径")
    parser.add_argument("--data-dir", "-d", default="data", 
                       help="数据目录")
    parser.add_argument("--output-dir", "-o", default="outputs/sigma_sweep", 
                       help="输出目录")
    
    args = parser.parse_args()
    
    try:
        results_df = run_sigma_sensitivity_analysis(
            args.config, args.data_dir, args.output_dir
        )
        
        print("σ敏感性分析完成！")
        print(f"结果保存在: {args.output_dir}")
        
        return 0
        
    except Exception as e:
        print(f"分析失败: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())