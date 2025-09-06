#!/usr/bin/env python3
"""
BMI分组分析脚本

专门用于运行和比较不同的BMI分组方法。
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
from src.io_utils import load_data
from src.mu_sigma_models import create_mu_sigma_model
from src.mi_interval_imputer import IntervalImputer
from src.aft_models import fit_aft_models
from src.surv_predict import create_survival_predictor
from src.grouping import create_bmi_segmentation, compare_segmentation_methods
from src.plots import create_plot_manager
from src.utils import set_random_seeds, timer

logger = logging.getLogger(__name__)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="BMI分组分析")
    
    parser.add_argument("--config", "-c", default="configs/q3_default.yaml")
    parser.add_argument("--data-dir", "-d", default="data")
    parser.add_argument("--output-dir", "-o", default="outputs/grouping")
    parser.add_argument("--methods", nargs="+", default=["dp", "cart"])
    parser.add_argument("--K-values", nargs="+", type=int, default=[3, 4, 5])
    
    args = parser.parse_args()
    
    try:
        # 加载配置
        config = load_config(args.config)
        setup_logging(config)
        
        logger.info("开始BMI分组分析")
        
        # 设置随机种子
        set_random_seeds(config.seed)
        
        # 路径设置
        data_dir = Path(args.data_dir)
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 加载数据
        long_data_path = data_dir / "step1_long_records.csv"
        long_df = load_data(long_data_path)
        
        logger.info(f"加载数据: {long_df.shape}")
        
        with timer("完整分组分析", logger):
            # 快速管道（简化版本用于演示）
            
            # 1. μ/σ模型
            mu_sigma_model = create_mu_sigma_model("empirical")
            mu_sigma_model.fit(long_df)
            
            # 2. 简化的插补（只做一次）
            imputer = IntervalImputer(
                mu_sigma_model=mu_sigma_model,
                threshold=config.threshold,
                M=1,  # 只做一次插补以加速
                q=config.mi.q,
                deterministic_by_id=True,
                global_seed=config.seed,
                n_jobs=1
            )
            
            imputed_datasets = imputer.impute_intervals(long_df)
            dataset = imputed_datasets[0]
            
            # 3. 准备数据
            feature_cols = ["BMI_base"]
            X = dataset[feature_cols].values
            X = np.column_stack([np.ones(len(X)), X])
            BMI = dataset["BMI_base"].values
            
            L = dataset["L"].values
            R = dataset["R"].fillna(np.inf).values
            censor_type = dataset["censor_type"].values
            
            # 4. AFT模型
            aft_model = fit_aft_models(
                families=["lognormal"],  # 只用一个分布族
                X=X, L=L, R=R, censor_type=censor_type
            )
            
            # 5. 创建生存预测器
            survival_predictor = create_survival_predictor(aft_model)
            
            # 6. 比较分组方法
            comparison_df = compare_segmentation_methods(
                BMI=BMI,
                X=X,
                survival_predictor=survival_predictor,
                risk_config=config.dict(),
                methods=args.methods,
                K_values=args.K_values
            )
            
            # 保存比较结果
            comparison_df.to_csv(output_dir / "segmentation_comparison.csv", index=False)
            
            logger.info("分组方法比较结果:")
            print(comparison_df)
            
            # 选择最佳方法进行详细分析
            best_idx = comparison_df['average_risk'].idxmin()
            best_method = comparison_df.loc[best_idx, 'method']
            best_K = comparison_df.loc[best_idx, 'K']
            
            logger.info(f"最佳分组方法: {best_method}, K={best_K}")
            
            # 使用最佳方法进行详细分组
            best_segmentation = create_bmi_segmentation(
                method=best_method,
                K=best_K,
                min_group_size=config.grouping.min_group_size
            )
            
            best_segmentation.fit(BMI, X, survival_predictor, config.dict())
            
            # 可视化
            plot_manager = create_plot_manager(config.dict())
            segmentation_info = best_segmentation.get_segment_info()
            
            plot_manager.plot_bmi_segmentation(
                segmentation_info,
                BMI,
                title=f"最佳BMI分组结果 ({best_method}, K={best_K})",
                filename="best_bmi_segmentation"
            )
            
            # 生成报告
            report_lines = [
                "# BMI分组分析报告",
                "",
                f"分析方法: {args.methods}",
                f"K值范围: {args.K_values}",
                "",
                "## 方法比较结果",
                "",
                comparison_df.to_string(index=False),
                "",
                f"## 最佳方法: {best_method} (K={best_K})",
                f"平均风险: {comparison_df.loc[best_idx, 'average_risk']:.4f}",
                f"w*多样性: {comparison_df.loc[best_idx, 'w_diversity']:.3f}",
                "",
                "## 最佳分组详情",
                ""
            ]
            
            for segment in segmentation_info['segments']:
                if segment['optimal_w'] is not None:
                    report_lines.append(
                        f"- 段 {segment['segment_id']}: "
                        f"BMI [{segment['bmi_range'][0]:.1f}, {segment['bmi_range'][1]:.1f}], "
                        f"w*={segment['optimal_w']:.3f}"
                    )
            
            report_content = "\n".join(report_lines)
            
            with open(output_dir / "grouping_analysis_report.md", "w", encoding="utf-8") as f:
                f.write(report_content)
        
        logger.info("BMI分组分析完成")
        return 0
        
    except Exception as e:
        logger.error(f"分组分析失败: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())