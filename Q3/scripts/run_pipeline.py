#!/usr/bin/env python3
"""
MCM Q3 主运行脚本

执行完整的分析管道：
1. 数据加载和验证
2. μ/σ先验模型拟合
3. 多重插补(MI)
4. AFT模型拟合
5. 风险优化和分组
6. 结果可视化和报告生成
"""

import sys
import argparse
from pathlib import Path
import logging
import traceback
from typing import Dict, Any

# 添加源码路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import load_config, setup_logging
from src.io_utils import load_data, save_data, create_data_summary
from src.exceptions import MCMError, handle_exception
from src.mu_sigma_models import create_mu_sigma_model, evaluate_mu_sigma_model
from src.mi_interval_imputer import IntervalImputer
from src.aft_models import fit_aft_models
from src.surv_predict import create_survival_predictor
from src.risk_objective import create_risk_objective, optimize_multiple_groups
from src.grouping import create_bmi_segmentation, evaluate_segmentation
from src.plots import create_plot_manager
from src.utils import set_random_seeds, timer, memory_usage

logger = logging.getLogger(__name__)


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="MCM Q3 分析管道")
    
    parser.add_argument(
        "--config", "-c",
        type=str,
        default="configs/q3_default.yaml",
        help="配置文件路径"
    )
    
    parser.add_argument(
        "--data-dir", "-d",
        type=str,
        default="data",
        help="数据目录路径"
    )
    
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default="outputs",
        help="输出目录路径"
    )
    
    parser.add_argument(
        "--skip-mi",
        action="store_true",
        help="跳过多重插补步骤（使用现有数据）"
    )
    
    parser.add_argument(
        "--skip-plots",
        action="store_true",
        help="跳过绘图步骤"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="详细输出"
    )
    
    parser.add_argument(
        "--profile",
        action="store_true",
        help="启用性能分析"
    )
    
    return parser.parse_args()


def load_and_validate_data(data_dir: Path, config: Dict[str, Any]) -> Dict[str, Any]:
    """加载和验证数据"""
    logger.info("=== 步骤1: 数据加载和验证 ===")
    
    data_results = {}
    
    with timer("数据加载", logger):
        # 加载长格式数据
        long_data_path = data_dir / "step1_long_records.csv"
        long_schema_path = data_dir / "schemas" / "long_records_schema.yaml"
        
        if long_data_path.exists():
            long_df = load_data(long_data_path, long_schema_path, validate=True)
            data_results["long_df"] = long_df
            logger.info(f"长格式数据: {long_df.shape}")
        else:
            raise MCMError(f"长格式数据文件不存在: {long_data_path}")
        
        # 加载生存数据（如果存在）
        surv_data_path = data_dir / "step1_surv_dat_fit.csv"
        surv_schema_path = data_dir / "schemas" / "surv_dat_schema.yaml"
        
        if surv_data_path.exists():
            surv_df = load_data(surv_data_path, surv_schema_path, validate=True)
            data_results["surv_df"] = surv_df
            logger.info(f"生存数据: {surv_df.shape}")
        else:
            logger.warning("生存数据文件不存在，将通过MI生成")
    
    # 创建数据摘要
    data_results["data_summary"] = {}
    for key, df in data_results.items():
        if key.endswith("_df"):
            data_results["data_summary"][key] = create_data_summary(df, key)
    
    return data_results


def fit_mu_sigma_model(long_df, config: Dict[str, Any]) -> Dict[str, Any]:
    """拟合μ/σ先验模型"""
    logger.info("=== 步骤2: μ/σ先验模型拟合 ===")
    
    with timer("μ/σ模型拟合", logger):
        # 创建模型（默认使用经验模型）
        model_type = config.get("mu_sigma_model", {}).get("type", "empirical")
        model_params = config.get("mu_sigma_model", {}).get("params", {})
        
        mu_sigma_model = create_mu_sigma_model(model_type, **model_params)
        
        # 拟合模型
        mu_sigma_model.fit(long_df)
        
        # 评估模型（使用训练数据，实际应该用验证集）
        evaluation = evaluate_mu_sigma_model(mu_sigma_model, long_df)
        
        logger.info(f"μ/σ模型评估: MSE={evaluation['mu_mse']:.4f}, R²={evaluation['mu_r2']:.4f}")
    
    return {
        "mu_sigma_model": mu_sigma_model,
        "evaluation": evaluation
    }


def perform_multiple_imputation(long_df, mu_sigma_model, config: Dict[str, Any], 
                               output_dir: Path) -> Dict[str, Any]:
    """执行多重插补"""
    logger.info("=== 步骤3: 多重插补(MI) ===")
    
    mi_config = config["mi"]
    
    with timer("多重插补", logger):
        # 创建插补器
        imputer = IntervalImputer(
            mu_sigma_model=mu_sigma_model,
            threshold=config["threshold"],
            M=mi_config["M"],
            q=mi_config["q"],
            deterministic_by_id=mi_config["deterministic_by_id"],
            global_seed=config["seed"],
            cache_dir=str(output_dir / "cache"),
            n_jobs=config["performance"]["max_workers"]
        )
        
        # 执行插补
        imputed_datasets = imputer.impute_intervals(long_df)
        
        # 保存插补数据集
        imputed_dir = output_dir / "imputed_datasets"
        imputer.save_imputed_datasets(imputed_datasets, imputed_dir, format="csv")
        
        # 获取插补摘要
        imputation_summary = imputer.get_imputation_summary(imputed_datasets)
        
        # 合并插补结果
        combination_results = imputer.combine_imputations(imputed_datasets, "rubin")
        
        logger.info(f"多重插补完成: M={len(imputed_datasets)}, "
                   f"平均区间删失数={imputation_summary['censor_type_stats']['interval']['mean']:.1f}")
    
    return {
        "imputer": imputer,
        "imputed_datasets": imputed_datasets,
        "imputation_summary": imputation_summary,
        "combination_results": combination_results
    }


def fit_aft_models_step(imputed_datasets, config: Dict[str, Any]) -> Dict[str, Any]:
    """拟合AFT模型"""
    logger.info("=== 步骤4: AFT模型拟合 ===")
    
    aft_config = config["aft"]
    
    with timer("AFT模型拟合", logger):
        # 使用第一个插补数据集进行演示（实际应该对所有数据集拟合然后合并）
        dataset = imputed_datasets[0]
        
        # 准备特征矩阵
        feature_cols = ["BMI_base"]
        Z_cols = [col for col in dataset.columns if col.startswith("Z")]
        Assay_cols = [col for col in dataset.columns if col.startswith("Assay_")]
        
        feature_cols.extend(Z_cols)
        feature_cols.extend(Assay_cols)
        
        # 添加截距项
        X = dataset[feature_cols].fillna(0).values
        X = np.column_stack([np.ones(len(X)), X])  # 添加截距
        
        # 提取删失数据
        L = dataset["L"].values
        R = dataset["R"].fillna(np.inf).values  # 右删失用无穷大表示
        censor_type = dataset["censor_type"].values
        
        # 拟合模型
        aft_model = fit_aft_models(
            families=aft_config["families"],
            X=X,
            L=L,
            R=R,
            censor_type=censor_type,
            ensemble_config=aft_config.get("ensemble", {}),
            n_jobs=1  # 简化并行
        )
        
        logger.info(f"AFT模型拟合完成: {type(aft_model).__name__}")
        
        # 如果是集成模型，显示权重信息
        if hasattr(aft_model, 'get_model_weights'):
            weights = aft_model.get_model_weights()
            logger.info(f"集成权重: {weights}")
    
    return {
        "aft_model": aft_model,
        "feature_columns": feature_cols,
        "X": X,
        "L": L,
        "R": R,
        "censor_type": censor_type
    }


def optimize_risk_and_grouping(aft_model, X, config: Dict[str, Any], 
                              output_dir: Path) -> Dict[str, Any]:
    """风险优化和分组"""
    logger.info("=== 步骤5: 风险优化和分组 ===")
    
    with timer("风险优化和分组", logger):
        # 创建生存预测器
        survival_predictor = create_survival_predictor(aft_model)
        
        # 创建风险目标函数
        risk_objective = create_risk_objective(survival_predictor, config)
        
        # 整体优化
        overall_result = risk_objective.find_optimal_w(X, optimization_method="grid_search")
        logger.info(f"整体最优w*: {overall_result['optimal_w']:.3f}")
        
        # BMI分组
        grouping_config = config["grouping"]
        
        # 提取BMI数据（假设是第二列，第一列是截距）
        BMI = X[:, 1]  # 假设BMI是第一个真实特征
        
        # 创建分段器
        segmentation = create_bmi_segmentation(
            method=grouping_config["method"],
            K=grouping_config["K"],
            min_group_size=grouping_config["min_group_size"]
        )
        
        # 拟合分段
        segmentation.fit(BMI, X, survival_predictor, config)
        
        # 评估分段效果
        segmentation_eval = evaluate_segmentation(
            segmentation, BMI, X, survival_predictor, config
        )
        
        # 计算风险曲线
        risk_curve_df = risk_objective.compute_risk_curve(X)
        
        logger.info(f"分组完成: {segmentation_eval['n_groups']} 个组, "
                   f"平均风险: {segmentation_eval['average_risk']:.4f}")
    
    return {
        "survival_predictor": survival_predictor,
        "risk_objective": risk_objective,
        "overall_result": overall_result,
        "segmentation": segmentation,
        "segmentation_eval": segmentation_eval,
        "risk_curve_df": risk_curve_df,
        "BMI": BMI
    }


def generate_visualizations(results: Dict[str, Any], config: Dict[str, Any],
                          output_dir: Path) -> None:
    """生成可视化"""
    logger.info("=== 步骤6: 生成可视化 ===")
    
    with timer("可视化生成", logger):
        # 创建绘图管理器
        plot_manager = create_plot_manager(config)
        
        # 1. 生存曲线
        if "survival_predictor" in results:
            survival_predictor = results["survival_predictor"]
            X = results["X"]
            
            # 预测生存曲线（选择部分样本）
            sample_indices = np.random.choice(len(X), min(10, len(X)), replace=False)
            X_sample = X[sample_indices]
            
            t_grid, F_curves = survival_predictor.predict_survival_curves(X_sample)
            plot_manager.plot_survival_curves(
                t_grid, F_curves, 
                labels=[f"样本 {i+1}" for i in range(len(X_sample))],
                filename="survival_curves_sample"
            )
        
        # 2. 风险曲线
        if "risk_curve_df" in results:
            plot_manager.plot_risk_curves(
                results["risk_curve_df"],
                filename="risk_curves"
            )
        
        # 3. BMI分段结果
        if "segmentation" in results:
            segmentation_info = results["segmentation"].get_segment_info()
            BMI_data = results.get("BMI")
            
            plot_manager.plot_bmi_segmentation(
                segmentation_info, 
                BMI_data,
                filename="bmi_segmentation"
            )
        
        # 4. 分组生存曲线
        if all(key in results for key in ["survival_predictor", "segmentation", "X", "BMI"]):
            survival_predictor = results["survival_predictor"]
            segmentation = results["segmentation"]
            X = results["X"]
            BMI = results["BMI"]
            
            groups = segmentation.predict_groups(BMI)
            group_curves = survival_predictor.predict_group_curves(X, groups)
            
            plot_manager.plot_group_survival_curves(
                group_curves,
                filename="group_survival_curves"
            )
        
        logger.info("可视化生成完成")


def generate_report(results: Dict[str, Any], config: Dict[str, Any],
                   output_dir: Path) -> None:
    """生成分析报告"""
    logger.info("=== 步骤7: 生成分析报告 ===")
    
    report_lines = []
    report_lines.append("# MCM Q3 分析报告")
    report_lines.append("")
    report_lines.append(f"配置文件: {config.get('config_file', 'N/A')}")
    report_lines.append(f"随机种子: {config['seed']}")
    report_lines.append(f"达标阈值: {config['threshold']}")
    report_lines.append(f"概率门槛: {config['tau']}")
    report_lines.append("")
    
    # 数据摘要
    if "data_summary" in results:
        report_lines.append("## 数据摘要")
        for key, summary in results["data_summary"].items():
            report_lines.append(f"### {key}")
            report_lines.append(f"- 样本数: {summary['shape'][0]}")
            report_lines.append(f"- 特征数: {summary['shape'][1]}")
            report_lines.append(f"- 内存使用: {summary['memory_usage_mb']:.2f} MB")
            report_lines.append("")
    
    # μ/σ模型结果
    if "mu_sigma_evaluation" in results:
        eval_result = results["mu_sigma_evaluation"]
        report_lines.append("## μ/σ先验模型")
        report_lines.append(f"- MSE: {eval_result['mu_mse']:.4f}")
        report_lines.append(f"- MAE: {eval_result['mu_mae']:.4f}")
        report_lines.append(f"- R²: {eval_result['mu_r2']:.4f}")
        report_lines.append("")
    
    # 多重插补结果
    if "imputation_summary" in results:
        summary = results["imputation_summary"]
        report_lines.append("## 多重插补结果")
        report_lines.append(f"- 插补次数 M: {summary['M']}")
        report_lines.append(f"- 总患者数: {summary['total_patients']}")
        
        for ctype, stats in summary["censor_type_stats"].items():
            report_lines.append(f"- {ctype}删失: 平均 {stats['mean']:.1f} 例")
        report_lines.append("")
    
    # AFT模型结果
    if "aft_model" in results:
        aft_model = results["aft_model"]
        if hasattr(aft_model, 'get_ensemble_summary'):
            summary = aft_model.get_ensemble_summary()
            report_lines.append("## AFT模型")
            report_lines.append(f"- 集成方法: {summary['ensemble_method']}")
            report_lines.append(f"- 模型数量: {summary['n_models']}")
            
            if 'model_criteria' in summary:
                report_lines.append("- 模型比较:")
                for family, criteria in summary['model_criteria'].items():
                    report_lines.append(f"  - {family}: AIC={criteria['AIC']:.2f}, BIC={criteria['BIC']:.2f}")
            report_lines.append("")
    
    # 风险优化结果
    if "overall_result" in results:
        result = results["overall_result"]
        report_lines.append("## 风险优化结果")
        report_lines.append(f"- 整体最优w*: {result['optimal_w']:.3f} 周")
        report_lines.append(f"- 最小风险: {result['optimal_risk']:.4f}")
        
        if "best_result" in result:
            best = result["best_result"]
            report_lines.append(f"- 达标概率: {best['mean_success_prob']:.3f}")
            report_lines.append(f"- 满足门槛: {'是' if best['meets_threshold'] else '否'}")
        report_lines.append("")
    
    # 分组结果
    if "segmentation_eval" in results:
        eval_result = results["segmentation_eval"]
        report_lines.append("## BMI分组结果")
        report_lines.append(f"- 分组方法: {eval_result['method']}")
        report_lines.append(f"- 分组数量: {eval_result['n_groups']}")
        report_lines.append(f"- 平均风险: {eval_result['average_risk']:.4f}")
        report_lines.append(f"- w*多样性: {eval_result['w_diversity']:.3f}")
        report_lines.append(f"- 满足门槛的组数: {eval_result['n_groups_meeting_threshold']}")
        
        if "group_results" in eval_result:
            report_lines.append("- 各组详情:")
            for group_id, info in eval_result["group_results"].items():
                report_lines.append(f"  - 组 {group_id}: w*={info['optimal_w']:.3f}, "
                                  f"样本数={info['n_samples']}, "
                                  f"达标率={info['success_probability']:.3f}")
        report_lines.append("")
    
    # 保存报告
    report_content = "\n".join(report_lines)
    
    # Markdown格式
    with open(output_dir / "reports" / "q3_report.md", "w", encoding="utf-8") as f:
        f.write(report_content)
    
    # 纯文本格式
    with open(output_dir / "reports" / "q3_report.txt", "w", encoding="utf-8") as f:
        f.write(report_content)
    
    logger.info("分析报告生成完成")


def main():
    """主函数"""
    args = parse_arguments()
    
    try:
        # 加载配置
        config_path = Path(args.config)
        config = load_config(config_path)
        config["config_file"] = str(config_path)
        
        # 设置日志
        if args.verbose:
            config.logging.level = "DEBUG"
        setup_logging(config)
        
        logger.info("=" * 60)
        logger.info("MCM Q3 分析管道启动")
        logger.info("=" * 60)
        
        # 设置随机种子
        set_random_seeds(config.seed)
        
        # 设置路径
        data_dir = Path(args.data_dir)
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 创建输出子目录
        for subdir in ["tables", "figures", "models", "reports", "cache"]:
            (output_dir / subdir).mkdir(parents=True, exist_ok=True)
        
        # 初始化结果字典
        results = {}
        
        # 步骤1: 数据加载和验证
        data_results = load_and_validate_data(data_dir, config)
        results.update(data_results)
        
        # 步骤2: μ/σ先验模型
        mu_sigma_results = fit_mu_sigma_model(data_results["long_df"], config)
        results.update(mu_sigma_results)
        results["mu_sigma_evaluation"] = mu_sigma_results["evaluation"]
        
        # 步骤3: 多重插补
        if not args.skip_mi:
            mi_results = perform_multiple_imputation(
                data_results["long_df"], 
                mu_sigma_results["mu_sigma_model"], 
                config, 
                output_dir
            )
            results.update(mi_results)
        else:
            logger.info("跳过多重插补步骤")
            # 这里应该加载现有的插补数据
        
        # 步骤4: AFT模型拟合
        if "imputed_datasets" in results:
            aft_results = fit_aft_models_step(results["imputed_datasets"], config)
            results.update(aft_results)
        
        # 步骤5: 风险优化和分组
        if "aft_model" in results:
            optimization_results = optimize_risk_and_grouping(
                results["aft_model"], 
                results["X"], 
                config, 
                output_dir
            )
            results.update(optimization_results)
        
        # 步骤6: 可视化
        if not args.skip_plots:
            generate_visualizations(results, config, output_dir)
        
        # 步骤7: 生成报告
        generate_report(results, config, output_dir)
        
        # 内存使用情况
        memory_info = memory_usage()
        logger.info(f"当前内存使用: {memory_info['rss_mb']:.1f} MB")
        
        logger.info("=" * 60)
        logger.info("MCM Q3 分析管道完成")
        logger.info("=" * 60)
        
        return 0
        
    except Exception as e:
        logger.error(f"管道执行失败: {e}")
        logger.error(traceback.format_exc())
        return 1


if __name__ == "__main__":
    import numpy as np  # 添加缺失的导入
    sys.exit(main())