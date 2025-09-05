"""
Problem 2 主程序
BMI 分组与最佳 NIPT 时点优化
"""

import os
import sys
import argparse
import logging
import numpy as np
from pathlib import Path

# 添加src目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# 导入模块
from io_utils import load_step1_products, load_step2_config, save_results, get_data_summary
from sigma_lookup import build_sigma_lookup
from models_long import fit_quantile_models
from grid_search import solve_w_star_curve_with_grid_search
from grouping import find_bmi_cuts, evaluate_grouping_quality, create_grouping_report

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('p2.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='Problem 2: BMI分组与最佳NIPT时点优化')
    parser.add_argument('--config', type=str, default='config/step2_config.yaml',
                       help='配置文件路径')
    parser.add_argument('--data-dir', type=str, default='data',
                       help='数据目录路径')
    parser.add_argument('--output-dir', type=str, default='outputs',
                       help='输出目录路径')
    parser.add_argument('--verbose', action='store_true',
                       help='详细输出')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        # 运行主流程
        results = run_analysis(args.config, args.data_dir, args.output_dir)
        
        # 保存结果
        save_results(results, args.output_dir, results['config'])
        
        logger.info("分析完成！")
        
    except Exception as e:
        logger.error(f"分析失败: {e}")
        raise


def run_analysis(config_path: str, data_dir: str, output_dir: str) -> dict:
    """
    运行完整分析流程
    
    Parameters:
    -----------
    config_path : str
        配置文件路径
    data_dir : str
        数据目录路径
    output_dir : str
        输出目录路径
        
    Returns:
    --------
    dict
        分析结果
    """
    logger.info("开始 Problem 2 分析...")
    
    # 1. 读取数据与配置
    logger.info("步骤 1: 读取数据与配置")
    long_df, surv_df, report, cfg1 = load_step1_products(data_dir)
    cfg2 = load_step2_config(config_path)
    
    # 合并配置
    config = {**cfg1, **cfg2}
    
    # 数据摘要
    data_summary = get_data_summary(long_df, surv_df)
    logger.info(f"数据摘要: {data_summary['n_individuals']} 个个体, {data_summary['n_records']} 条记录")
    
    # 2. 构建σ查表器
    logger.info("步骤 2: 构建σ查表器")
    sigma_lookup = build_sigma_lookup(
        report, 
        config.get('sigma_estimation', {}).get('local_sigma_path'),
        config.get('sigma_estimation', {}).get('shrinkage_lambda', 0.1),
        config.get('sigma_estimation', {}).get('interpolation_method', 'linear')
    )
    
    # 3. 纵向通道：拟合μ/分位模型
    logger.info("步骤 3: 纵向通道建模")
    model_params = config.get('model_params', {})
    
    # 使用GAM分位数回归模型
    use_gam = model_params.get('use_gam', True)
    tau = model_params.get('quantile_tau', 0.9)  # 使用90%分位数作为保守估计
    
    logger.info(f"使用GAM分位数回归: use_gam={use_gam}, tau={tau}")
    
    mu_model = fit_quantile_models(
        long_df,
        tau=tau,
        features=model_params.get('features', 'poly+interactions'),
        model=model_params.get('quantile_model', 'GAM'),
        use_gam=use_gam
    )
    
    # 4. 求解w*(b)曲线（使用网格搜索）
    logger.info("步骤 4: 使用网格搜索求解w*(b)曲线")
    
    # 网格搜索参数（基于实际数据范围）
    actual_week_min = long_df['week'].min()
    actual_week_max = long_df['week'].max()
    
    grid_params = {
        'thr_adj': 0.04,  # 浓度阈值4%
        'delta': 0.1,     # 失败风险阈值10%
        'w_min': actual_week_min,  # 使用实际最小孕周
        'w_max': actual_week_max,  # 使用实际最大孕周
        'w_step': 0.5,    # 孕周搜索步长
        'b_resolution': 40  # BMI网格分辨率
    }
    
    logger.info(f"使用实际数据范围: 孕周 {actual_week_min:.1f} - {actual_week_max:.1f} 周")
    
    # 使用网格搜索求解w*(b)曲线
    wstar_curve = solve_w_star_curve_with_grid_search(
        long_df, mu_model, sigma_lookup, 
        model_params.get('scenario_params', {}),
        **grid_params
    )
    
    # 5. BMI分组（基于实际数据分布）
    logger.info("步骤 5: BMI分组")
    grouping_params = config.get('grouping', {})
    
    # 使用实际数据的BMI四分位数作为切点
    bmi_quartiles = long_df['BMI_used'].quantile([0.25, 0.5, 0.75]).values
    custom_cuts = bmi_quartiles.tolist()
    
    logger.info(f"使用BMI四分位数作为切点: {custom_cuts}")
    
    cuts, groups = find_bmi_cuts(
        wstar_curve,
        who_cuts=grouping_params.get('who_cuts', [18.5, 25.0, 30.0]),
        method='custom',
        custom_cuts=custom_cuts,
        delta=grouping_params.get('delta', 2.0),
        min_group_n=grouping_params.get('min_group_n', 10),  # 降低最小组数要求
        min_cut_distance=grouping_params.get('min_cut_distance', 1.0),
        search=grouping_params.get('search', 'tree'),
        tree_params=grouping_params.get('tree_params', {}),
        dp_params=grouping_params.get('dp_params', {})
    )
    
    # 6. 评估分组质量
    logger.info("步骤 6: 评估分组质量")
    evaluation = evaluate_grouping_quality(groups, wstar_curve)
    
    # 7. 生成报告
    logger.info("步骤 7: 生成报告")
    report = create_grouping_report(groups, evaluation)
    
    # 8. 跳过可视化
    logger.info("步骤 8: 跳过可视化（已禁用）")
    main_plot = None
    
    # 9. 汇总结果
    results = {
        'config': config,
        'data_summary': data_summary,
        'wstar_curve': wstar_curve,
        'groups': groups,
        'cuts': cuts,
        'evaluation': evaluation,
        'report': report,
        'main_plot': main_plot,
        'models': {
            'mu_model': mu_model,
            'sigma_lookup': sigma_lookup
        }
    }
    
    logger.info("分析流程完成")
    
    return results


if __name__ == "__main__":
    main()
