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
    wstar_curve_EC = solve_w_star_curve_with_grid_search(
        long_df, mu_model, sigma_lookup, 
        model_params.get('scenario_params', {}),
        **grid_params
    )
    
    # 为了保持兼容性，创建相同的S版本（实际使用相同结果）
    wstar_curve_S = wstar_curve_EC.copy()
    
    # 5. BMI分组（基于实际数据分布）
    logger.info("步骤 5: BMI分组")
    grouping_params = config.get('grouping', {})
    
    # 使用实际数据的BMI四分位数作为切点
    bmi_quartiles = long_df['BMI_used'].quantile([0.25, 0.5, 0.75]).values
    custom_cuts = bmi_quartiles.tolist()
    
    logger.info(f"使用BMI四分位数作为切点: {custom_cuts}")
    
    cuts_EC, groups_EC = find_bmi_cuts(
        wstar_curve_EC,
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
    
    cuts_S, groups_S = find_bmi_cuts(
        wstar_curve_S,
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
    evaluation_EC = evaluate_grouping_quality(groups_EC, wstar_curve_EC)
    evaluation_S = evaluate_grouping_quality(groups_S, wstar_curve_S)
    
    # 7. 生成报告
    logger.info("步骤 7: 生成报告")
    report_EC = create_grouping_report(groups_EC, evaluation_EC)
    report_S = create_grouping_report(groups_S, evaluation_S)
    
    # 8. 创建可视化（简化版）
    logger.info("步骤 8: 创建可视化")
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use('Agg')  # 非交互式后端
        
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
        
        # 主分析图
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # w*(b)曲线对比
        axes[0, 0].plot(wstar_curve_EC['BMI'], wstar_curve_EC['w_star_smooth'], 'b-', label='EC版本', linewidth=2)
        axes[0, 0].plot(wstar_curve_S['BMI'], wstar_curve_S['w_star_smooth'], 'r--', label='S版本', linewidth=2)
        axes[0, 0].set_xlabel('BMI')
        axes[0, 0].set_ylabel('最优时间 w*(b)')
        axes[0, 0].set_title('w*(b) 曲线对比')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # BMI分布
        axes[0, 1].hist(long_df['BMI_used'], bins=20, alpha=0.7, edgecolor='black')
        axes[0, 1].set_xlabel('BMI')
        axes[0, 1].set_ylabel('频数')
        axes[0, 1].set_title('BMI分布')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 分组结果对比
        axes[1, 0].bar(groups_EC['group_id'], groups_EC['optimal_time'], alpha=0.7, label='EC版本')
        axes[1, 0].bar(groups_S['group_id'], groups_S['optimal_time'], alpha=0.7, label='S版本')
        axes[1, 0].set_xlabel('分组ID')
        axes[1, 0].set_ylabel('推荐时间')
        axes[1, 0].set_title('分组推荐时间对比')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 最优时点分布
        axes[1, 1].hist(wstar_curve_EC['w_star'], bins=20, alpha=0.7, edgecolor='black')
        axes[1, 1].set_xlabel('最优时点 (周)')
        axes[1, 1].set_ylabel('频数')
        axes[1, 1].set_title('最优时点分布')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        main_plot = fig
        
    except ImportError:
        logger.warning("matplotlib未安装，跳过可视化")
        main_plot = None
    
    # 9. 汇总结果
    results = {
        'config': config,
        'data_summary': data_summary,
        'wstar_curve_EC': wstar_curve_EC,
        'wstar_curve_S': wstar_curve_S,
        'groups_EC': groups_EC,
        'groups_S': groups_S,
        'cuts_EC': cuts_EC,
        'cuts_S': cuts_S,
        'evaluation_EC': evaluation_EC,
        'evaluation_S': evaluation_S,
        'report_EC': report_EC,
        'report_S': report_S,
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
