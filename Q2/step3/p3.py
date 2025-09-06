"""
问题3：综合多因素的NIPT最佳时点优化
基于Q2扩展，考虑身高、体重、年龄等多种因素的影响
"""

import os
import sys
import argparse
import logging
import warnings
import pandas as pd
import numpy as np
import yaml
from pathlib import Path
from datetime import datetime

# 添加src目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# 导入模块
from io_utils_enhanced import (
    load_step1_products_enhanced,
    load_step3_config,
    save_results_enhanced,
    get_data_summary_enhanced
)
from feature_engineering import MultiFactorFeatureEngineer
from multi_factor_modeling import (
    MultiFactorYConcentrationModel,
    MultiFactorSuccessModel
)
from enhanced_error_modeling import EnhancedSigmaEstimator
from multi_objective_optimization import MultiObjectiveOptimizer
from grouping_enhanced import EnhancedBMIGrouper
from validation_enhanced import EnhancedValidator
from sensitivity_analysis_enhanced import EnhancedSensitivityAnalyzer
from report_generator import Q3ReportGenerator

# 设置警告过滤
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('p3.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='问题3：综合多因素的NIPT最佳时点优化')
    parser.add_argument('--config', type=str, default='config/step3_config.yaml',
                       help='配置文件路径')
    parser.add_argument('--data-dir', type=str, default='../step2_1',
                       help='数据目录路径（指向step2_1）')
    parser.add_argument('--output-dir', type=str, default='outputs',
                       help='输出目录路径')
    parser.add_argument('--verbose', action='store_true',
                       help='详细输出')
    parser.add_argument('--debug', action='store_true',
                       help='调试模式')
    
    args = parser.parse_args()
    
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    elif args.verbose:
        logging.getLogger().setLevel(logging.INFO)
    
    try:
        # 创建输出目录
        os.makedirs(args.output_dir, exist_ok=True)
        
        # 运行主流程
        results = run_enhanced_analysis(args.config, args.data_dir, args.output_dir)
        
        # 保存结果
        save_results_enhanced(results, args.output_dir, results['config'])
        
        logger.info("=" * 60)
        logger.info("问题3分析完成！")
        logger.info("=" * 60)
        
        # 输出主要结果摘要
        print_results_summary(results)
        
    except Exception as e:
        logger.error(f"分析失败: {e}")
        import traceback
        traceback.print_exc()
        raise


def run_enhanced_analysis(config_path: str, data_dir: str, output_dir: str) -> dict:
    """
    运行增强的分析流程（问题3）
    
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
    logger.info("=" * 60)
    logger.info("开始问题3：综合多因素的NIPT最佳时点优化分析")
    logger.info("=" * 60)
    
    # 1. 数据加载与配置
    logger.info("步骤1: 数据加载与配置")
    
    # 加载配置
    config = load_step3_config(config_path)
    logger.info(f"配置加载完成，启用功能：")
    logger.info(f"  - 多因素建模: {config['model_params']['multi_factor_features']['enabled']}")
    logger.info(f"  - 增强误差建模: {config['sigma_estimation']['multi_dimensional']}")
    logger.info(f"  - 达标比例建模: {config['attainment_modeling']['enabled']}")
    
    # 加载数据
    long_df, surv_df, report_df, config_step1 = load_step1_products_enhanced(data_dir)
    
    # 合并配置
    full_config = {**config_step1, **config}
    
    # 数据摘要
    data_summary = get_data_summary_enhanced(long_df, surv_df)
    logger.info(f"数据摘要: {data_summary['n_individuals']} 个个体, {data_summary['n_records']} 条记录")
    logger.info(f"可用特征: {list(data_summary['available_features'])}")
    
    # 2. 多因素特征工程
    logger.info("步骤2: 多因素特征工程")
    feature_engineer = MultiFactorFeatureEngineer(
        config['model_params']['multi_factor_features']
    )
    
    enhanced_df = feature_engineer.fit_transform(long_df)
    logger.info(f"特征工程完成，原始特征: {len(long_df.columns)}, 增强特征: {len(enhanced_df.columns)}")
    
    # 3. 增强的检测误差建模
    logger.info("步骤3: 增强的检测误差建模")
    sigma_estimator = EnhancedSigmaEstimator(
        config['sigma_estimation'],
        report_df
    )
    
    sigma_models = sigma_estimator.fit(enhanced_df)
    logger.info("检测误差建模完成")
    logger.info(f"  - 全局标准差: {sigma_models['global_sigma']:.4f}")
    logger.info(f"  - 局部建模维度: {len(sigma_models['local_factors'])}")
    
    # 4. 多因素Y染色体浓度建模
    logger.info("步骤4: 多因素Y染色体浓度建模")
    concentration_model = MultiFactorYConcentrationModel(
        config['model_params']
    )
    
    concentration_model.fit(enhanced_df)
    model_performance = concentration_model.get_performance_metrics()
    logger.info("Y染色体浓度建模完成")
    logger.info(f"  - 模型类型: {model_performance['model_type']}")
    logger.info(f"  - R²得分: {model_performance.get('r2_score', 'N/A')}")
    logger.info(f"  - 特征重要性前5: {list(model_performance.get('top_features', [])[:5])}")
    
    # 5. 多因素成功率建模
    logger.info("步骤5: 多因素成功率建模")
    success_model = MultiFactorSuccessModel(
        config['success_modeling'],
        enhanced_df
    )
    
    success_model.fit(enhanced_df)
    logger.info("成功率建模完成")
    
    # 6. 多目标优化
    logger.info("步骤6: 多目标优化求解")
    optimizer = MultiObjectiveOptimizer(
        config['optimization'],
        concentration_model,
        success_model,
        sigma_models
    )
    
    # 计算最优曲线
    wstar_results = optimizer.solve_optimal_timing(enhanced_df)
    logger.info("最优时点求解完成")
    logger.info(f"  - BMI范围: [{wstar_results['bmi_range'][0]:.1f}, {wstar_results['bmi_range'][1]:.1f}]")
    logger.info(f"  - 推荐时点范围: [{wstar_results['timing_range'][0]:.1f}, {wstar_results['timing_range'][1]:.1f}] 周")
    
    # 7. 增强的BMI分组
    logger.info("步骤7: 增强的BMI分组")
    grouper = EnhancedBMIGrouper(
        config['grouping']
    )
    
    grouping_results = grouper.find_optimal_groups(
        wstar_results, enhanced_df
    )
    logger.info("BMI分组完成")
    logger.info(f"  - 最终分组数: {grouping_results['n_groups']}")
    for i, group_info in enumerate(grouping_results['groups']):
        logger.info(f"  - 组{i+1}: BMI [{group_info['bmi_range'][0]:.1f}, {group_info['bmi_range'][1]:.1f}], "
                   f"推荐时点: {group_info['optimal_timing']:.1f}周, "
                   f"样本数: {group_info['n_samples']}")
    
    # 8. 模型验证
    logger.info("步骤8: 模型验证")
    validator = EnhancedValidator(
        config['validation']
    )
    
    validation_results = validator.validate_models(
        enhanced_df, concentration_model, success_model, grouping_results
    )
    logger.info("模型验证完成")
    logger.info(f"  - 交叉验证得分: {validation_results.get('cv_score', 'N/A')}")
    logger.info(f"  - 时间外推准确性: {validation_results.get('temporal_accuracy', 'N/A')}")
    
    # 9. 敏感性分析
    logger.info("步骤9: 敏感性分析")
    sensitivity_analyzer = EnhancedSensitivityAnalyzer(
        config['sensitivity']
    )
    
    sensitivity_results = sensitivity_analyzer.analyze(
        enhanced_df, concentration_model, success_model, 
        grouping_results, sigma_models
    )
    logger.info("敏感性分析完成")
    logger.info(f"  - 参数敏感性测试: {len(sensitivity_results['parameter_sensitivity'])} 个场景")
    logger.info(f"  - 误差敏感性测试: {len(sensitivity_results['error_sensitivity'])} 个场景")
    
    # 10. 生成报告
    logger.info("步骤10: 生成综合报告")
    report_generator = Q3ReportGenerator(
        config['output']
    )
    
    report_results = report_generator.generate_comprehensive_report(
        enhanced_df=enhanced_df,
        concentration_model=concentration_model,
        success_model=success_model,
        sigma_models=sigma_models,
        wstar_results=wstar_results,
        grouping_results=grouping_results,
        validation_results=validation_results,
        sensitivity_results=sensitivity_results,
        config=full_config,
        output_dir=output_dir
    )
    
    logger.info("报告生成完成")
    logger.info(f"  - 输出文件: {len(report_results['output_files'])} 个")
    
    # 汇总所有结果
    final_results = {
        'config': full_config,
        'data_summary': data_summary,
        'enhanced_df': enhanced_df,
        'feature_engineering': {
            'engineer': feature_engineer,
            'feature_names': list(enhanced_df.columns)
        },
        'sigma_models': sigma_models,
        'concentration_model': concentration_model,
        'success_model': success_model,
        'wstar_results': wstar_results,
        'grouping_results': grouping_results,
        'validation_results': validation_results,
        'sensitivity_results': sensitivity_results,
        'report_results': report_results,
        'performance_metrics': {
            'model_performance': model_performance,
            'data_quality': data_summary
        }
    }
    
    logger.info("问题3分析流程完成")
    return final_results


def print_results_summary(results):
    """打印结果摘要"""
    print("\n" + "=" * 60)
    print("问题3分析结果摘要")
    print("=" * 60)
    
    # 分组结果
    grouping = results['grouping_results']
    print(f"\n【BMI分组结果】")
    print(f"最终分组数: {grouping['n_groups']}")
    
    for i, group in enumerate(grouping['groups']):
        print(f"组{i+1}:")
        print(f"  BMI范围: [{group['bmi_range'][0]:.1f}, {group['bmi_range'][1]:.1f}]")
        print(f"  推荐时点: {group['optimal_timing']:.1f} 周")
        print(f"  预期成功率: {group.get('expected_success_rate', 'N/A')}")
        print(f"  样本数量: {group['n_samples']}")
        print(f"  风险水平: {group.get('risk_level', 'N/A')}")
    
    # 模型性能
    perf = results['performance_metrics']['model_performance']
    print(f"\n【模型性能】")
    print(f"浓度预测模型R²: {perf.get('r2_score', 'N/A')}")
    print(f"重要特征: {', '.join(perf.get('top_features', [])[:3])}")
    
    # 敏感性分析结果
    sens = results['sensitivity_results']
    print(f"\n【敏感性分析】")
    print(f"检测误差影响: {sens.get('error_impact_summary', 'N/A')}")
    print(f"参数稳定性: {sens.get('parameter_stability', 'N/A')}")
    
    # 输出文件
    files = results['report_results']['output_files']
    print(f"\n【输出文件】")
    for file in files[:5]:  # 显示前5个文件
        print(f"  {file}")
    if len(files) > 5:
        print(f"  ... 和其他 {len(files) - 5} 个文件")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()