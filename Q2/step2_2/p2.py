"""
基于经验数据的理论模型主程序
使用数学公式支撑的理论框架，便于后续迭代
"""

import sys
import os
import logging
import argparse
from pathlib import Path

# 添加src目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from io_utils import load_step1_products, load_step2_config
from empirical_model import create_empirical_model_from_data, EmpiricalDetectionModel

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    """主程序入口"""
    parser = argparse.ArgumentParser(description='基于经验数据的理论模型分析')
    parser.add_argument('--config', default='config/step2_config.yaml', help='配置文件路径')
    parser.add_argument('--data-dir', default='../step2_1', help='数据目录')
    parser.add_argument('--output-dir', default='outputs', help='输出目录')
    parser.add_argument('--verbose', action='store_true', help='详细输出')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    logger.info("开始基于经验数据的理论模型分析...")
    
    try:
        # 1. 读取数据与配置
        logger.info("步骤 1: 读取数据与配置")
        long_df, surv_df, report, step1_config = load_step1_products(args.data_dir)
        config = load_step2_config(args.config)
        
        logger.info(f"数据摘要: {len(long_df)} 条记录, {long_df['id'].nunique()} 个个体")
        
        # 2. 创建经验检测模型
        logger.info("步骤 2: 创建经验检测模型")
        empirical_model = create_empirical_model_from_data(long_df)
        
        # 显示模型参数
        params = empirical_model.get_model_parameters()
        logger.info(f"模型参数: α={params['alpha']:.3f}, β={params['beta']:.3f}")
        logger.info(f"约束条件: t_min={params['t_min']:.1f}, t_max={params['t_max']:.1f}")
        
        # 3. 生成w*(b)曲线
        logger.info("步骤 3: 生成w*(b)曲线")
        
        # 创建BMI网格
        bmi_min = long_df['BMI_used'].min()
        bmi_max = long_df['BMI_used'].max()
        bmi_grid = np.linspace(bmi_min, bmi_max, 40)
        
        # 计算每个BMI点的最优时点
        wstar_curve_data = []
        for bmi in bmi_grid:
            optimal_time = empirical_model.predict_optimal_time(bmi)
            lower_bound, upper_bound = empirical_model.predict_confidence_interval(bmi)
            
            wstar_curve_data.append({
                'BMI': bmi,
                'w_star': optimal_time,
                'w_star_smooth': optimal_time,
                'min_risk': 0.1,  # 经验模型的风险值
                'lower_bound': lower_bound,
                'upper_bound': upper_bound
            })
        
        wstar_curve = pd.DataFrame(wstar_curve_data)
        logger.info(f"生成了 {len(wstar_curve)} 个BMI点的w*(b)曲线")
        
        # 4. BMI分组
        logger.info("步骤 4: BMI分组")
        grouping_params = config.get('grouping', {})
        custom_cuts = grouping_params.get('custom_cuts', [20.0, 30.5, 32.7, 34.4, 50.0])
        
        logger.info(f"使用自定义BMI切点: {custom_cuts}")
        
        # 使用经验模型进行分组
        bmi_groups = {
            '低BMI组': (20.0, 30.5),
            '中BMI组': (30.5, 32.7),
            '高BMI组': (32.7, 34.4),
            '极高BMI组': (34.4, 50.0)
        }
        
        group_times = empirical_model.predict_group_time(bmi_groups)
        
        # 创建分组结果
        groups_data = []
        for i, (group_name, (bmi_min, bmi_max)) in enumerate(bmi_groups.items(), 1):
            bmi_median = (bmi_min + bmi_max) / 2
            optimal_time = group_times[group_name]
            
            # 计算该组的实际数据统计
            group_mask = (long_df['BMI_used'] >= bmi_min) & (long_df['BMI_used'] < bmi_max)
            group_data = long_df[group_mask]
            n_points = len(group_data)
            
            groups_data.append({
                'group_id': i,
                'bmi_min': bmi_min,
                'bmi_max': bmi_max,
                'bmi_mean': group_data['BMI_used'].mean() if n_points > 0 else bmi_median,
                'n_points': n_points,
                'optimal_time': optimal_time,
                'time_std': 0.0,  # 经验模型的标准差
                'mean_risk': 0.1
            })
        
        groups_df = pd.DataFrame(groups_data)
        
        # 5. 模型评估
        logger.info("步骤 5: 模型评估")
        
        # 计算实际数据的首次达标时点
        first_hit = long_df[long_df['Y_frac'] > 0.04].groupby(['id', 'bmi_group'])['week'].min().reset_index()
        
        # 评估模型性能
        bmi_data = []
        time_data = []
        for group in ['低BMI组', '中BMI组', '高BMI组', '极高BMI组']:
            group_data = first_hit[first_hit['bmi_group'] == group]
            if len(group_data) > 0:
                bmi_median = long_df[long_df['bmi_group'] == group]['BMI_used'].median()
                time_mean = group_data['week'].mean()
                bmi_data.append(bmi_median)
                time_data.append(time_mean)
        
        if len(bmi_data) > 0:
            performance = empirical_model.evaluate_model(np.array(bmi_data), np.array(time_data))
            logger.info(f"模型性能: MAE={performance['mae']:.2f}, RMSE={performance['rmse']:.2f}, R={performance['correlation']:.3f}")
        
        # 6. 生成报告
        logger.info("步骤 6: 生成报告")
        
        report_content = f"""=== 基于经验数据的理论模型分析报告 ===

1. 模型参数
   - 对数关系: t = {params['alpha']:.3f} × ln(BMI) + {params['beta']:.3f}
   - 个体变异修正系数: {params['gamma']:.3f}
   - BMI偏离修正系数: {params['delta']:.3f}
   - 参考BMI: {params['bmi_ref']:.1f}
   - 时点约束: {params['t_min']:.1f} - {params['t_max']:.1f} 周

2. BMI分组推荐
"""
        
        for _, row in groups_df.iterrows():
            report_content += f"   组 {row['group_id']}: BMI [{row['bmi_min']:.1f}, {row['bmi_max']:.1f}), 推荐时点 {row['optimal_time']:.1f} 周\n"
        
        if len(bmi_data) > 0:
            report_content += f"""
3. 模型性能
   - 平均绝对误差: {performance['mae']:.2f} 周
   - 均方根误差: {performance['rmse']:.2f} 周
   - 相关系数: {performance['correlation']:.3f}

4. 数学公式
   基础时点: t_base(BMI) = {params['alpha']:.3f} × ln(BMI) + {params['beta']:.3f}
   最优时点: t_optimal(BMI,σ) = t_base(BMI) + {params['gamma']:.3f} × σ + {params['delta']:.3f} × (BMI - {params['bmi_ref']:.1f})²
   置信区间: CI(t) = t_optimal(BMI,σ) ± 1.96 × SE(t)
"""
        
        # 7. 保存结果
        logger.info("步骤 7: 保存结果")
        
        # 确保输出目录存在
        os.makedirs(args.output_dir, exist_ok=True)
        
        # 保存文件
        wstar_curve.to_csv(f"{args.output_dir}/p2_empirical_wstar_curve.csv", index=False)
        groups_df.to_csv(f"{args.output_dir}/p2_empirical_group_recommendation.csv", index=False)
        
        with open(f"{args.output_dir}/p2_empirical_report.txt", 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        # 保存模型参数
        import yaml
        with open(f"{args.output_dir}/empirical_model_params.yaml", 'w', encoding='utf-8') as f:
            yaml.dump(params, f, default_flow_style=False, allow_unicode=True)
        
        logger.info("分析完成！")
        logger.info(f"结果已保存到 {args.output_dir}/")
        
        # 显示结果摘要
        print("\n=== 结果摘要 ===")
        print("BMI分组推荐时点:")
        for _, row in groups_df.iterrows():
            print(f"  组 {row['group_id']}: {row['bmi_min']:.1f}-{row['bmi_max']:.1f} BMI → {row['optimal_time']:.1f} 周")
        
        if len(bmi_data) > 0:
            print(f"\n模型性能: MAE={performance['mae']:.2f}周, R={performance['correlation']:.3f}")
        
    except Exception as e:
        logger.error(f"分析失败: {e}")
        raise


if __name__ == "__main__":
    import numpy as np
    import pandas as pd
    main()
