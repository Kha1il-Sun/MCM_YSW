#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BMI分组回归分析 - 简化版本
根据新的BMI分组标准对数据进行分组，建立独立模型和分段回归模型
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')

def load_data():
    """加载数据"""
    X = pd.read_csv('../step1/X_features.csv')
    y = pd.read_csv('../step1/y_target.csv')
    
    # 合并数据
    data = pd.concat([X, y], axis=1)
    data.columns = ['孕周数', 'BMI', 'Y染色体浓度']
    
    return data

def create_bmi_groups(data):
    """根据新的BMI分组标准创建分组"""
    # 定义BMI分组标准：[20,28)，[28,32)，[32,36)，[36,40)，40以上
    def categorize_bmi(bmi):
        if bmi < 20:
            return 'BMI<20'
        elif 20 <= bmi < 28:
            return '[20,28)'
        elif 28 <= bmi < 32:
            return '[28,32)'
        elif 32 <= bmi < 36:
            return '[32,36)'
        elif 36 <= bmi < 40:
            return '[36,40)'
        else:
            return 'BMI≥40'
    
    data['BMI组'] = data['BMI'].apply(categorize_bmi)
    
    return data

def analyze_groups(data):
    """分析各BMI组的基本统计信息"""
    print("=" * 60)
    print("BMI分组统计信息")
    print("=" * 60)
    
    group_stats = data.groupby('BMI组').agg({
        '孕周数': ['count', 'mean', 'std', 'min', 'max'],
        'BMI': ['mean', 'std', 'min', 'max'],
        'Y染色体浓度': ['mean', 'std', 'min', 'max']
    }).round(4)
    
    print(group_stats)
    
    return group_stats

def build_independent_models(data):
    """为每个BMI组建立独立的回归模型"""
    print("\n" + "=" * 60)
    print("独立模型分析")
    print("=" * 60)
    
    groups = data['BMI组'].unique()
    models = {}
    results = {}
    
    for group in groups:
        group_data = data[data['BMI组'] == group]
        
        if len(group_data) < 3:  # 数据点太少，跳过
            print(f"{group}组数据点太少（{len(group_data)}个），跳过建模")
            continue
        
        print(f"\n{group}组 (n={len(group_data)})")
        
        X = group_data[['孕周数']].values
        y = group_data['Y染色体浓度'].values
        
        # 线性回归
        lr_model = LinearRegression()
        lr_model.fit(X, y)
        y_pred_lr = lr_model.predict(X)
        
        # 多项式回归（2次）
        poly_model = Pipeline([
            ('poly', PolynomialFeatures(degree=2)),
            ('linear', LinearRegression())
        ])
        poly_model.fit(X, y)
        y_pred_poly = poly_model.predict(X)
        
        # 计算评估指标
        lr_r2 = r2_score(y, y_pred_lr)
        lr_rmse = np.sqrt(mean_squared_error(y, y_pred_lr))
        lr_mae = mean_absolute_error(y, y_pred_lr)
        
        poly_r2 = r2_score(y, y_pred_poly)
        poly_rmse = np.sqrt(mean_squared_error(y, y_pred_poly))
        poly_mae = mean_absolute_error(y, y_pred_poly)
        
        # 存储结果
        models[group] = {
            'linear': lr_model,
            'polynomial': poly_model
        }
        
        results[group] = {
            'linear': {'R2': lr_r2, 'RMSE': lr_rmse, 'MAE': lr_mae},
            'polynomial': {'R2': poly_r2, 'RMSE': poly_rmse, 'MAE': poly_mae}
        }
        
        # 打印结果
        print(f"线性模型: R²={lr_r2:.4f}, RMSE={lr_rmse:.4f}, MAE={lr_mae:.4f}")
        print(f"多项式模型: R²={poly_r2:.4f}, RMSE={poly_rmse:.4f}, MAE={poly_mae:.4f}")
    
    return models, results

def build_piecewise_regression(data):
    """建立分段回归模型"""
    print("\n" + "=" * 60)
    print("分段回归分析")
    print("=" * 60)
    
    # 定义断点：[20,28)，[28,32)，[32,36)，[36,40)，40以上
    breakpoints = [20, 28, 32, 36, 40]
    
    # 创建分段特征
    data['segment_1'] = np.where(data['BMI'] < breakpoints[0], 1, 0)  # BMI<20
    data['segment_2'] = np.where((data['BMI'] >= breakpoints[0]) & (data['BMI'] < breakpoints[1]), 1, 0)  # [20,28)
    data['segment_3'] = np.where((data['BMI'] >= breakpoints[1]) & (data['BMI'] < breakpoints[2]), 1, 0)  # [28,32)
    data['segment_4'] = np.where((data['BMI'] >= breakpoints[2]) & (data['BMI'] < breakpoints[3]), 1, 0)  # [32,36)
    data['segment_5'] = np.where((data['BMI'] >= breakpoints[3]) & (data['BMI'] < breakpoints[4]), 1, 0)  # [36,40)
    data['segment_6'] = np.where(data['BMI'] >= breakpoints[4], 1, 0)  # BMI≥40
    
    # 创建交互项
    data['孕周数_段1'] = data['孕周数'] * data['segment_1']
    data['孕周数_段2'] = data['孕周数'] * data['segment_2']
    data['孕周数_段3'] = data['孕周数'] * data['segment_3']
    data['孕周数_段4'] = data['孕周数'] * data['segment_4']
    data['孕周数_段5'] = data['孕周数'] * data['segment_5']
    data['孕周数_段6'] = data['孕周数'] * data['segment_6']
    
    # 准备特征矩阵
    X_piecewise = data[['segment_1', 'segment_2', 'segment_3', 'segment_4', 'segment_5', 'segment_6',
                       '孕周数_段1', '孕周数_段2', '孕周数_段3', '孕周数_段4', '孕周数_段5', '孕周数_段6']].values
    y = data['Y染色体浓度'].values
    
    # 建立分段回归模型
    piecewise_model = LinearRegression()
    piecewise_model.fit(X_piecewise, y)
    y_pred_piecewise = piecewise_model.predict(X_piecewise)
    
    # 计算评估指标
    r2 = r2_score(y, y_pred_piecewise)
    rmse = np.sqrt(mean_squared_error(y, y_pred_piecewise))
    mae = mean_absolute_error(y, y_pred_piecewise)
    
    print(f"分段回归模型性能:")
    print(f"R² = {r2:.4f}")
    print(f"RMSE = {rmse:.4f}")
    print(f"MAE = {mae:.4f}")
    
    # 打印各段的系数
    feature_names = ['段1截距(BMI<20)', '段2截距([20,28))', '段3截距([28,32))', '段4截距([32,36))', '段5截距([36,40))', '段6截距(BMI≥40)',
                    '段1斜率(BMI<20)', '段2斜率([20,28))', '段3斜率([28,32))', '段4斜率([32,36))', '段5斜率([36,40))', '段6斜率(BMI≥40)']
    print(f"\n分段回归系数:")
    for name, coef in zip(feature_names, piecewise_model.coef_):
        print(f"{name}: {coef:.6f}")
    print(f"总截距: {piecewise_model.intercept_:.6f}")
    
    return piecewise_model, {'R2': r2, 'RMSE': rmse, 'MAE': mae}

def compare_models(independent_results, piecewise_results):
    """比较不同模型的性能"""
    print("\n" + "=" * 60)
    print("模型性能比较")
    print("=" * 60)
    
    # 创建比较表
    comparison_data = []
    
    # 独立模型结果
    for group, results in independent_results.items():
        for model_type, metrics in results.items():
            comparison_data.append({
                '模型类型': f'{group}_{model_type}',
                'R²': metrics['R2'],
                'RMSE': metrics['RMSE'],
                'MAE': metrics['MAE']
            })
    
    # 分段回归结果
    comparison_data.append({
        '模型类型': '分段回归',
        'R²': piecewise_results['R2'],
        'RMSE': piecewise_results['RMSE'],
        'MAE': piecewise_results['MAE']
    })
    
    comparison_df = pd.DataFrame(comparison_data)
    comparison_df = comparison_df.sort_values('R²', ascending=False)
    
    print(comparison_df.to_string(index=False))
    
    return comparison_df

def main():
    """主函数"""
    print("BMI分组回归分析 - 简化版本")
    print("=" * 60)
    
    # 加载数据
    print("正在加载数据...")
    data = load_data()
    print(f"数据形状: {data.shape}")
    
    # 创建BMI分组
    print("\n正在创建BMI分组...")
    data = create_bmi_groups(data)
    
    # 分析各组
    print("\n正在分析各组统计信息...")
    group_stats = analyze_groups(data)
    
    # 建立独立模型
    print("\n正在建立独立模型...")
    independent_models, independent_results = build_independent_models(data)
    
    # 建立分段回归模型
    print("\n正在建立分段回归模型...")
    piecewise_model, piecewise_results = build_piecewise_regression(data)
    
    # 比较模型性能
    print("\n正在比较模型性能...")
    comparison_df = compare_models(independent_results, piecewise_results)
    
    # 保存结果
    print("\n正在保存结果...")
    group_stats.to_csv('bmi_group_statistics.csv')
    comparison_df.to_csv('model_comparison_results.csv')
    
    print("\n分析完成！结果已保存到文件。")

if __name__ == "__main__":
    main()
