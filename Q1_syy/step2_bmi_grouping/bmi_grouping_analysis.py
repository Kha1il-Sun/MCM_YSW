#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BMI分组回归分析
根据临床BMI标准对数据进行分组，建立独立模型和分段回归模型
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

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
    
    # 创建可视化
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # BMI分布
    axes[0, 0].hist(data['BMI'], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0, 0].axvline(x=20, color='red', linestyle='--', label='20')
    axes[0, 0].axvline(x=28, color='orange', linestyle='--', label='28')
    axes[0, 0].axvline(x=32, color='green', linestyle='--', label='32')
    axes[0, 0].axvline(x=36, color='purple', linestyle='--', label='36')
    axes[0, 0].axvline(x=40, color='brown', linestyle='--', label='40')
    axes[0, 0].set_xlabel('BMI')
    axes[0, 0].set_ylabel('频数')
    axes[0, 0].set_title('BMI分布直方图（新分组标准）')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # BMI组分布
    bmi_counts = data['BMI组'].value_counts()
    axes[0, 1].pie(bmi_counts.values, labels=bmi_counts.index, autopct='%1.1f%%', startangle=90)
    axes[0, 1].set_title('BMI组分布')
    
    # 各组的Y染色体浓度分布
    groups = data['BMI组'].unique()
    colors = ['lightblue', 'lightgreen', 'orange', 'pink', 'lightcoral', 'lightyellow']
    for i, group in enumerate(groups):
        group_data = data[data['BMI组'] == group]['Y染色体浓度']
        axes[1, 0].hist(group_data, alpha=0.6, label=group, color=colors[i % len(colors)])
    axes[1, 0].set_xlabel('Y染色体浓度')
    axes[1, 0].set_ylabel('频数')
    axes[1, 0].set_title('各BMI组的Y染色体浓度分布')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 孕周数 vs Y染色体浓度散点图（按BMI组着色）
    for i, group in enumerate(groups):
        group_data = data[data['BMI组'] == group]
        axes[1, 1].scatter(group_data['孕周数'], group_data['Y染色体浓度'], 
                          label=group, alpha=0.6, color=colors[i % len(colors)])
    axes[1, 1].set_xlabel('孕周数')
    axes[1, 1].set_ylabel('Y染色体浓度')
    axes[1, 1].set_title('孕周数 vs Y染色体浓度（按BMI组着色）')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('bmi_group_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return group_stats

def build_independent_models(data):
    """为每个BMI组建立独立的回归模型"""
    print("\n" + "=" * 60)
    print("独立模型分析")
    print("=" * 60)
    
    groups = data['BMI组'].unique()
    models = {}
    results = {}
    
    # 计算需要的子图数量
    valid_groups = [group for group in groups if len(data[data['BMI组'] == group]) >= 3]
    n_groups = len(valid_groups)
    
    if n_groups == 0:
        print("没有足够的组进行建模")
        return models, results
    
    # 动态调整子图布局
    if n_groups <= 2:
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        axes = axes.flatten()
    elif n_groups <= 4:
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()
    else:
        fig, axes = plt.subplots(3, 2, figsize=(15, 18))
        axes = axes.flatten()
    
    for i, group in enumerate(valid_groups):
        group_data = data[data['BMI组'] == group]
        
        if len(group_data) < 3:  # 数据点太少，跳过
            print(f"{group}组数据点太少（{len(group_data)}个），跳过建模")
            continue
        
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
        print(f"\n{group}组 (n={len(group_data)})")
        print(f"线性模型: R²={lr_r2:.4f}, RMSE={lr_r2:.4f}, MAE={lr_mae:.4f}")
        print(f"多项式模型: R²={poly_r2:.4f}, RMSE={poly_rmse:.4f}, MAE={poly_mae:.4f}")
        
        # 绘制拟合结果
        if i < len(axes):
            axes[i].scatter(X, y, alpha=0.6, label='实际值', color='blue')
            
            # 绘制线性拟合
            X_sorted = np.sort(X, axis=0)
            y_pred_lr_sorted = lr_model.predict(X_sorted)
            axes[i].plot(X_sorted, y_pred_lr_sorted, 'r-', label=f'线性拟合 (R²={lr_r2:.3f})', linewidth=2)
            
            # 绘制多项式拟合
            y_pred_poly_sorted = poly_model.predict(X_sorted)
            axes[i].plot(X_sorted, y_pred_poly_sorted, 'g--', label=f'多项式拟合 (R²={poly_r2:.3f})', linewidth=2)
            
            axes[i].set_xlabel('孕周数')
            axes[i].set_ylabel('Y染色体浓度')
            axes[i].set_title(f'{group}组回归拟合')
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)
    
    # 隐藏多余的子图
    for i in range(n_groups, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.savefig('independent_models_fitting.png', dpi=300, bbox_inches='tight')
    plt.show()
    
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
    
    # 可视化分段回归结果
    plt.figure(figsize=(12, 8))
    
    # 按BMI组着色绘制散点图
    groups = data['BMI组'].unique()
    colors = ['lightblue', 'lightgreen', 'orange', 'pink', 'lightcoral', 'lightyellow']
    
    for i, group in enumerate(groups):
        group_data = data[data['BMI组'] == group]
        plt.scatter(group_data['孕周数'], group_data['Y染色体浓度'], 
                   label=group, alpha=0.6, color=colors[i % len(colors)])
    
    # 绘制分段回归线
    X_plot = np.linspace(data['孕周数'].min(), data['孕周数'].max(), 100).reshape(-1, 1)
    
    # 为每个BMI段绘制回归线
    bmi_representatives = [15, 24, 30, 34, 38, 42]  # 各段的代表性BMI值
    segment_labels = ['BMI<20', '[20,28)', '[28,32)', '[32,36)', '[36,40)', 'BMI≥40']
    
    for i, bmi_val in enumerate(bmi_representatives):
        segment_data = np.zeros((100, 12))
        if i == 0:  # BMI<20
            segment_data[:, 0] = 1
            segment_data[:, 6] = X_plot.flatten()
        elif i == 1:  # [20,28)
            segment_data[:, 1] = 1
            segment_data[:, 7] = X_plot.flatten()
        elif i == 2:  # [28,32)
            segment_data[:, 2] = 1
            segment_data[:, 8] = X_plot.flatten()
        elif i == 3:  # [32,36)
            segment_data[:, 3] = 1
            segment_data[:, 9] = X_plot.flatten()
        elif i == 4:  # [36,40)
            segment_data[:, 4] = 1
            segment_data[:, 10] = X_plot.flatten()
        else:  # BMI≥40
            segment_data[:, 5] = 1
            segment_data[:, 11] = X_plot.flatten()
        
        y_plot = piecewise_model.predict(segment_data)
        plt.plot(X_plot, y_plot, '--', linewidth=2, 
                label=f'{segment_labels[i]}拟合线', alpha=0.8)
    
    plt.xlabel('孕周数')
    plt.ylabel('Y染色体浓度')
    plt.title('分段回归模型拟合结果')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('piecewise_regression.png', dpi=300, bbox_inches='tight')
    plt.show()
    
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
    
    # 可视化比较
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    metrics = ['R²', 'RMSE', 'MAE']
    for i, metric in enumerate(metrics):
        axes[i].bar(range(len(comparison_df)), comparison_df[metric])
        axes[i].set_xticks(range(len(comparison_df)))
        axes[i].set_xticklabels(comparison_df['模型类型'], rotation=45, ha='right')
        axes[i].set_ylabel(metric)
        axes[i].set_title(f'各模型{metric}比较')
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return comparison_df

def main():
    """主函数"""
    print("BMI分组回归分析")
    print("=" * 60)
    
    # 加载数据
    print("正在加载数据...")
    data = load_data()
    print(f"数据形状: {data.shape}")
    print(f"数据预览:")
    print(data.head())
    
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
