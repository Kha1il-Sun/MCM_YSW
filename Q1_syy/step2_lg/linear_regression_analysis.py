#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
线性回归分析
以Y染色体浓度为target，孕周数和BMI为features
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def load_and_explore_data():
    """加载和探索数据"""
    print("=" * 50)
    print("数据加载和探索")
    print("=" * 50)
    
    # 加载数据
    df = pd.read_csv('../step1/normalized_merged_data.csv')
    
    print(f"数据形状: {df.shape}")
    print(f"\n列名: {list(df.columns)}")
    print(f"\n数据类型:\n{df.dtypes}")
    print(f"\n前5行数据:\n{df.head()}")
    print(f"\n数据基本统计信息:\n{df.describe()}")
    
    # 检查缺失值
    print(f"\n缺失值检查:\n{df.isnull().sum()}")
    
    return df

def prepare_data(df):
    """准备特征变量和目标变量"""
    print("\n" + "=" * 50)
    print("数据准备")
    print("=" * 50)
    
    # 分离特征和目标变量
    X = df[['孕周数', 'BMI']].copy()
    y = df['Y染色体浓度'].copy()
    
    print(f"特征变量形状: {X.shape}")
    print(f"目标变量形状: {y.shape}")
    print(f"\n特征变量统计信息:\n{X.describe()}")
    print(f"\n目标变量统计信息:\n{y.describe()}")
    
    # 检查特征之间的相关性
    correlation_matrix = df[['孕周数', 'BMI', 'Y染色体浓度']].corr()
    print(f"\n相关性矩阵:\n{correlation_matrix}")
    
    return X, y

def perform_linear_regression(X, y):
    """执行线性回归"""
    print("\n" + "=" * 50)
    print("线性回归建模")
    print("=" * 50)
    
    # 分割数据
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"训练集大小: {X_train.shape[0]}")
    print(f"测试集大小: {X_test.shape[0]}")
    
    # 标准化特征
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 创建和训练模型
    model = LinearRegression()
    model.fit(X_train_scaled, y_train)
    
    # 预测
    y_train_pred = model.predict(X_train_scaled)
    y_test_pred = model.predict(X_test_scaled)
    
    # 计算评估指标
    train_mse = mean_squared_error(y_train, y_train_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    train_mae = mean_absolute_error(y_train, y_train_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    
    print(f"\n模型性能评估:")
    print(f"训练集 - MSE: {train_mse:.6f}, R²: {train_r2:.6f}, MAE: {train_mae:.6f}")
    print(f"测试集 - MSE: {test_mse:.6f}, R²: {test_r2:.6f}, MAE: {test_mae:.6f}")
    
    # 模型系数
    print(f"\n模型系数:")
    print(f"截距: {model.intercept_:.6f}")
    for i, feature in enumerate(X.columns):
        print(f"{feature}系数: {model.coef_[i]:.6f}")
    
    return model, scaler, X_train, X_test, y_train, y_test, y_train_pred, y_test_pred

def visualize_results(X, y, model, scaler, X_train, X_test, y_train, y_test, y_train_pred, y_test_pred):
    """可视化结果"""
    print("\n" + "=" * 50)
    print("结果可视化")
    print("=" * 50)
    
    # 创建图形
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('线性回归分析结果', fontsize=16, fontweight='bold')
    
    # 1. 实际值 vs 预测值 (训练集)
    axes[0, 0].scatter(y_train, y_train_pred, alpha=0.6, color='blue')
    axes[0, 0].plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'r--', lw=2)
    axes[0, 0].set_xlabel('实际值')
    axes[0, 0].set_ylabel('预测值')
    axes[0, 0].set_title('训练集: 实际值 vs 预测值')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. 实际值 vs 预测值 (测试集)
    axes[0, 1].scatter(y_test, y_test_pred, alpha=0.6, color='red')
    axes[0, 1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    axes[0, 1].set_xlabel('实际值')
    axes[0, 1].set_ylabel('预测值')
    axes[0, 1].set_title('测试集: 实际值 vs 预测值')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. 残差图 (训练集)
    residuals_train = y_train - y_train_pred
    axes[0, 2].scatter(y_train_pred, residuals_train, alpha=0.6, color='blue')
    axes[0, 2].axhline(y=0, color='r', linestyle='--')
    axes[0, 2].set_xlabel('预测值')
    axes[0, 2].set_ylabel('残差')
    axes[0, 2].set_title('训练集残差图')
    axes[0, 2].grid(True, alpha=0.3)
    
    # 4. 残差图 (测试集)
    residuals_test = y_test - y_test_pred
    axes[1, 0].scatter(y_test_pred, residuals_test, alpha=0.6, color='red')
    axes[1, 0].axhline(y=0, color='r', linestyle='--')
    axes[1, 0].set_xlabel('预测值')
    axes[1, 0].set_ylabel('残差')
    axes[1, 0].set_title('测试集残差图')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 5. 特征重要性
    feature_importance = np.abs(model.coef_)
    feature_names = X.columns
    bars = axes[1, 1].bar(feature_names, feature_importance, color=['skyblue', 'lightcoral'])
    axes[1, 1].set_xlabel('特征')
    axes[1, 1].set_ylabel('系数绝对值')
    axes[1, 1].set_title('特征重要性 (系数绝对值)')
    axes[1, 1].tick_params(axis='x', rotation=45)
    
    # 添加数值标签
    for bar, importance in zip(bars, feature_importance):
        axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                       f'{importance:.4f}', ha='center', va='bottom')
    
    # 6. 预测误差分布
    axes[1, 2].hist(residuals_test, bins=30, alpha=0.7, color='green', edgecolor='black')
    axes[1, 2].set_xlabel('残差')
    axes[1, 2].set_ylabel('频次')
    axes[1, 2].set_title('测试集残差分布')
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('linear_regression_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 创建3D散点图显示特征与目标的关系
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # 绘制实际数据点
    scatter = ax.scatter(X['孕周数'], X['BMI'], y, c=y, cmap='viridis', alpha=0.6)
    
    # 创建网格用于绘制回归平面
    x1_range = np.linspace(X['孕周数'].min(), X['孕周数'].max(), 20)
    x2_range = np.linspace(X['BMI'].min(), X['BMI'].max(), 20)
    X1_grid, X2_grid = np.meshgrid(x1_range, x2_range)
    
    # 标准化网格数据
    grid_data = np.column_stack([X1_grid.ravel(), X2_grid.ravel()])
    grid_data_scaled = scaler.transform(grid_data)
    
    # 预测网格点的Y值
    Y_grid_pred = model.predict(grid_data_scaled).reshape(X1_grid.shape)
    
    # 绘制回归平面
    ax.plot_surface(X1_grid, X2_grid, Y_grid_pred, alpha=0.3, color='red')
    
    ax.set_xlabel('孕周数')
    ax.set_ylabel('BMI')
    ax.set_zlabel('Y染色体浓度')
    ax.set_title('3D回归平面: 孕周数、BMI vs Y染色体浓度')
    
    plt.colorbar(scatter, ax=ax, shrink=0.5, aspect=5)
    plt.savefig('3d_regression_plane.png', dpi=300, bbox_inches='tight')
    plt.show()

def save_results(model, scaler, X, y, y_train_pred, y_test_pred):
    """保存结果"""
    print("\n" + "=" * 50)
    print("保存结果")
    print("=" * 50)
    
    # 保存模型系数和性能指标
    results = {
        'model_intercept': model.intercept_,
        'model_coefficients': dict(zip(X.columns, model.coef_)),
        'feature_names': list(X.columns),
        'n_features': len(X.columns),
        'n_samples': len(y)
    }
    
    # 保存到文件
    with open('linear_regression_results.txt', 'w', encoding='utf-8') as f:
        f.write("线性回归分析结果\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"模型截距: {model.intercept_:.6f}\n\n")
        f.write("特征系数:\n")
        for feature, coef in results['model_coefficients'].items():
            f.write(f"  {feature}: {coef:.6f}\n")
        f.write(f"\n特征数量: {results['n_features']}\n")
        f.write(f"样本数量: {results['n_samples']}\n")
    
    print("结果已保存到 linear_regression_results.txt")
    print("可视化图表已保存为 linear_regression_analysis.png 和 3d_regression_plane.png")

def main():
    """主函数"""
    print("开始线性回归分析...")
    
    # 1. 加载和探索数据
    df = load_and_explore_data()
    
    # 2. 准备数据
    X, y = prepare_data(df)
    
    # 3. 执行线性回归
    model, scaler, X_train, X_test, y_train, y_test, y_train_pred, y_test_pred = perform_linear_regression(X, y)
    
    # 4. 可视化结果
    visualize_results(X, y, model, scaler, X_train, X_test, y_train, y_test, y_train_pred, y_test_pred)
    
    # 5. 保存结果
    save_results(model, scaler, X, y, y_train_pred, y_test_pred)
    
    print("\n线性回归分析完成！")

if __name__ == "__main__":
    main()
