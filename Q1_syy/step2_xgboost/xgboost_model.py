#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
XGBoost 回归模型实现
用于预测Y染色体浓度
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import os
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

def load_data():
    """加载特征数据和目标变量数据"""
    print("正在加载数据...")
    
    # 获取当前脚本所在目录的父目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    step1_dir = os.path.join(parent_dir, 'step1')
    
    # 读取特征数据和目标变量数据
    X_path = os.path.join(step1_dir, 'X_features.csv')
    y_path = os.path.join(step1_dir, 'y_target.csv')
    
    X = pd.read_csv(X_path)  # 特征数据
    y_df = pd.read_csv(y_path)  # 目标变量数据
    
    # 确保目标变量是Series格式
    y = y_df['Y染色体浓度'].copy()
    
    print(f"特征数据形状: {X.shape}")
    print(f"目标变量形状: {y.shape}")
    print("\n特征数据前5行:")
    print(X.head())
    print("\n目标变量前5行:")
    print(y.head())
    
    return X, y

def split_data(X, y, test_size=0.2, random_state=42):
    """数据拆分"""
    print(f"\n正在拆分数据 (训练集: {1-test_size:.0%}, 测试集: {test_size:.0%})...")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    print(f"训练集形状: X_train {X_train.shape}, y_train {y_train.shape}")
    print(f"测试集形状: X_test {X_test.shape}, y_test {y_test.shape}")
    
    return X_train, X_test, y_train, y_test

def train_xgboost_model(X_train, y_train, n_estimators=100, random_state=42):
    """创建和训练XGBoost模型"""
    print("\n正在创建和训练XGBoost模型...")
    
    # 创建XGBoost回归模型
    xgb_model = xgb.XGBRegressor(
        n_estimators=n_estimators,
        random_state=random_state,
        learning_rate=0.1,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8
    )
    
    # 训练模型
    xgb_model.fit(X_train, y_train)
    
    print("XGBoost模型训练完成!")
    return xgb_model

def make_predictions(model, X_test):
    """使用模型进行预测"""
    print("\n正在进行预测...")
    
    y_pred = model.predict(X_test)
    
    print(f"预测完成! 预测值数量: {len(y_pred)}")
    print(f"前5个预测值: {y_pred[:5]}")
    
    return y_pred

def evaluate_model(y_test, y_pred):
    """评估模型性能"""
    print("\n正在评估模型性能...")
    
    # 计算评估指标
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    # 输出评估结果
    print(f"XGBoost - MSE (均方误差): {mse:.6f}")
    print(f"XGBoost - RMSE (均方根误差): {rmse:.6f}")
    print(f"XGBoost - R² (决定系数): {r2:.6f}")
    
    return mse, rmse, r2

def plot_feature_importance(model, feature_names, save_path=None):
    """绘制特征重要性图"""
    print("\n正在分析特征重要性...")
    
    # 获取特征重要性
    feature_importances = model.feature_importances_
    
    # 创建特征重要性图
    plt.figure(figsize=(10, 6))
    bars = plt.bar(feature_names, feature_importances, color='skyblue', edgecolor='navy', alpha=0.7)
    
    # 添加数值标签
    for bar, importance in zip(bars, feature_importances):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                f'{importance:.4f}', ha='center', va='bottom', fontsize=10)
    
    plt.xlabel('特征', fontsize=12)
    plt.ylabel('重要性', fontsize=12)
    plt.title('XGBoost模型特征重要性分析', fontsize=14, fontweight='bold')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"特征重要性图已保存到: {save_path}")
    
    plt.show()
    
    # 输出特征重要性排序
    importance_df = pd.DataFrame({
        '特征': feature_names,
        '重要性': feature_importances
    }).sort_values('重要性', ascending=False)
    
    print("\n特征重要性排序:")
    print(importance_df)
    
    return importance_df

def plot_predictions(y_test, y_pred, save_path=None):
    """绘制预测结果对比图"""
    print("\n正在绘制预测结果对比图...")
    
    plt.figure(figsize=(12, 5))
    
    # 子图1: 实际值 vs 预测值散点图
    plt.subplot(1, 2, 1)
    plt.scatter(y_test, y_pred, alpha=0.6, color='blue')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('实际值', fontsize=12)
    plt.ylabel('预测值', fontsize=12)
    plt.title('实际值 vs 预测值', fontsize=12, fontweight='bold')
    plt.grid(alpha=0.3)
    
    # 子图2: 残差图
    plt.subplot(1, 2, 2)
    residuals = y_test.values.flatten() - y_pred
    plt.scatter(y_pred, residuals, alpha=0.6, color='green')
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('预测值', fontsize=12)
    plt.ylabel('残差', fontsize=12)
    plt.title('残差图', fontsize=12, fontweight='bold')
    plt.grid(alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"预测结果对比图已保存到: {save_path}")
    
    plt.show()

def save_results(mse, rmse, r2, importance_df, output_path):
    """保存结果到文件"""
    print(f"\n正在保存结果到: {output_path}")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("XGBoost回归模型结果\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("模型评估指标:\n")
        f.write(f"MSE (均方误差): {mse:.6f}\n")
        f.write(f"RMSE (均方根误差): {rmse:.6f}\n")
        f.write(f"R² (决定系数): {r2:.6f}\n\n")
        
        f.write("特征重要性分析:\n")
        f.write(importance_df.to_string(index=False))
        f.write("\n\n")
        
        f.write("结果解释:\n")
        f.write("- MSE (均方误差): 衡量预测值与真实值之间的误差，误差越小，模型预测越准确\n")
        f.write("- RMSE (均方根误差): 是MSE的平方根，越小表示误差越小\n")
        f.write("- R²值: 评估模型的拟合能力，值越接近1说明模型越好\n")
        f.write("- 特征重要性: 通过XGBoost的特征重要性分析，了解哪些特征对Y染色体浓度的预测贡献最大\n")
    
    print("结果保存完成!")

def main():
    """主函数"""
    print("XGBoost回归模型 - Y染色体浓度预测")
    print("=" * 50)
    
    try:
        # 1. 加载数据
        X, y = load_data()
        
        # 2. 数据拆分
        X_train, X_test, y_train, y_test = split_data(X, y)
        
        # 3. 训练模型
        model = train_xgboost_model(X_train, y_train)
        
        # 4. 预测
        y_pred = make_predictions(model, X_test)
        
        # 5. 评估模型
        mse, rmse, r2 = evaluate_model(y_test, y_pred)
        
        # 6. 特征重要性分析
        importance_df = plot_feature_importance(model, X.columns, 'xgboost_feature_importance.png')
        
        # 7. 绘制预测结果
        plot_predictions(y_test, y_pred, 'xgboost_predictions.png')
        
        # 8. 保存结果
        save_results(mse, rmse, r2, importance_df, 'xgboost_results.txt')
        
        print("\n" + "=" * 50)
        print("XGBoost模型训练和评估完成!")
        print("生成的文件:")
        print("- xgboost_feature_importance.png: 特征重要性图")
        print("- xgboost_predictions.png: 预测结果对比图")
        print("- xgboost_results.txt: 详细结果报告")
        
    except Exception as e:
        print(f"程序执行出错: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
