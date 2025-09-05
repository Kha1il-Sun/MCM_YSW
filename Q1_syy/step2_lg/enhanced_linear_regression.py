#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
增强特征工程的线性回归分析
包含特征交互、特征选择等方法
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.feature_selection import RFE, SelectKBest, f_regression, mutual_info_regression
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def load_data():
    """加载数据"""
    print("=" * 60)
    print("加载数据")
    print("=" * 60)
    
    df = pd.read_csv('../step1/normalized_merged_data.csv')
    print(f"数据形状: {df.shape}")
    return df

def create_interaction_features(X):
    """创建特征交互项"""
    print("\n" + "=" * 60)
    print("创建特征交互项")
    print("=" * 60)
    
    X_enhanced = X.copy()
    
    # 1. 基本交互项
    X_enhanced['孕周数_BMI'] = X_enhanced['孕周数'] * X_enhanced['BMI']
    
    # 2. 平方项
    X_enhanced['孕周数_平方'] = X_enhanced['孕周数'] ** 2
    X_enhanced['BMI_平方'] = X_enhanced['BMI'] ** 2
    
    # 3. 更多交互项
    X_enhanced['孕周数_BMI_平方'] = X_enhanced['孕周数'] * (X_enhanced['BMI'] ** 2)
    X_enhanced['孕周数_平方_BMI'] = (X_enhanced['孕周数'] ** 2) * X_enhanced['BMI']
    
    # 4. 比值特征
    X_enhanced['孕周数_BMI_比值'] = X_enhanced['孕周数'] / (X_enhanced['BMI'] + 1e-8)
    
    # 5. 组合特征
    X_enhanced['孕周数_BMI_和'] = X_enhanced['孕周数'] + X_enhanced['BMI']
    X_enhanced['孕周数_BMI_差'] = X_enhanced['孕周数'] - X_enhanced['BMI']
    
    print(f"原始特征数: {X.shape[1]}")
    print(f"增强后特征数: {X_enhanced.shape[1]}")
    print(f"新增特征: {list(X_enhanced.columns[X.shape[1]:])}")
    
    return X_enhanced

def create_polynomial_features(X, degree=2):
    """创建多项式特征"""
    print(f"\n创建{degree}次多项式特征...")
    
    poly = PolynomialFeatures(degree=degree, include_bias=False, interaction_only=False)
    X_poly = poly.fit_transform(X)
    
    # 创建特征名称
    feature_names = poly.get_feature_names_out(X.columns)
    X_poly_df = pd.DataFrame(X_poly, columns=feature_names, index=X.index)
    
    print(f"多项式特征数: {X_poly_df.shape[1]}")
    return X_poly_df

def perform_feature_selection(X, y, method='rfe', n_features=None):
    """执行特征选择"""
    print(f"\n" + "=" * 60)
    print(f"特征选择 - {method.upper()}")
    print("=" * 60)
    
    if method == 'rfe':
        # 递归特征消除
        if n_features is None:
            n_features = min(5, X.shape[1])
        
        selector = RFE(LinearRegression(), n_features_to_select=n_features)
        X_selected = selector.fit_transform(X, y)
        selected_features = X.columns[selector.support_]
        feature_scores = selector.ranking_
        
    elif method == 'kbest':
        # 基于统计检验的特征选择
        if n_features is None:
            n_features = min(5, X.shape[1])
        
        selector = SelectKBest(score_func=f_regression, k=n_features)
        X_selected = selector.fit_transform(X, y)
        selected_features = X.columns[selector.get_support()]
        feature_scores = selector.scores_
        
    elif method == 'mutual_info':
        # 基于互信息的特征选择
        if n_features is None:
            n_features = min(5, X.shape[1])
        
        selector = SelectKBest(score_func=mutual_info_regression, k=n_features)
        X_selected = selector.fit_transform(X, y)
        selected_features = X.columns[selector.get_support()]
        feature_scores = selector.scores_
    
    print(f"选择的特征数: {len(selected_features)}")
    print(f"选择的特征: {list(selected_features)}")
    
    return X_selected, selected_features, feature_scores

def compare_models(X_original, X_enhanced, X_poly, y):
    """比较不同特征集的模型性能"""
    print("\n" + "=" * 60)
    print("模型性能比较")
    print("=" * 60)
    
    # 分割数据
    X_train_orig, X_test_orig, y_train, y_test = train_test_split(
        X_original, y, test_size=0.2, random_state=42
    )
    X_train_enh, X_test_enh, _, _ = train_test_split(
        X_enhanced, y, test_size=0.2, random_state=42
    )
    X_train_poly, X_test_poly, _, _ = train_test_split(
        X_poly, y, test_size=0.2, random_state=42
    )
    
    models = {
        '原始特征': (X_train_orig, X_test_orig),
        '增强特征': (X_train_enh, X_test_enh),
        '多项式特征': (X_train_poly, X_test_poly)
    }
    
    results = {}
    
    for name, (X_train, X_test) in models.items():
        print(f"\n{name}模型:")
        
        # 标准化
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # 训练模型
        model = LinearRegression()
        model.fit(X_train_scaled, y_train)
        
        # 预测
        y_train_pred = model.predict(X_train_scaled)
        y_test_pred = model.predict(X_test_scaled)
        
        # 计算指标
        train_r2 = r2_score(y_train, y_train_pred)
        test_r2 = r2_score(y_test, y_test_pred)
        train_mse = mean_squared_error(y_train, y_train_pred)
        test_mse = mean_squared_error(y_test, y_test_pred)
        train_mae = mean_absolute_error(y_train, y_train_pred)
        test_mae = mean_absolute_error(y_test, y_test_pred)
        
        # 交叉验证
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='r2')
        
        results[name] = {
            'train_r2': train_r2,
            'test_r2': test_r2,
            'train_mse': train_mse,
            'test_mse': test_mse,
            'train_mae': train_mae,
            'test_mae': test_mae,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'model': model,
            'scaler': scaler,
            'X_train': X_train,
            'X_test': X_test,
            'y_train_pred': y_train_pred,
            'y_test_pred': y_test_pred
        }
        
        print(f"  训练集 R²: {train_r2:.6f}")
        print(f"  测试集 R²: {test_r2:.6f}")
        print(f"  训练集 MSE: {train_mse:.6f}")
        print(f"  测试集 MSE: {test_mse:.6f}")
        print(f"  交叉验证 R²: {cv_scores.mean():.6f} ± {cv_scores.std():.6f}")
    
    return results

def feature_importance_analysis(X_enhanced, y):
    """特征重要性分析"""
    print("\n" + "=" * 60)
    print("特征重要性分析")
    print("=" * 60)
    
    # 分割数据
    X_train, X_test, y_train, y_test = train_test_split(
        X_enhanced, y, test_size=0.2, random_state=42
    )
    
    # 标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 训练模型
    model = LinearRegression()
    model.fit(X_train_scaled, y_train)
    
    # 特征重要性（系数绝对值）
    feature_importance = np.abs(model.coef_)
    feature_names = X_enhanced.columns
    
    # 排序
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': feature_importance,
        'coefficient': model.coef_
    }).sort_values('importance', ascending=False)
    
    print("特征重要性排序:")
    for idx, row in importance_df.iterrows():
        print(f"  {row['feature']}: {row['importance']:.6f} (系数: {row['coefficient']:.6f})")
    
    return importance_df, model, scaler

def visualize_results(results, importance_df):
    """可视化结果"""
    print("\n" + "=" * 60)
    print("结果可视化")
    print("=" * 60)
    
    # 创建图形
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle('增强特征工程 - 线性回归分析结果', fontsize=16, fontweight='bold')
    
    # 1. 模型性能比较
    model_names = list(results.keys())
    train_r2_scores = [results[name]['train_r2'] for name in model_names]
    test_r2_scores = [results[name]['test_r2'] for name in model_names]
    
    x = np.arange(len(model_names))
    width = 0.35
    
    axes[0, 0].bar(x - width/2, train_r2_scores, width, label='训练集', alpha=0.8)
    axes[0, 0].bar(x + width/2, test_r2_scores, width, label='测试集', alpha=0.8)
    axes[0, 0].set_xlabel('模型类型')
    axes[0, 0].set_ylabel('R² 分数')
    axes[0, 0].set_title('模型性能比较')
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(model_names, rotation=45)
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. 特征重要性
    top_features = importance_df.head(10)
    bars = axes[0, 1].barh(range(len(top_features)), top_features['importance'], color='skyblue')
    axes[0, 1].set_yticks(range(len(top_features)))
    axes[0, 1].set_yticklabels(top_features['feature'])
    axes[0, 1].set_xlabel('特征重要性 (系数绝对值)')
    axes[0, 1].set_title('特征重要性排序 (前10)')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 添加数值标签
    for i, (bar, importance) in enumerate(zip(bars, top_features['importance'])):
        axes[0, 1].text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
                       f'{importance:.4f}', ha='left', va='center')
    
    # 3. 最佳模型的实际值 vs 预测值
    best_model_name = max(results.keys(), key=lambda x: results[x]['test_r2'])
    best_result = results[best_model_name]
    
    axes[0, 2].scatter(best_result['X_test']['孕周数'], best_result['y_test_pred'], 
                      alpha=0.6, label='预测值', color='blue')
    axes[0, 2].scatter(best_result['X_test']['孕周数'], 
                      best_result['X_test'].iloc[:, 1] if best_result['X_test'].shape[1] > 1 else best_result['X_test']['BMI'], 
                      alpha=0.6, label='实际值', color='red')
    axes[0, 2].set_xlabel('孕周数')
    axes[0, 2].set_ylabel('Y染色体浓度')
    axes[0, 2].set_title(f'最佳模型预测 ({best_model_name})')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    # 4. 残差分析
    residuals = best_result['X_test'].iloc[:, 1] if best_result['X_test'].shape[1] > 1 else best_result['X_test']['BMI'] - best_result['y_test_pred']
    axes[1, 0].scatter(best_result['y_test_pred'], residuals, alpha=0.6)
    axes[1, 0].axhline(y=0, color='r', linestyle='--')
    axes[1, 0].set_xlabel('预测值')
    axes[1, 0].set_ylabel('残差')
    axes[1, 0].set_title('残差分析')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 5. 交叉验证分数比较
    cv_means = [results[name]['cv_mean'] for name in model_names]
    cv_stds = [results[name]['cv_std'] for name in model_names]
    
    axes[1, 1].bar(model_names, cv_means, yerr=cv_stds, capsize=5, alpha=0.8, color='lightgreen')
    axes[1, 1].set_xlabel('模型类型')
    axes[1, 1].set_ylabel('交叉验证 R² 分数')
    axes[1, 1].set_title('交叉验证性能比较')
    axes[1, 1].tick_params(axis='x', rotation=45)
    axes[1, 1].grid(True, alpha=0.3)
    
    # 6. 特征系数热图
    if len(importance_df) <= 15:  # 只显示前15个特征
        coef_matrix = importance_df.head(15)[['feature', 'coefficient']].set_index('feature')
        sns.heatmap(coef_matrix.T, annot=True, cmap='RdBu_r', center=0, 
                   ax=axes[1, 2], cbar_kws={'label': '系数值'})
        axes[1, 2].set_title('特征系数热图')
        axes[1, 2].set_xlabel('特征')
    else:
        axes[1, 2].text(0.5, 0.5, '特征过多\n无法显示热图', 
                       ha='center', va='center', transform=axes[1, 2].transAxes)
        axes[1, 2].set_title('特征系数热图')
    
    plt.tight_layout()
    plt.savefig('enhanced_linear_regression_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def save_enhanced_results(results, importance_df):
    """保存增强结果"""
    print("\n" + "=" * 60)
    print("保存结果")
    print("=" * 60)
    
    # 保存详细结果
    with open('enhanced_linear_regression_results.txt', 'w', encoding='utf-8') as f:
        f.write("增强特征工程 - 线性回归分析结果\n")
        f.write("=" * 60 + "\n\n")
        
        f.write("模型性能比较:\n")
        f.write("-" * 40 + "\n")
        for name, result in results.items():
            f.write(f"\n{name}:\n")
            f.write(f"  训练集 R²: {result['train_r2']:.6f}\n")
            f.write(f"  测试集 R²: {result['test_r2']:.6f}\n")
            f.write(f"  训练集 MSE: {result['train_mse']:.6f}\n")
            f.write(f"  测试集 MSE: {result['test_mse']:.6f}\n")
            f.write(f"  交叉验证 R²: {result['cv_mean']:.6f} ± {result['cv_std']:.6f}\n")
        
        f.write(f"\n\n特征重要性排序:\n")
        f.write("-" * 40 + "\n")
        for idx, row in importance_df.iterrows():
            f.write(f"{row['feature']}: {row['importance']:.6f} (系数: {row['coefficient']:.6f})\n")
    
    print("结果已保存到 enhanced_linear_regression_results.txt")

def main():
    """主函数"""
    print("开始增强特征工程的线性回归分析...")
    
    # 1. 加载数据
    df = load_data()
    X_original = df[['孕周数', 'BMI']].copy()
    y = df['Y染色体浓度'].copy()
    
    # 2. 创建增强特征
    X_enhanced = create_interaction_features(X_original)
    
    # 3. 创建多项式特征
    X_poly = create_polynomial_features(X_original, degree=2)
    
    # 4. 比较模型性能
    results = compare_models(X_original, X_enhanced, X_poly, y)
    
    # 5. 特征重要性分析
    importance_df, model, scaler = feature_importance_analysis(X_enhanced, y)
    
    # 6. 可视化结果
    visualize_results(results, importance_df)
    
    # 7. 保存结果
    save_enhanced_results(results, importance_df)
    
    print("\n增强特征工程分析完成！")
    
    # 输出最佳模型信息
    best_model_name = max(results.keys(), key=lambda x: results[x]['test_r2'])
    best_result = results[best_model_name]
    print(f"\n最佳模型: {best_model_name}")
    print(f"测试集 R²: {best_result['test_r2']:.6f}")
    print(f"交叉验证 R²: {best_result['cv_mean']:.6f} ± {best_result['cv_std']:.6f}")

if __name__ == "__main__":
    main()
