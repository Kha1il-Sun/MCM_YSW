#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
特征选择分析
使用多种特征选择方法优化模型
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFE, SelectKBest, f_regression, mutual_info_regression, SelectFromModel
from sklearn.ensemble import RandomForestRegressor
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def load_and_prepare_data():
    """加载和准备数据"""
    print("=" * 60)
    print("加载和准备数据")
    print("=" * 60)
    
    df = pd.read_csv('../step1/normalized_merged_data.csv')
    X_original = df[['孕周数', 'BMI']].copy()
    y = df['Y染色体浓度'].copy()
    
    # 创建增强特征
    X_enhanced = X_original.copy()
    X_enhanced['孕周数_BMI'] = X_enhanced['孕周数'] * X_enhanced['BMI']
    X_enhanced['孕周数_平方'] = X_enhanced['孕周数'] ** 2
    X_enhanced['BMI_平方'] = X_enhanced['BMI'] ** 2
    X_enhanced['孕周数_BMI_平方'] = X_enhanced['孕周数'] * (X_enhanced['BMI'] ** 2)
    X_enhanced['孕周数_平方_BMI'] = (X_enhanced['孕周数'] ** 2) * X_enhanced['BMI']
    X_enhanced['孕周数_BMI_比值'] = X_enhanced['孕周数'] / (X_enhanced['BMI'] + 1e-8)
    X_enhanced['孕周数_BMI_和'] = X_enhanced['孕周数'] + X_enhanced['BMI']
    X_enhanced['孕周数_BMI_差'] = X_enhanced['孕周数'] - X_enhanced['BMI']
    
    print(f"原始特征数: {X_original.shape[1]}")
    print(f"增强特征数: {X_enhanced.shape[1]}")
    print(f"样本数: {len(y)}")
    
    return X_original, X_enhanced, y

def apply_feature_selection_methods(X, y):
    """应用多种特征选择方法"""
    print("\n" + "=" * 60)
    print("特征选择方法比较")
    print("=" * 60)
    
    # 分割数据
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    methods = {}
    
    # 1. 递归特征消除 (RFE)
    print("\n1. 递归特征消除 (RFE)")
    for n_features in [1, 2, 3, 5]:
        rfe = RFE(LinearRegression(), n_features_to_select=n_features)
        X_train_rfe = rfe.fit_transform(X_train_scaled, y_train)
        X_test_rfe = rfe.transform(X_test_scaled)
        
        model = LinearRegression()
        model.fit(X_train_rfe, y_train)
        
        train_r2 = r2_score(y_train, model.predict(X_train_rfe))
        test_r2 = r2_score(y_test, model.predict(X_test_rfe))
        cv_scores = cross_val_score(model, X_train_rfe, y_train, cv=5, scoring='r2')
        
        selected_features = X.columns[rfe.support_]
        methods[f'RFE_{n_features}'] = {
            'selected_features': selected_features,
            'train_r2': train_r2,
            'test_r2': test_r2,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'n_features': n_features
        }
        
        print(f"  {n_features}个特征: {list(selected_features)}")
        print(f"  测试集R²: {test_r2:.6f}, 交叉验证R²: {cv_scores.mean():.6f} ± {cv_scores.std():.6f}")
    
    # 2. 基于统计检验的特征选择
    print("\n2. 基于统计检验的特征选择 (SelectKBest)")
    for k in [1, 2, 3, 5]:
        selector = SelectKBest(score_func=f_regression, k=k)
        X_train_kbest = selector.fit_transform(X_train_scaled, y_train)
        X_test_kbest = selector.transform(X_test_scaled)
        
        model = LinearRegression()
        model.fit(X_train_kbest, y_train)
        
        train_r2 = r2_score(y_train, model.predict(X_train_kbest))
        test_r2 = r2_score(y_test, model.predict(X_test_kbest))
        cv_scores = cross_val_score(model, X_train_kbest, y_train, cv=5, scoring='r2')
        
        selected_features = X.columns[selector.get_support()]
        methods[f'KBest_{k}'] = {
            'selected_features': selected_features,
            'train_r2': train_r2,
            'test_r2': test_r2,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'n_features': k,
            'scores': selector.scores_
        }
        
        print(f"  {k}个特征: {list(selected_features)}")
        print(f"  测试集R²: {test_r2:.6f}, 交叉验证R²: {cv_scores.mean():.6f} ± {cv_scores.std():.6f}")
    
    # 3. 基于互信息的特征选择
    print("\n3. 基于互信息的特征选择 (Mutual Info)")
    for k in [1, 2, 3, 5]:
        selector = SelectKBest(score_func=mutual_info_regression, k=k)
        X_train_mi = selector.fit_transform(X_train_scaled, y_train)
        X_test_mi = selector.transform(X_test_scaled)
        
        model = LinearRegression()
        model.fit(X_train_mi, y_train)
        
        train_r2 = r2_score(y_train, model.predict(X_train_mi))
        test_r2 = r2_score(y_test, model.predict(X_test_mi))
        cv_scores = cross_val_score(model, X_train_mi, y_train, cv=5, scoring='r2')
        
        selected_features = X.columns[selector.get_support()]
        methods[f'MutualInfo_{k}'] = {
            'selected_features': selected_features,
            'train_r2': train_r2,
            'test_r2': test_r2,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'n_features': k,
            'scores': selector.scores_
        }
        
        print(f"  {k}个特征: {list(selected_features)}")
        print(f"  测试集R²: {test_r2:.6f}, 交叉验证R²: {cv_scores.mean():.6f} ± {cv_scores.std():.6f}")
    
    # 4. 基于Lasso的特征选择
    print("\n4. 基于Lasso的特征选择")
    for alpha in [0.001, 0.01, 0.1, 1.0]:
        lasso = Lasso(alpha=alpha, random_state=42)
        lasso.fit(X_train_scaled, y_train)
        
        # 选择非零系数的特征
        selected_mask = np.abs(lasso.coef_) > 1e-6
        selected_features = X.columns[selected_mask]
        
        if len(selected_features) > 0:
            X_train_lasso = X_train_scaled[:, selected_mask]
            X_test_lasso = X_test_scaled[:, selected_mask]
            
            model = LinearRegression()
            model.fit(X_train_lasso, y_train)
            
            train_r2 = r2_score(y_train, model.predict(X_train_lasso))
            test_r2 = r2_score(y_test, model.predict(X_test_lasso))
            cv_scores = cross_val_score(model, X_train_lasso, y_train, cv=5, scoring='r2')
            
            methods[f'Lasso_{alpha}'] = {
                'selected_features': selected_features,
                'train_r2': train_r2,
                'test_r2': test_r2,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'n_features': len(selected_features),
                'alpha': alpha
            }
            
            print(f"  α={alpha}: {list(selected_features)} ({len(selected_features)}个特征)")
            print(f"  测试集R²: {test_r2:.6f}, 交叉验证R²: {cv_scores.mean():.6f} ± {cv_scores.std():.6f}")
        else:
            print(f"  α={alpha}: 没有选择任何特征")
    
    # 5. 基于随机森林的特征选择
    print("\n5. 基于随机森林的特征选择")
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train_scaled, y_train)
    
    # 使用特征重要性进行选择
    feature_importance = rf.feature_importances_
    importance_thresholds = [0.01, 0.05, 0.1, 0.15]
    
    for threshold in importance_thresholds:
        selected_mask = feature_importance > threshold
        selected_features = X.columns[selected_mask]
        
        if len(selected_features) > 0:
            X_train_rf = X_train_scaled[:, selected_mask]
            X_test_rf = X_test_scaled[:, selected_mask]
            
            model = LinearRegression()
            model.fit(X_train_rf, y_train)
            
            train_r2 = r2_score(y_train, model.predict(X_train_rf))
            test_r2 = r2_score(y_test, model.predict(X_test_rf))
            cv_scores = cross_val_score(model, X_train_rf, y_train, cv=5, scoring='r2')
            
            methods[f'RandomForest_{threshold}'] = {
                'selected_features': selected_features,
                'train_r2': train_r2,
                'test_r2': test_r2,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'n_features': len(selected_features),
                'threshold': threshold,
                'importance': feature_importance[selected_mask]
            }
            
            print(f"  阈值={threshold}: {list(selected_features)} ({len(selected_features)}个特征)")
            print(f"  测试集R²: {test_r2:.6f}, 交叉验证R²: {cv_scores.mean():.6f} ± {cv_scores.std():.6f}")
        else:
            print(f"  阈值={threshold}: 没有选择任何特征")
    
    return methods, X_train_scaled, X_test_scaled, y_train, y_test, scaler

def visualize_feature_selection_results(methods, X, y):
    """可视化特征选择结果"""
    print("\n" + "=" * 60)
    print("结果可视化")
    print("=" * 60)
    
    # 创建图形
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('特征选择方法比较', fontsize=16, fontweight='bold')
    
    # 1. 方法性能比较
    method_names = list(methods.keys())
    test_r2_scores = [methods[name]['test_r2'] for name in method_names]
    cv_means = [methods[name]['cv_mean'] for name in method_names]
    cv_stds = [methods[name]['cv_std'] for name in method_names]
    n_features = [methods[name]['n_features'] for name in method_names]
    
    # 按测试集R²排序
    sorted_indices = np.argsort(test_r2_scores)[::-1]
    
    axes[0, 0].bar(range(len(method_names)), [test_r2_scores[i] for i in sorted_indices], 
                   alpha=0.7, color='skyblue')
    axes[0, 0].set_xlabel('特征选择方法')
    axes[0, 0].set_ylabel('测试集 R²')
    axes[0, 0].set_title('各方法测试集性能比较')
    axes[0, 0].set_xticks(range(len(method_names)))
    axes[0, 0].set_xticklabels([method_names[i] for i in sorted_indices], rotation=45, ha='right')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. 交叉验证性能比较
    axes[0, 1].bar(range(len(method_names)), [cv_means[i] for i in sorted_indices], 
                   yerr=[cv_stds[i] for i in sorted_indices], capsize=5, alpha=0.7, color='lightcoral')
    axes[0, 1].set_xlabel('特征选择方法')
    axes[0, 1].set_ylabel('交叉验证 R²')
    axes[0, 1].set_title('各方法交叉验证性能比较')
    axes[0, 1].set_xticks(range(len(method_names)))
    axes[0, 1].set_xticklabels([method_names[i] for i in sorted_indices], rotation=45, ha='right')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. 特征数量 vs 性能
    axes[1, 0].scatter(n_features, test_r2_scores, alpha=0.7, s=100, color='green')
    for i, name in enumerate(method_names):
        axes[1, 0].annotate(name, (n_features[i], test_r2_scores[i]), 
                           xytext=(5, 5), textcoords='offset points', fontsize=8)
    axes[1, 0].set_xlabel('选择的特征数量')
    axes[1, 0].set_ylabel('测试集 R²')
    axes[1, 0].set_title('特征数量 vs 性能')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. 最佳方法特征重要性
    best_method = max(methods.keys(), key=lambda x: methods[x]['test_r2'])
    best_features = methods[best_method]['selected_features']
    
    if len(best_features) > 0:
        # 获取特征重要性（如果有的话）
        if 'importance' in methods[best_method]:
            importance_values = methods[best_method]['importance']
        else:
            # 使用随机森林计算特征重要性
            rf = RandomForestRegressor(n_estimators=100, random_state=42)
            rf.fit(X[best_features], y)
            importance_values = rf.feature_importances_
        
        bars = axes[1, 1].barh(range(len(best_features)), importance_values, color='orange')
        axes[1, 1].set_yticks(range(len(best_features)))
        axes[1, 1].set_yticklabels(best_features)
        axes[1, 1].set_xlabel('特征重要性')
        axes[1, 1].set_title(f'最佳方法特征重要性 ({best_method})')
        axes[1, 1].grid(True, alpha=0.3)
        
        # 添加数值标签
        for i, (bar, importance) in enumerate(zip(bars, importance_values)):
            axes[1, 1].text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
                           f'{importance:.4f}', ha='left', va='center')
    else:
        axes[1, 1].text(0.5, 0.5, '没有选择任何特征', ha='center', va='center', 
                       transform=axes[1, 1].transAxes)
        axes[1, 1].set_title(f'最佳方法特征重要性 ({best_method})')
    
    plt.tight_layout()
    plt.savefig('feature_selection_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def save_feature_selection_results(methods):
    """保存特征选择结果"""
    print("\n" + "=" * 60)
    print("保存结果")
    print("=" * 60)
    
    with open('feature_selection_results.txt', 'w', encoding='utf-8') as f:
        f.write("特征选择分析结果\n")
        f.write("=" * 60 + "\n\n")
        
        # 按测试集R²排序
        sorted_methods = sorted(methods.items(), key=lambda x: x[1]['test_r2'], reverse=True)
        
        f.write("方法性能排名:\n")
        f.write("-" * 40 + "\n")
        for i, (name, result) in enumerate(sorted_methods, 1):
            f.write(f"{i}. {name}:\n")
            f.write(f"   选择的特征: {list(result['selected_features'])}\n")
            f.write(f"   特征数量: {result['n_features']}\n")
            f.write(f"   测试集 R²: {result['test_r2']:.6f}\n")
            f.write(f"   交叉验证 R²: {result['cv_mean']:.6f} ± {result['cv_std']:.6f}\n\n")
        
        # 最佳方法详情
        best_method_name, best_result = sorted_methods[0]
        f.write(f"最佳方法: {best_method_name}\n")
        f.write("-" * 40 + "\n")
        f.write(f"选择的特征: {list(best_result['selected_features'])}\n")
        f.write(f"特征数量: {best_result['n_features']}\n")
        f.write(f"测试集 R²: {best_result['test_r2']:.6f}\n")
        f.write(f"训练集 R²: {best_result['train_r2']:.6f}\n")
        f.write(f"交叉验证 R²: {best_result['cv_mean']:.6f} ± {best_result['cv_std']:.6f}\n")
    
    print("结果已保存到 feature_selection_results.txt")

def main():
    """主函数"""
    print("开始特征选择分析...")
    
    # 1. 加载和准备数据
    X_original, X_enhanced, y = load_and_prepare_data()
    
    # 2. 应用特征选择方法
    methods, X_train_scaled, X_test_scaled, y_train, y_test, scaler = apply_feature_selection_methods(X_enhanced, y)
    
    # 3. 可视化结果
    visualize_feature_selection_results(methods, X_enhanced, y)
    
    # 4. 保存结果
    save_feature_selection_results(methods)
    
    # 5. 输出最佳方法
    best_method = max(methods.keys(), key=lambda x: methods[x]['test_r2'])
    best_result = methods[best_method]
    
    print(f"\n最佳特征选择方法: {best_method}")
    print(f"选择的特征: {list(best_result['selected_features'])}")
    print(f"测试集 R²: {best_result['test_r2']:.6f}")
    print(f"交叉验证 R²: {best_result['cv_mean']:.6f} ± {best_result['cv_std']:.6f}")
    
    print("\n特征选择分析完成！")

if __name__ == "__main__":
    main()
