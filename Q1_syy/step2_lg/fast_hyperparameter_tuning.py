#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快速超参数调优分析
优化版本，减少计算时间
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
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

def create_optimized_models():
    """创建优化的模型"""
    print("\n" + "=" * 60)
    print("创建优化模型")
    print("=" * 60)
    
    models = {
        'LinearRegression': LinearRegression(),
        'Ridge': Ridge(random_state=42),
        'Lasso': Lasso(random_state=42, max_iter=2000),
        'ElasticNet': ElasticNet(random_state=42, max_iter=2000),
        'RandomForest': RandomForestRegressor(random_state=42, n_estimators=100),
        'SVR': SVR()
    }
    
    print(f"创建了 {len(models)} 个模型")
    for name in models.keys():
        print(f"  - {name}")
    
    return models

def define_parameter_grids():
    """定义参数网格（简化版）"""
    print("\n" + "=" * 60)
    print("定义参数网格")
    print("=" * 60)
    
    param_grids = {
        'Ridge': {
            'alpha': [0.001, 0.01, 0.1, 1.0, 10.0]
        },
        'Lasso': {
            'alpha': [0.001, 0.01, 0.1, 1.0, 10.0]
        },
        'ElasticNet': {
            'alpha': [0.001, 0.01, 0.1, 1.0],
            'l1_ratio': [0.1, 0.5, 0.9]
        },
        'RandomForest': {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5]
        },
        'SVR': {
            'C': [0.1, 1, 10, 100],
            'gamma': ['scale', 'auto', 0.01, 0.1],
            'kernel': ['rbf', 'linear']
        }
    }
    
    print("参数网格定义完成:")
    for model, params in param_grids.items():
        print(f"  {model}: {len(params)} 个参数")
    
    return param_grids

def perform_grid_search(models, param_grids, X, y):
    """执行网格搜索"""
    print("\n" + "=" * 60)
    print("执行网格搜索")
    print("=" * 60)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    grid_results = {}
    
    for model_name, model in models.items():
        print(f"\n{model_name}: 开始调优...")
        
        if model_name in param_grids:
            # 网格搜索
            grid_search = GridSearchCV(
                model, 
                param_grids[model_name], 
                cv=3,  # 减少CV折数
                scoring='r2',
                n_jobs=1,
                verbose=0
            )
            
            grid_search.fit(X_train_scaled, y_train)
            
            # 预测
            y_train_pred = grid_search.predict(X_train_scaled)
            y_test_pred = grid_search.predict(X_test_scaled)
            
            # 计算指标
            train_r2 = r2_score(y_train, y_train_pred)
            test_r2 = r2_score(y_test, y_test_pred)
            train_mse = mean_squared_error(y_train, y_train_pred)
            test_mse = mean_squared_error(y_test, y_test_pred)
            
            grid_results[model_name] = {
                'best_params': grid_search.best_params_,
                'best_score': grid_search.best_score_,
                'train_r2': train_r2,
                'test_r2': test_r2,
                'train_mse': train_mse,
                'test_mse': test_mse,
                'model': grid_search.best_estimator_
            }
            
            print(f"  最佳参数: {grid_search.best_params_}")
            print(f"  最佳CV分数: {grid_search.best_score_:.6f}")
            print(f"  测试集R²: {test_r2:.6f}")
        else:
            # 无参数调优
            model.fit(X_train_scaled, y_train)
            y_train_pred = model.predict(X_train_scaled)
            y_test_pred = model.predict(X_test_scaled)
            
            train_r2 = r2_score(y_train, y_train_pred)
            test_r2 = r2_score(y_test, y_test_pred)
            train_mse = mean_squared_error(y_train, y_train_pred)
            test_mse = mean_squared_error(y_test, y_test_pred)
            
            grid_results[model_name] = {
                'best_params': {},
                'best_score': cross_val_score(model, X_train_scaled, y_train, cv=3, scoring='r2').mean(),
                'train_r2': train_r2,
                'test_r2': test_r2,
                'train_mse': train_mse,
                'test_mse': test_mse,
                'model': model
            }
            
            print(f"  测试集R²: {test_r2:.6f}")
    
    return grid_results, scaler

def perform_random_search(models, param_grids, X, y, n_iter=20):
    """执行随机搜索"""
    print("\n" + "=" * 60)
    print("执行随机搜索")
    print("=" * 60)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    random_results = {}
    
    for model_name, model in models.items():
        if model_name not in param_grids:
            continue
            
        print(f"\n{model_name}: 开始随机搜索...")
        
        # 随机搜索
        random_search = RandomizedSearchCV(
            model, 
            param_grids[model_name], 
            n_iter=n_iter,
            cv=3,
            scoring='r2',
            n_jobs=1,
            random_state=42,
            verbose=0
        )
        
        random_search.fit(X_train_scaled, y_train)
        
        # 预测
        y_train_pred = random_search.predict(X_train_scaled)
        y_test_pred = random_search.predict(X_test_scaled)
        
        # 计算指标
        train_r2 = r2_score(y_train, y_train_pred)
        test_r2 = r2_score(y_test, y_test_pred)
        train_mse = mean_squared_error(y_train, y_train_pred)
        test_mse = mean_squared_error(y_test, y_test_pred)
        
        random_results[model_name] = {
            'best_params': random_search.best_params_,
            'best_score': random_search.best_score_,
            'train_r2': train_r2,
            'test_r2': test_r2,
            'train_mse': train_mse,
            'test_mse': test_mse,
            'model': random_search.best_estimator_
        }
        
        print(f"  最佳参数: {random_search.best_params_}")
        print(f"  最佳CV分数: {random_search.best_score_:.6f}")
        print(f"  测试集R²: {test_r2:.6f}")
    
    return random_results

def test_polynomial_features(X, y):
    """测试多项式特征"""
    print("\n" + "=" * 60)
    print("测试多项式特征")
    print("=" * 60)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 创建多项式特征
    poly = PolynomialFeatures(degree=2, include_bias=False)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)
    
    # 标准化
    scaler = StandardScaler()
    X_train_poly_scaled = scaler.fit_transform(X_train_poly)
    X_test_poly_scaled = scaler.transform(X_test_poly)
    
    poly_results = {}
    
    # 测试不同模型
    models = {
        'Ridge_Poly': Ridge(random_state=42),
        'Lasso_Poly': Lasso(random_state=42, max_iter=2000),
        'ElasticNet_Poly': ElasticNet(random_state=42, max_iter=2000)
    }
    
    param_grids = {
        'Ridge_Poly': {'alpha': [0.001, 0.01, 0.1, 1.0, 10.0]},
        'Lasso_Poly': {'alpha': [0.001, 0.01, 0.1, 1.0, 10.0]},
        'ElasticNet_Poly': {'alpha': [0.001, 0.01, 0.1, 1.0], 'l1_ratio': [0.1, 0.5, 0.9]}
    }
    
    for model_name, model in models.items():
        print(f"\n{model_name}: 开始调优...")
        
        grid_search = GridSearchCV(
            model, 
            param_grids[model_name], 
            cv=3,
            scoring='r2',
            n_jobs=1,
            verbose=0
        )
        
        grid_search.fit(X_train_poly_scaled, y_train)
        
        # 预测
        y_train_pred = grid_search.predict(X_train_poly_scaled)
        y_test_pred = grid_search.predict(X_test_poly_scaled)
        
        # 计算指标
        train_r2 = r2_score(y_train, y_train_pred)
        test_r2 = r2_score(y_test, y_test_pred)
        train_mse = mean_squared_error(y_train, y_train_pred)
        test_mse = mean_squared_error(y_test, y_test_pred)
        
        poly_results[model_name] = {
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_,
            'train_r2': train_r2,
            'test_r2': test_r2,
            'train_mse': train_mse,
            'test_mse': test_mse,
            'model': grid_search.best_estimator_
        }
        
        print(f"  最佳参数: {grid_search.best_params_}")
        print(f"  最佳CV分数: {grid_search.best_score_:.6f}")
        print(f"  测试集R²: {test_r2:.6f}")
    
    return poly_results

def visualize_tuning_results(grid_results, random_results, poly_results):
    """可视化调优结果"""
    print("\n" + "=" * 60)
    print("结果可视化")
    print("=" * 60)
    
    # 合并所有结果
    all_results = {}
    for method, results in [('Grid', grid_results), ('Random', random_results), ('Poly', poly_results)]:
        for model, result in results.items():
            key = f"{method}_{model}"
            all_results[key] = result
    
    if not all_results:
        print("没有可可视化的结果")
        return
    
    # 创建图形
    fig, axes = plt.subplots(2, 2, figsize=(20, 12))
    fig.suptitle('超参数调优结果分析', fontsize=16, fontweight='bold')
    
    # 1. 模型性能比较
    model_names = list(all_results.keys())
    test_r2_scores = [all_results[name]['test_r2'] for name in model_names]
    cv_scores = [all_results[name]['best_score'] for name in model_names]
    
    x = np.arange(len(model_names))
    width = 0.35
    
    bars1 = axes[0, 0].bar(x - width/2, test_r2_scores, width, label='测试集R²', alpha=0.8, color='skyblue')
    bars2 = axes[0, 0].bar(x + width/2, cv_scores, width, label='交叉验证R²', alpha=0.8, color='lightcoral')
    
    axes[0, 0].set_xlabel('模型')
    axes[0, 0].set_ylabel('R² 分数')
    axes[0, 0].set_title('模型性能比较')
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(model_names, rotation=45, ha='right')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 添加数值标签
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            axes[0, 0].text(bar.get_x() + bar.get_width()/2., height + 0.001,
                           f'{height:.3f}', ha='center', va='bottom', fontsize=8)
    
    # 2. 调优方法比较
    methods = ['Grid', 'Random', 'Poly']
    method_scores = {}
    
    for method in methods:
        method_models = [name for name in model_names if name.startswith(method)]
        if method_models:
            scores = [all_results[name]['test_r2'] for name in method_models]
            method_scores[method] = np.mean(scores)
    
    if method_scores:
        bars = axes[0, 1].bar(method_scores.keys(), method_scores.values(), 
                              alpha=0.8, color=['lightgreen', 'orange', 'purple'])
        axes[0, 1].set_xlabel('调优方法')
        axes[0, 1].set_ylabel('平均测试集R²')
        axes[0, 1].set_title('调优方法效果比较')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 添加数值标签
        for bar in bars:
            height = bar.get_height()
            axes[0, 1].text(bar.get_x() + bar.get_width()/2., height + 0.001,
                           f'{height:.3f}', ha='center', va='bottom')
    
    # 3. 最佳模型参数
    best_model = max(all_results.keys(), key=lambda x: all_results[x]['test_r2'])
    best_params = all_results[best_model]['best_params']
    
    if best_params:
        param_names = list(best_params.keys())
        param_values = [str(v) for v in best_params.values()]
        
        # 创建条形图
        y_pos = np.arange(len(param_names))
        axes[1, 0].barh(y_pos, [1] * len(param_names), alpha=0.7, color='lightblue')
        axes[1, 0].set_yticks(y_pos)
        axes[1, 0].set_yticklabels(param_names)
        axes[1, 0].set_xlabel('参数值')
        axes[1, 0].set_title(f'最佳模型参数 ({best_model})')
        
        # 添加参数值标签
        for i, (name, value) in enumerate(zip(param_names, param_values)):
            axes[1, 0].text(0.5, i, f'{name}: {value}', ha='center', va='center', fontweight='bold')
    else:
        axes[1, 0].text(0.5, 0.5, '无参数信息', ha='center', va='center', 
                       transform=axes[1, 0].transAxes)
        axes[1, 0].set_title('最佳模型参数')
    
    # 4. 性能分布
    axes[1, 1].hist(test_r2_scores, bins=8, alpha=0.7, color='lightblue', edgecolor='black')
    axes[1, 1].axvline(np.mean(test_r2_scores), color='red', linestyle='--', 
                      label=f'平均值: {np.mean(test_r2_scores):.3f}')
    axes[1, 1].axvline(np.max(test_r2_scores), color='green', linestyle='--', 
                      label=f'最大值: {np.max(test_r2_scores):.3f}')
    axes[1, 1].set_xlabel('测试集R²')
    axes[1, 1].set_ylabel('频次')
    axes[1, 1].set_title('性能分布')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('fast_hyperparameter_tuning_results.png', dpi=300, bbox_inches='tight')
    plt.show()

def save_tuning_results(grid_results, random_results, poly_results):
    """保存调优结果"""
    print("\n" + "=" * 60)
    print("保存结果")
    print("=" * 60)
    
    with open('fast_hyperparameter_tuning_results.txt', 'w', encoding='utf-8') as f:
        f.write("快速超参数调优结果\n")
        f.write("=" * 60 + "\n\n")
        
        # 合并所有结果
        all_results = {}
        for method, results in [('Grid', grid_results), ('Random', random_results), ('Poly', poly_results)]:
            for model, result in results.items():
                key = f"{method}_{model}"
                all_results[key] = result
        
        # 按测试集R²排序
        sorted_results = sorted(all_results.items(), key=lambda x: x[1]['test_r2'], reverse=True)
        
        f.write("模型性能排名:\n")
        f.write("-" * 40 + "\n")
        for i, (name, result) in enumerate(sorted_results, 1):
            f.write(f"{i}. {name}:\n")
            f.write(f"   最佳参数: {result['best_params']}\n")
            f.write(f"   交叉验证R²: {result['best_score']:.6f}\n")
            f.write(f"   测试集R²: {result['test_r2']:.6f}\n")
            f.write(f"   训练集R²: {result['train_r2']:.6f}\n")
            f.write(f"   测试集MSE: {result['test_mse']:.6f}\n\n")
        
        # 最佳模型详情
        if sorted_results:
            best_name, best_result = sorted_results[0]
            f.write(f"最佳模型: {best_name}\n")
            f.write("-" * 40 + "\n")
            f.write(f"最佳参数: {best_result['best_params']}\n")
            f.write(f"交叉验证R²: {best_result['best_score']:.6f}\n")
            f.write(f"测试集R²: {best_result['test_r2']:.6f}\n")
            f.write(f"训练集R²: {best_result['train_r2']:.6f}\n")
            f.write(f"测试集MSE: {best_result['test_mse']:.6f}\n")
    
    print("结果已保存到 fast_hyperparameter_tuning_results.txt")

def main():
    """主函数"""
    print("开始快速超参数调优分析...")
    
    # 1. 加载和准备数据
    X_original, X_enhanced, y = load_and_prepare_data()
    
    # 2. 创建模型
    models = create_optimized_models()
    
    # 3. 定义参数网格
    param_grids = define_parameter_grids()
    
    # 4. 执行网格搜索（增强特征）
    print("\n使用增强特征进行调优...")
    grid_results, scaler = perform_grid_search(models, param_grids, X_enhanced, y)
    
    # 5. 执行随机搜索
    random_results = perform_random_search(models, param_grids, X_enhanced, y, n_iter=15)
    
    # 6. 测试多项式特征
    poly_results = test_polynomial_features(X_original, y)
    
    # 7. 可视化结果
    visualize_tuning_results(grid_results, random_results, poly_results)
    
    # 8. 保存结果
    save_tuning_results(grid_results, random_results, poly_results)
    
    # 9. 输出最佳模型
    all_results = {}
    for method, results in [('Grid', grid_results), ('Random', random_results), ('Poly', poly_results)]:
        for model, result in results.items():
            key = f"{method}_{model}"
            all_results[key] = result
    
    if all_results:
        best_model = max(all_results.keys(), key=lambda x: all_results[x]['test_r2'])
        best_result = all_results[best_model]
        
        print(f"\n最佳模型: {best_model}")
        print(f"最佳参数: {best_result['best_params']}")
        print(f"测试集R²: {best_result['test_r2']:.6f}")
        print(f"交叉验证R²: {best_result['best_score']:.6f}")
    
    print("\n快速超参数调优分析完成！")

if __name__ == "__main__":
    main()
