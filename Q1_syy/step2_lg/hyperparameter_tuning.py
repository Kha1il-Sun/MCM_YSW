#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
超参数调优分析
对多种模型进行超参数调优，包括网格搜索、随机搜索和贝叶斯优化
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
from sklearn.feature_selection import SelectKBest, f_regression
import warnings
warnings.filterwarnings('ignore')

# 贝叶斯优化
try:
    from skopt import BayesSearchCV
    from skopt.space import Real, Integer, Categorical
    BAYES_AVAILABLE = True
except ImportError:
    BAYES_AVAILABLE = False
    print("警告: scikit-optimize未安装，将跳过贝叶斯优化")

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

def create_model_pipelines():
    """创建模型管道"""
    print("\n" + "=" * 60)
    print("创建模型管道")
    print("=" * 60)
    
    pipelines = {
        'LinearRegression': Pipeline([
            ('scaler', StandardScaler()),
            ('regressor', LinearRegression())
        ]),
        
        'Ridge': Pipeline([
            ('scaler', StandardScaler()),
            ('regressor', Ridge(random_state=42))
        ]),
        
        'Lasso': Pipeline([
            ('scaler', StandardScaler()),
            ('regressor', Lasso(random_state=42, max_iter=2000))
        ]),
        
        'ElasticNet': Pipeline([
            ('scaler', StandardScaler()),
            ('regressor', ElasticNet(random_state=42, max_iter=2000))
        ]),
        
        'Ridge_Poly': Pipeline([
            ('poly', PolynomialFeatures(degree=2, include_bias=False)),
            ('scaler', StandardScaler()),
            ('regressor', Ridge(random_state=42))
        ]),
        
        'Lasso_Poly': Pipeline([
            ('poly', PolynomialFeatures(degree=2, include_bias=False)),
            ('scaler', StandardScaler()),
            ('regressor', Lasso(random_state=42, max_iter=2000))
        ]),
        
        'RandomForest': Pipeline([
            ('scaler', StandardScaler()),
            ('regressor', RandomForestRegressor(random_state=42))
        ]),
        
        'SVR': Pipeline([
            ('scaler', StandardScaler()),
            ('regressor', SVR())
        ])
    }
    
    print(f"创建了 {len(pipelines)} 个模型管道")
    for name in pipelines.keys():
        print(f"  - {name}")
    
    return pipelines

def define_parameter_grids():
    """定义参数网格"""
    print("\n" + "=" * 60)
    print("定义参数网格")
    print("=" * 60)
    
    param_grids = {
        'LinearRegression': {},
        
        'Ridge': {
            'regressor__alpha': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
        },
        
        'Lasso': {
            'regressor__alpha': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
        },
        
        'ElasticNet': {
            'regressor__alpha': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
            'regressor__l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9]
        },
        
        'Ridge_Poly': {
            'regressor__alpha': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
        },
        
        'Lasso_Poly': {
            'regressor__alpha': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
        },
        
        'RandomForest': {
            'regressor__n_estimators': [50, 100, 200, 300],
            'regressor__max_depth': [None, 10, 20, 30],
            'regressor__min_samples_split': [2, 5, 10],
            'regressor__min_samples_leaf': [1, 2, 4]
        },
        
        'SVR': {
            'regressor__C': [0.1, 1, 10, 100, 1000],
            'regressor__gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1],
            'regressor__kernel': ['rbf', 'linear', 'poly']
        }
    }
    
    print("参数网格定义完成:")
    for model, params in param_grids.items():
        print(f"  {model}: {len(params)} 个参数")
    
    return param_grids

def define_bayesian_search_spaces():
    """定义贝叶斯搜索空间"""
    if not BAYES_AVAILABLE:
        return {}
    
    print("\n" + "=" * 60)
    print("定义贝叶斯搜索空间")
    print("=" * 60)
    
    search_spaces = {
        'Ridge': {
            'regressor__alpha': Real(0.001, 100.0, prior='log-uniform')
        },
        
        'Lasso': {
            'regressor__alpha': Real(0.001, 100.0, prior='log-uniform')
        },
        
        'ElasticNet': {
            'regressor__alpha': Real(0.001, 100.0, prior='log-uniform'),
            'regressor__l1_ratio': Real(0.1, 0.9, prior='uniform')
        },
        
        'RandomForest': {
            'regressor__n_estimators': Integer(50, 300),
            'regressor__max_depth': Integer(5, 30),
            'regressor__min_samples_split': Integer(2, 10),
            'regressor__min_samples_leaf': Integer(1, 5)
        },
        
        'SVR': {
            'regressor__C': Real(0.1, 1000.0, prior='log-uniform'),
            'regressor__gamma': Real(0.001, 1.0, prior='log-uniform'),
            'regressor__kernel': Categorical(['rbf', 'linear', 'poly'])
        }
    }
    
    print("贝叶斯搜索空间定义完成:")
    for model, space in search_spaces.items():
        print(f"  {model}: {len(space)} 个参数")
    
    return search_spaces

def perform_grid_search(pipelines, param_grids, X, y):
    """执行网格搜索"""
    print("\n" + "=" * 60)
    print("执行网格搜索")
    print("=" * 60)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    grid_results = {}
    
    for model_name, pipeline in pipelines.items():
        if model_name not in param_grids or not param_grids[model_name]:
            print(f"\n{model_name}: 跳过（无参数需要调优）")
            continue
            
        print(f"\n{model_name}: 开始网格搜索...")
        
        # 网格搜索
        grid_search = GridSearchCV(
            pipeline, 
            param_grids[model_name], 
            cv=5, 
            scoring='r2',
            n_jobs=1,
            verbose=0
        )
        
        grid_search.fit(X_train, y_train)
        
        # 预测
        y_train_pred = grid_search.predict(X_train)
        y_test_pred = grid_search.predict(X_test)
        
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
            'cv_scores': grid_search.cv_results_['mean_test_score'],
            'cv_std': grid_search.cv_results_['std_test_score']
        }
        
        print(f"  最佳参数: {grid_search.best_params_}")
        print(f"  最佳CV分数: {grid_search.best_score_:.6f}")
        print(f"  测试集R²: {test_r2:.6f}")
    
    return grid_results

def perform_random_search(pipelines, param_grids, X, y, n_iter=50):
    """执行随机搜索"""
    print("\n" + "=" * 60)
    print("执行随机搜索")
    print("=" * 60)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    random_results = {}
    
    for model_name, pipeline in pipelines.items():
        if model_name not in param_grids or not param_grids[model_name]:
            print(f"\n{model_name}: 跳过（无参数需要调优）")
            continue
            
        print(f"\n{model_name}: 开始随机搜索...")
        
        # 随机搜索
        random_search = RandomizedSearchCV(
            pipeline, 
            param_grids[model_name], 
            n_iter=n_iter,
            cv=5, 
            scoring='r2',
            n_jobs=1,
            random_state=42,
            verbose=0
        )
        
        random_search.fit(X_train, y_train)
        
        # 预测
        y_train_pred = random_search.predict(X_train)
        y_test_pred = random_search.predict(X_test)
        
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
            'cv_scores': random_search.cv_results_['mean_test_score'],
            'cv_std': random_search.cv_results_['std_test_score']
        }
        
        print(f"  最佳参数: {random_search.best_params_}")
        print(f"  最佳CV分数: {random_search.best_score_:.6f}")
        print(f"  测试集R²: {test_r2:.6f}")
    
    return random_results

def perform_bayesian_search(pipelines, search_spaces, X, y, n_iter=50):
    """执行贝叶斯搜索"""
    if not BAYES_AVAILABLE:
        print("\n贝叶斯搜索不可用（scikit-optimize未安装）")
        return {}
    
    print("\n" + "=" * 60)
    print("执行贝叶斯搜索")
    print("=" * 60)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    bayesian_results = {}
    
    for model_name, pipeline in pipelines.items():
        if model_name not in search_spaces:
            print(f"\n{model_name}: 跳过（无贝叶斯搜索空间）")
            continue
            
        print(f"\n{model_name}: 开始贝叶斯搜索...")
        
        # 贝叶斯搜索
        bayesian_search = BayesSearchCV(
            pipeline, 
            search_spaces[model_name], 
            n_iter=n_iter,
            cv=5, 
            scoring='r2',
            n_jobs=1,
            random_state=42,
            verbose=0
        )
        
        bayesian_search.fit(X_train, y_train)
        
        # 预测
        y_train_pred = bayesian_search.predict(X_train)
        y_test_pred = bayesian_search.predict(X_test)
        
        # 计算指标
        train_r2 = r2_score(y_train, y_train_pred)
        test_r2 = r2_score(y_test, y_test_pred)
        train_mse = mean_squared_error(y_train, y_train_pred)
        test_mse = mean_squared_error(y_test, y_test_pred)
        
        bayesian_results[model_name] = {
            'best_params': bayesian_search.best_params_,
            'best_score': bayesian_search.best_score_,
            'train_r2': train_r2,
            'test_r2': test_r2,
            'train_mse': train_mse,
            'test_mse': test_mse,
            'cv_scores': bayesian_search.cv_results_['mean_test_score'],
            'cv_std': bayesian_search.cv_results_['std_test_score']
        }
        
        print(f"  最佳参数: {bayesian_search.best_params_}")
        print(f"  最佳CV分数: {bayesian_search.best_score_:.6f}")
        print(f"  测试集R²: {test_r2:.6f}")
    
    return bayesian_results

def visualize_tuning_results(grid_results, random_results, bayesian_results):
    """可视化调优结果"""
    print("\n" + "=" * 60)
    print("结果可视化")
    print("=" * 60)
    
    # 合并所有结果
    all_results = {}
    for method, results in [('Grid', grid_results), ('Random', random_results), ('Bayesian', bayesian_results)]:
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
    methods = ['Grid', 'Random', 'Bayesian']
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
    
    # 3. 最佳模型参数热图
    best_model = max(all_results.keys(), key=lambda x: all_results[x]['test_r2'])
    best_params = all_results[best_model]['best_params']
    
    if best_params:
        param_names = list(best_params.keys())
        param_values = [str(v) for v in best_params.values()]
        
        # 创建热图数据
        heatmap_data = np.array([[float(v) if v.replace('.', '').replace('-', '').isdigit() else 0 for v in param_values]])
        
        sns.heatmap(heatmap_data, 
                   xticklabels=param_names, 
                   yticklabels=['最佳参数'],
                   annot=True, 
                   fmt='.3f',
                   cmap='YlOrRd',
                   ax=axes[1, 0])
        axes[1, 0].set_title(f'最佳模型参数 ({best_model})')
    else:
        axes[1, 0].text(0.5, 0.5, '无参数信息', ha='center', va='center', 
                       transform=axes[1, 0].transAxes)
        axes[1, 0].set_title('最佳模型参数')
    
    # 4. 性能分布
    axes[1, 1].hist(test_r2_scores, bins=10, alpha=0.7, color='lightblue', edgecolor='black')
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
    plt.savefig('hyperparameter_tuning_results.png', dpi=300, bbox_inches='tight')
    plt.show()

def save_tuning_results(grid_results, random_results, bayesian_results):
    """保存调优结果"""
    print("\n" + "=" * 60)
    print("保存结果")
    print("=" * 60)
    
    with open('hyperparameter_tuning_results.txt', 'w', encoding='utf-8') as f:
        f.write("超参数调优结果\n")
        f.write("=" * 60 + "\n\n")
        
        # 合并所有结果
        all_results = {}
        for method, results in [('Grid', grid_results), ('Random', random_results), ('Bayesian', bayesian_results)]:
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
    
    print("结果已保存到 hyperparameter_tuning_results.txt")

def main():
    """主函数"""
    print("开始超参数调优分析...")
    
    # 1. 加载和准备数据
    X_original, X_enhanced, y = load_and_prepare_data()
    
    # 2. 创建模型管道
    pipelines = create_model_pipelines()
    
    # 3. 定义参数网格
    param_grids = define_parameter_grids()
    
    # 4. 定义贝叶斯搜索空间
    search_spaces = define_bayesian_search_spaces()
    
    # 5. 执行网格搜索
    print("\n使用增强特征进行调优...")
    grid_results = perform_grid_search(pipelines, param_grids, X_enhanced, y)
    
    # 6. 执行随机搜索
    random_results = perform_random_search(pipelines, param_grids, X_enhanced, y, n_iter=30)
    
    # 7. 执行贝叶斯搜索
    bayesian_results = perform_bayesian_search(pipelines, search_spaces, X_enhanced, y, n_iter=30)
    
    # 8. 可视化结果
    visualize_tuning_results(grid_results, random_results, bayesian_results)
    
    # 9. 保存结果
    save_tuning_results(grid_results, random_results, bayesian_results)
    
    # 10. 输出最佳模型
    all_results = {}
    for method, results in [('Grid', grid_results), ('Random', random_results), ('Bayesian', bayesian_results)]:
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
    
    print("\n超参数调优分析完成！")

if __name__ == "__main__":
    main()
