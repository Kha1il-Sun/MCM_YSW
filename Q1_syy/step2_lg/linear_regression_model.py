# 线性回归模型：Y染色体浓度与孕周数、BMI的关系分析
# 作者：基于2.md说明实现
# 日期：2024

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

def load_data():
    """加载预处理后的数据"""
    print("正在加载数据...")
    
    # 读取特征数据和目标变量数据
    X = pd.read_csv('../step1/X_features.csv')
    y = pd.read_csv('../step1/y_target.csv')
    
    # 合并特征数据和目标变量数据
    data = pd.concat([X, y], axis=1)
    
    print(f"数据加载完成，共{len(data)}条记录")
    print(f"特征列：{list(X.columns)}")
    print(f"目标变量：{list(y.columns)}")
    print("\n数据预览：")
    print(data.head())
    print(f"\n数据形状：{data.shape}")
    print(f"\n数据描述性统计：")
    print(data.describe())
    
    return data

def split_data(data):
    """数据分割"""
    print("\n" + "="*50)
    print("步骤2：数据分割")
    print("="*50)
    
    # 选择自变量和因变量
    X = data[['孕周数', 'BMI']]  # 自变量
    y = data['Y染色体浓度']  # 因变量
    
    print(f"自变量：{list(X.columns)}")
    print(f"因变量：Y染色体浓度")
    
    # 数据拆分：80%训练数据，20%测试数据
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # 检查数据分割的结果
    print(f"\n训练集形状：{X_train.shape}")
    print(f"测试集形状：{X_test.shape}")
    print(f"训练集目标变量形状：{y_train.shape}")
    print(f"测试集目标变量形状：{y_test.shape}")
    
    return X_train, X_test, y_train, y_test

def train_model(X_train, y_train):
    """训练线性回归模型"""
    print("\n" + "="*50)
    print("步骤3：训练线性回归模型")
    print("="*50)
    
    # 创建线性回归模型
    lr_model = LinearRegression()
    
    # 训练模型
    print("正在训练模型...")
    lr_model.fit(X_train, y_train)
    
    # 输出回归系数
    print("\n模型训练完成！")
    print("回归系数:")
    for i, feature in enumerate(X_train.columns):
        print(f"  {feature}: {lr_model.coef_[i]:.6f}")
    print(f"截距项: {lr_model.intercept_:.6f}")
    
    return lr_model

def evaluate_model(model, X_test, y_test):
    """模型评估"""
    print("\n" + "="*50)
    print("步骤4：预测与评估")
    print("="*50)
    
    # 使用测试集进行预测
    y_pred = model.predict(X_test)
    
    # 计算评估指标
    mse = mean_squared_error(y_test, y_pred)  # 均方误差
    rmse = np.sqrt(mse)  # 均方根误差
    mae = mean_absolute_error(y_test, y_pred)  # 平均绝对误差
    r2 = r2_score(y_test, y_pred)  # R²值
    
    # 输出评估结果
    print("模型评估结果：")
    print(f"MSE (均方误差): {mse:.6f}")
    print(f"RMSE (均方根误差): {rmse:.6f}")
    print(f"MAE (平均绝对误差): {mae:.6f}")
    print(f"R² (决定系数): {r2:.6f}")
    
    return y_pred, mse, rmse, mae, r2

def analyze_results(model, X_train, y_pred, y_test, r2):
    """结果分析"""
    print("\n" + "="*50)
    print("步骤5：结果分析")
    print("="*50)
    
    # 1. 回归系数分析
    print("1. 回归系数分析：")
    print("   回归系数表示每个特征对Y染色体浓度的影响程度：")
    for i, feature in enumerate(X_train.columns):
        coef = model.coef_[i]
        print(f"   - {feature}: {coef:.6f}")
        if coef > 0:
            print(f"     正值表示{feature}增加时，Y染色体浓度增加")
        else:
            print(f"     负值表示{feature}增加时，Y染色体浓度减少")
    
    print(f"\n   截距项: {model.intercept_:.6f}")
    print("   截距项表示当所有特征为0时的Y染色体浓度预测值")
    
    # 2. 评估指标分析
    print("\n2. 评估指标分析：")
    print(f"   R² = {r2:.6f}")
    if r2 > 0.8:
        print("   模型拟合度很好，能够解释大部分数据变异性")
    elif r2 > 0.6:
        print("   模型拟合度较好，能够解释较多数据变异性")
    elif r2 > 0.4:
        print("   模型拟合度一般，能够解释部分数据变异性")
    else:
        print("   模型拟合度较差，解释数据变异性能力有限")
    
    # 3. 特征重要性分析
    print("\n3. 特征重要性分析：")
    feature_importance = abs(model.coef_)
    total_importance = sum(feature_importance)
    
    for i, feature in enumerate(X_train.columns):
        importance_ratio = feature_importance[i] / total_importance * 100
        print(f"   {feature}: {importance_ratio:.2f}%")

def create_visualizations(model, X_train, X_test, y_train, y_test, y_pred):
    """创建可视化图表"""
    print("\n" + "="*50)
    print("创建可视化图表")
    print("="*50)
    
    # 创建子图
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('线性回归模型分析结果', fontsize=16, fontweight='bold')
    
    # 1. 实际值 vs 预测值散点图
    axes[0, 0].scatter(y_test, y_pred, alpha=0.6, color='blue')
    axes[0, 0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    axes[0, 0].set_xlabel('实际Y染色体浓度')
    axes[0, 0].set_ylabel('预测Y染色体浓度')
    axes[0, 0].set_title('实际值 vs 预测值')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. 残差图
    residuals = y_test - y_pred
    axes[0, 1].scatter(y_pred, residuals, alpha=0.6, color='green')
    axes[0, 1].axhline(y=0, color='r', linestyle='--')
    axes[0, 1].set_xlabel('预测值')
    axes[0, 1].set_ylabel('残差')
    axes[0, 1].set_title('残差图')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. 特征重要性
    feature_names = X_train.columns
    feature_importance = abs(model.coef_)
    axes[1, 0].bar(feature_names, feature_importance, color=['skyblue', 'lightcoral'])
    axes[1, 0].set_xlabel('特征')
    axes[1, 0].set_ylabel('回归系数绝对值')
    axes[1, 0].set_title('特征重要性（回归系数绝对值）')
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    # 4. 预测误差分布
    axes[1, 1].hist(residuals, bins=30, alpha=0.7, color='orange', edgecolor='black')
    axes[1, 1].set_xlabel('残差')
    axes[1, 1].set_ylabel('频次')
    axes[1, 1].set_title('预测误差分布')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('linear_regression_analysis.png', dpi=300, bbox_inches='tight')
    print("可视化图表已保存为 'linear_regression_analysis.png'")
    plt.show()

def save_results(model, X_train, y_pred, y_test, mse, rmse, mae, r2):
    """保存结果到文件"""
    print("\n" + "="*50)
    print("保存结果")
    print("="*50)
    
    # 保存模型参数
    results = {
        '模型类型': '线性回归',
        '特征': list(X_train.columns),
        '回归系数': model.coef_.tolist(),
        '截距项': model.intercept_,
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'R²': r2
    }
    
    # 保存到文本文件
    with open('linear_regression_results.txt', 'w', encoding='utf-8') as f:
        f.write("线性回归模型结果\n")
        f.write("="*50 + "\n\n")
        f.write(f"模型类型: {results['模型类型']}\n")
        f.write(f"特征: {results['特征']}\n\n")
        f.write("回归系数:\n")
        for i, feature in enumerate(X_train.columns):
            f.write(f"  {feature}: {model.coef_[i]:.6f}\n")
        f.write(f"截距项: {model.intercept_:.6f}\n\n")
        f.write("评估指标:\n")
        f.write(f"  MSE: {mse:.6f}\n")
        f.write(f"  RMSE: {rmse:.6f}\n")
        f.write(f"  MAE: {mae:.6f}\n")
        f.write(f"  R²: {r2:.6f}\n")
    
    print("结果已保存到 'linear_regression_results.txt'")

def main():
    """主函数"""
    print("线性回归模型：Y染色体浓度与孕周数、BMI的关系分析")
    print("="*60)
    
    try:
        # 步骤1：加载数据
        print("\n" + "="*50)
        print("步骤1：导入所需的库和数据")
        print("="*50)
        data = load_data()
        
        # 步骤2：数据分割
        X_train, X_test, y_train, y_test = split_data(data)
        
        # 步骤3：训练模型
        model = train_model(X_train, y_train)
        
        # 步骤4：模型评估
        y_pred, mse, rmse, mae, r2 = evaluate_model(model, X_test, y_test)
        
        # 步骤5：结果分析
        analyze_results(model, X_train, y_pred, y_test, r2)
        
        # 创建可视化
        create_visualizations(model, X_train, X_test, y_train, y_test, y_pred)
        
        # 保存结果
        save_results(model, X_train, y_pred, y_test, mse, rmse, mae, r2)
        
        print("\n" + "="*60)
        print("线性回归模型分析完成！")
        print("="*60)
        
    except Exception as e:
        print(f"程序执行出错：{e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
