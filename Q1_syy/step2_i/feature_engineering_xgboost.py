import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

def load_data():
    """加载数据"""
    print("正在加载数据...")
    
    # 读取特征数据和目标数据
    X = pd.read_csv('../step1/X_features.csv')
    y = pd.read_csv('../step1/y_target.csv')
    
    # 从processed_data.csv中获取年龄信息
    processed_data = pd.read_csv('../step1/processed_data.csv')
    
    # 将年龄信息添加到特征中
    X['年龄'] = processed_data['年龄'].copy()
    
    # 确保目标变量是数值类型（从DataFrame中提取Series）
    y = y['Y染色体浓度'].copy()
    
    print(f"数据形状: X={X.shape}, y={y.shape}")
    print(f"特征列: {list(X.columns)}")
    print(f"目标变量: Y染色体浓度")
    
    return X, y

def feature_engineering(X):
    """特征工程：创建交互项和标准化"""
    print("\n正在进行特征工程...")
    
    # 创建原始特征的副本
    X_processed = X.copy()
    
    # 1. 创建特征交互项
    print("创建特征交互项...")
    
    # 孕周数 * BMI
    X_processed['孕周数_BMI'] = X_processed['孕周数'] * X_processed['BMI']
    
    # BMI和孕妇年龄的交互项
    X_processed['BMI_年龄'] = X_processed['BMI'] * X_processed['年龄']
    
    # 孕周数 * 年龄
    X_processed['孕周数_年龄'] = X_processed['孕周数'] * X_processed['年龄']
    
    print(f"原始特征: {list(X.columns)}")
    print(f"新增交互特征: ['孕周数_BMI', 'BMI_年龄', '孕周数_年龄']")
    
    # 2. 数据标准化：对BMI和孕周数进行标准化
    print("进行数据标准化...")
    scaler = StandardScaler()
    
    # 选择需要标准化的特征
    features_to_scale = ['BMI', '孕周数']
    X_processed[features_to_scale] = scaler.fit_transform(X_processed[features_to_scale])
    
    print(f"已标准化特征: {features_to_scale}")
    
    # 3. 显示处理后的特征信息
    print(f"\n处理后的特征矩阵形状: {X_processed.shape}")
    print("特征列:", list(X_processed.columns))
    print("\n前5行数据:")
    print(X_processed.head())
    
    return X_processed, scaler

def train_xgboost_model(X, y):
    """训练XGBoost模型"""
    print("\n开始训练XGBoost模型...")
    
    # 数据拆分：80%训练数据，20%测试数据
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"训练集大小: {X_train.shape}")
    print(f"测试集大小: {X_test.shape}")
    
    # 创建并训练XGBoost模型
    xgb_model = xgb.XGBRegressor(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        n_jobs=-1
    )
    
    print("正在训练模型...")
    xgb_model.fit(X_train, y_train)
    
    # 预测
    y_pred_train = xgb_model.predict(X_train)
    y_pred_test = xgb_model.predict(X_test)
    
    return xgb_model, X_train, X_test, y_train, y_test, y_pred_train, y_pred_test

def evaluate_model(y_true, y_pred, dataset_name):
    """评估模型性能"""
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    
    print(f"\n{dataset_name}集评估结果:")
    print(f"MSE: {mse:.6f}")
    print(f"RMSE: {rmse:.6f}")
    print(f"R²: {r2:.6f}")
    
    return mse, rmse, r2

def plot_feature_importance(model, feature_names, save_path='feature_importance.png'):
    """绘制特征重要性图"""
    print("\n正在生成特征重要性图...")
    
    feature_importances = model.feature_importances_
    
    # 创建DataFrame便于排序
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': feature_importances
    }).sort_values('importance', ascending=True)
    
    # 绘制水平条形图
    plt.figure(figsize=(10, 6))
    bars = plt.barh(importance_df['feature'], importance_df['importance'], 
                    color='skyblue', edgecolor='navy', alpha=0.7)
    
    plt.xlabel('特征重要性')
    plt.ylabel('特征')
    plt.title('XGBoost模型特征重要性分析')
    plt.grid(axis='x', alpha=0.3)
    
    # 添加数值标签
    for i, bar in enumerate(bars):
        width = bar.get_width()
        plt.text(width + 0.001, bar.get_y() + bar.get_height()/2, 
                f'{width:.3f}', ha='left', va='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"特征重要性图已保存为: {save_path}")
    
    return importance_df

def plot_predictions(y_true, y_pred, dataset_name, save_path=None):
    """绘制预测结果对比图"""
    plt.figure(figsize=(8, 6))
    
    # 散点图
    plt.scatter(y_true, y_pred, alpha=0.6, color='blue')
    
    # 完美预测线
    min_val = min(min(y_true), min(y_pred))
    max_val = max(max(y_true), max(y_pred))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='完美预测线')
    
    plt.xlabel('实际值')
    plt.ylabel('预测值')
    plt.title(f'{dataset_name}集预测结果对比')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

def save_results(train_results, test_results, feature_importance_df, output_file='xgboost_results.txt'):
    """保存结果到文件"""
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("XGBoost模型训练结果\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("训练集评估结果:\n")
        f.write(f"MSE: {train_results[0]:.6f}\n")
        f.write(f"RMSE: {train_results[1]:.6f}\n")
        f.write(f"R²: {train_results[2]:.6f}\n\n")
        
        f.write("测试集评估结果:\n")
        f.write(f"MSE: {test_results[0]:.6f}\n")
        f.write(f"RMSE: {test_results[1]:.6f}\n")
        f.write(f"R²: {test_results[2]:.6f}\n\n")
        
        f.write("特征重要性排序:\n")
        f.write("-" * 30 + "\n")
        for idx, row in feature_importance_df.iterrows():
            f.write(f"{row['feature']}: {row['importance']:.6f}\n")
    
    print(f"结果已保存到: {output_file}")

def main():
    """主函数"""
    print("开始XGBoost特征工程和模型训练")
    print("=" * 50)
    
    # 1. 加载数据
    X, y = load_data()
    
    # 2. 特征工程
    X_processed, scaler = feature_engineering(X)
    
    # 3. 训练模型
    model, X_train, X_test, y_train, y_test, y_pred_train, y_pred_test = train_xgboost_model(X_processed, y)
    
    # 4. 评估模型
    train_results = evaluate_model(y_train, y_pred_train, "训练")
    test_results = evaluate_model(y_test, y_pred_test, "测试")
    
    # 5. 特征重要性分析
    feature_importance_df = plot_feature_importance(model, X_processed.columns)
    
    # 6. 预测结果可视化
    plot_predictions(y_train, y_pred_train, "训练", "train_predictions.png")
    plot_predictions(y_test, y_pred_test, "测试", "test_predictions.png")
    
    # 7. 保存结果
    save_results(train_results, test_results, feature_importance_df)
    
    print("\n" + "=" * 50)
    print("XGBoost模型训练完成！")
    print("生成的文件:")
    print("- feature_importance.png: 特征重要性图")
    print("- train_predictions.png: 训练集预测对比图")
    print("- test_predictions.png: 测试集预测对比图")
    print("- xgboost_results.txt: 详细结果报告")

if __name__ == "__main__":
    main()
