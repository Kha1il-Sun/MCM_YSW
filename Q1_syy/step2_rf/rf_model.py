import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import os

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

def main():
    print("=" * 50)
    print("随机森林回归模型训练开始")
    print("=" * 50)
    
    # 步骤1：导入库和加载数据
    print("\n步骤1：加载数据...")
    
    # 读取特征数据和目标变量数据
    X = pd.read_csv('../step1/X_features.csv')  # 特征数据
    y = pd.read_csv('../step1/y_target.csv')    # 目标变量数据
    
    # 查看数据结构
    print(f"特征数据形状: {X.shape}")
    print(f"目标变量形状: {y.shape}")
    print("\n特征数据前5行:")
    print(X.head())
    print("\n目标变量前5行:")
    print(y.head())
    
    # 步骤2：数据分割
    print("\n步骤2：数据分割...")
    
    # 数据拆分：80%训练数据，20%测试数据
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 检查分割结果
    print(f"训练集特征形状: {X_train.shape}")
    print(f"测试集特征形状: {X_test.shape}")
    print(f"训练集目标形状: {y_train.shape}")
    print(f"测试集目标形状: {y_test.shape}")
    
    # 步骤3：训练随机森林模型
    print("\n步骤3：训练随机森林模型...")
    
    # 创建随机森林回归模型
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=6)
    
    # 训练模型
    rf_model.fit(X_train, y_train.values.ravel())
    
    # 输出训练完成提示
    print("随机森林模型训练完成!")
    
    # 步骤4：模型预测与评估
    print("\n步骤4：模型预测与评估...")
    
    # 预测结果
    y_pred_rf = rf_model.predict(X_test)
    
    # 计算评估指标
    mse_rf = mean_squared_error(y_test, y_pred_rf)  # 均方误差
    rmse_rf = np.sqrt(mse_rf)  # 均方根误差
    r2_rf = r2_score(y_test, y_pred_rf)  # R²值
    
    # 输出评估结果
    print(f'Random Forest - MSE: {mse_rf:.4f}')
    print(f'Random Forest - RMSE: {rmse_rf:.4f}')
    print(f'Random Forest - R²: {r2_rf:.4f}')
    
    # 步骤5：特征重要性分析
    print("\n步骤5：特征重要性分析...")
    
    # 获取特征重要性
    feature_importances = rf_model.feature_importances_
    
    # 创建特征重要性图
    plt.figure(figsize=(10, 6))
    plt.bar(X.columns, feature_importances, color='skyblue', edgecolor='navy', alpha=0.7)
    plt.xlabel('特征')
    plt.ylabel('重要性')
    plt.title('随机森林模型特征重要性')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('rf_feature_importance.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 打印特征重要性
    print("\n特征重要性排序:")
    feature_importance_df = pd.DataFrame({
        'feature': X.columns,
        'importance': feature_importances
    }).sort_values('importance', ascending=False)
    print(feature_importance_df)
    
    # 步骤6：模型调优
    print("\n步骤6：模型调优...")
    
    # 定义参数网格
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [6, 8, 10],
        'min_samples_split': [2, 4, 6],
        'min_samples_leaf': [1, 2, 4]
    }
    
    # 创建GridSearchCV对象
    grid_search = GridSearchCV(
        estimator=RandomForestRegressor(random_state=42), 
        param_grid=param_grid, 
        cv=3, 
        verbose=2,
        n_jobs=1  # 设置为1避免中文路径编码问题
    )
    
    # 训练模型
    print("开始网格搜索...")
    grid_search.fit(X_train, y_train.values.ravel())
    
    # 输出最佳参数
    print("最佳参数组合:", grid_search.best_params_)
    
    # 使用最佳参数进行预测
    best_rf_model = grid_search.best_estimator_
    y_pred_best = best_rf_model.predict(X_test)
    
    # 计算评估指标
    mse_best = mean_squared_error(y_test, y_pred_best)
    rmse_best = np.sqrt(mse_best)
    r2_best = r2_score(y_test, y_pred_best)
    
    # 输出评估结果
    print(f'Optimized Random Forest - MSE: {mse_best:.4f}')
    print(f'Optimized Random Forest - RMSE: {rmse_best:.4f}')
    print(f'Optimized Random Forest - R²: {r2_best:.4f}')
    
    # 创建预测结果对比图
    plt.figure(figsize=(15, 5))
    
    # 子图1：实际值 vs 预测值（原始模型）
    plt.subplot(1, 3, 1)
    plt.scatter(y_test, y_pred_rf, alpha=0.6, color='blue')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('实际值')
    plt.ylabel('预测值')
    plt.title(f'原始模型 (R² = {r2_rf:.4f})')
    
    # 子图2：实际值 vs 预测值（优化模型）
    plt.subplot(1, 3, 2)
    plt.scatter(y_test, y_pred_best, alpha=0.6, color='green')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('实际值')
    plt.ylabel('预测值')
    plt.title(f'优化模型 (R² = {r2_best:.4f})')
    
    # 子图3：残差图
    plt.subplot(1, 3, 3)
    residuals = y_test.values.ravel() - y_pred_best
    plt.scatter(y_pred_best, residuals, alpha=0.6, color='orange')
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('预测值')
    plt.ylabel('残差')
    plt.title('残差图')
    
    plt.tight_layout()
    plt.savefig('rf_predictions.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 保存结果到文件
    with open('rf_results.txt', 'w', encoding='utf-8') as f:
        f.write("随机森林回归模型结果\n")
        f.write("=" * 50 + "\n\n")
        f.write("原始模型结果:\n")
        f.write(f"MSE: {mse_rf:.4f}\n")
        f.write(f"RMSE: {rmse_rf:.4f}\n")
        f.write(f"R²: {r2_rf:.4f}\n\n")
        f.write("优化模型结果:\n")
        f.write(f"最佳参数: {grid_search.best_params_}\n")
        f.write(f"MSE: {mse_best:.4f}\n")
        f.write(f"RMSE: {rmse_best:.4f}\n")
        f.write(f"R²: {r2_best:.4f}\n\n")
        f.write("特征重要性排序:\n")
        f.write(feature_importance_df.to_string(index=False))
    
    print("\n" + "=" * 50)
    print("随机森林模型训练完成!")
    print("结果已保存到:")
    print("- rf_results.txt: 详细结果")
    print("- rf_feature_importance.png: 特征重要性图")
    print("- rf_predictions.png: 预测结果图")
    print("=" * 50)

if __name__ == "__main__":
    main()
