import pandas as pd
import numpy as np
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

def load_data():
    """加载特征数据和目标变量数据"""
    print("正在加载数据...")
    
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
    
    return X, y

def basic_svr_model(X, y):
    """训练基本的SVR模型"""
    print("\n" + "="*50)
    print("训练基本SVR模型")
    print("="*50)
    
    # 数据拆分：80%训练数据，20%测试数据
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 检查分割结果
    print(f"训练集形状: {X_train.shape}")
    print(f"测试集形状: {X_test.shape}")
    
    # 创建支持向量回归（SVR）模型
    svr_model = SVR(kernel='rbf')  # 使用RBF核
    
    # 训练模型
    print("正在训练SVR模型...")
    svr_model.fit(X_train, y_train.values.ravel())
    
    # 输出训练完成提示
    print("SVR模型训练完成!")
    
    # 使用测试集进行预测
    y_pred_svr = svr_model.predict(X_test)
    
    # 计算评估指标
    mse_svr = mean_squared_error(y_test, y_pred_svr)  # 均方误差
    rmse_svr = np.sqrt(mse_svr)  # 均方根误差
    r2_svr = r2_score(y_test, y_pred_svr)  # R²值
    
    # 输出评估结果
    print(f'\n基本SVR模型评估结果:')
    print(f'MSE: {mse_svr:.6f}')
    print(f'RMSE: {rmse_svr:.6f}')
    print(f'R²: {r2_svr:.6f}')
    
    return svr_model, X_test, y_test, y_pred_svr, mse_svr, rmse_svr, r2_svr

def optimized_svr_model(X, y):
    """使用GridSearchCV优化SVR模型参数"""
    print("\n" + "="*50)
    print("优化SVR模型参数")
    print("="*50)
    
    # 数据拆分：80%训练数据，20%测试数据
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 定义参数网格
    param_grid = {
        'C': [0.1, 1, 10, 100],           # 惩罚参数
        'epsilon': [0.01, 0.1, 0.2, 0.5], # epsilon参数
        'kernel': ['rbf', 'linear'],       # 核函数选择
    }
    
    print("参数网格:")
    for key, value in param_grid.items():
        print(f"  {key}: {value}")
    
    # 创建GridSearchCV对象
    print("\n开始网格搜索...")
    grid_search = GridSearchCV(
        estimator=SVR(), 
        param_grid=param_grid, 
        cv=5, 
        verbose=1, 
        n_jobs=1,  # 改为单线程避免编码问题
        scoring='neg_mean_squared_error'
    )
    
    # 训练模型
    grid_search.fit(X_train, y_train.values.ravel())
    
    # 输出最佳参数
    print(f"\n最佳参数组合: {grid_search.best_params_}")
    print(f"最佳交叉验证分数: {-grid_search.best_score_:.6f}")
    
    # 使用最佳参数进行预测
    best_model = grid_search.best_estimator_
    y_pred_best = best_model.predict(X_test)
    
    # 计算评估指标
    mse_best = mean_squared_error(y_test, y_pred_best)
    rmse_best = np.sqrt(mse_best)
    r2_best = r2_score(y_test, y_pred_best)
    
    # 输出评估结果
    print(f'\n优化后SVR模型评估结果:')
    print(f'MSE: {mse_best:.6f}')
    print(f'RMSE: {rmse_best:.6f}')
    print(f'R²: {r2_best:.6f}')
    
    return best_model, X_test, y_test, y_pred_best, mse_best, rmse_best, r2_best, grid_search.best_params_

def plot_results(y_test, y_pred_basic, y_pred_optimized, mse_basic, mse_optimized, r2_basic, r2_optimized):
    """绘制预测结果对比图"""
    print("\n正在生成结果可视化图表...")
    
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 创建子图
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. 基本SVR预测结果散点图
    axes[0, 0].scatter(y_test, y_pred_basic, alpha=0.6, color='blue')
    axes[0, 0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    axes[0, 0].set_xlabel('实际值')
    axes[0, 0].set_ylabel('预测值')
    axes[0, 0].set_title(f'基本SVR模型预测结果\nR² = {r2_basic:.4f}, MSE = {mse_basic:.6f}')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. 优化SVR预测结果散点图
    axes[0, 1].scatter(y_test, y_pred_optimized, alpha=0.6, color='green')
    axes[0, 1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    axes[0, 1].set_xlabel('实际值')
    axes[0, 1].set_ylabel('预测值')
    axes[0, 1].set_title(f'优化SVR模型预测结果\nR² = {r2_optimized:.4f}, MSE = {mse_optimized:.6f}')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. 残差图 - 基本SVR
    residuals_basic = y_test.values.ravel() - y_pred_basic
    axes[1, 0].scatter(y_pred_basic, residuals_basic, alpha=0.6, color='blue')
    axes[1, 0].axhline(y=0, color='r', linestyle='--')
    axes[1, 0].set_xlabel('预测值')
    axes[1, 0].set_ylabel('残差')
    axes[1, 0].set_title('基本SVR模型残差图')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. 残差图 - 优化SVR
    residuals_optimized = y_test.values.ravel() - y_pred_optimized
    axes[1, 1].scatter(y_pred_optimized, residuals_optimized, alpha=0.6, color='green')
    axes[1, 1].axhline(y=0, color='r', linestyle='--')
    axes[1, 1].set_xlabel('预测值')
    axes[1, 1].set_ylabel('残差')
    axes[1, 1].set_title('优化SVR模型残差图')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('svr_analysis.png', dpi=300, bbox_inches='tight')
    print("结果图表已保存为 'svr_analysis.png'")
    
    # 显示图表
    plt.show()

def save_results(mse_basic, rmse_basic, r2_basic, mse_optimized, rmse_optimized, r2_optimized, best_params):
    """保存结果到文件"""
    print("\n正在保存结果...")
    
    with open('svr_results.txt', 'w', encoding='utf-8') as f:
        f.write("SVR模型训练结果\n")
        f.write("="*50 + "\n\n")
        
        f.write("基本SVR模型结果:\n")
        f.write(f"MSE: {mse_basic:.6f}\n")
        f.write(f"RMSE: {rmse_basic:.6f}\n")
        f.write(f"R²: {r2_basic:.6f}\n\n")
        
        f.write("优化SVR模型结果:\n")
        f.write(f"最佳参数: {best_params}\n")
        f.write(f"MSE: {mse_optimized:.6f}\n")
        f.write(f"RMSE: {rmse_optimized:.6f}\n")
        f.write(f"R²: {r2_optimized:.6f}\n\n")
        
        f.write("模型性能对比:\n")
        f.write(f"MSE改善: {((mse_basic - mse_optimized) / mse_basic * 100):.2f}%\n")
        f.write(f"RMSE改善: {((rmse_basic - rmse_optimized) / rmse_basic * 100):.2f}%\n")
        f.write(f"R²改善: {((r2_optimized - r2_basic) / abs(r2_basic) * 100):.2f}%\n")
    
    print("结果已保存到 'svr_results.txt'")

def main():
    """主函数"""
    print("SVR模型训练开始")
    print("="*50)
    
    try:
        # 1. 加载数据
        X, y = load_data()
        
        # 2. 训练基本SVR模型
        basic_model, X_test, y_test, y_pred_basic, mse_basic, rmse_basic, r2_basic = basic_svr_model(X, y)
        
        # 3. 训练优化SVR模型
        optimized_model, _, _, y_pred_optimized, mse_optimized, rmse_optimized, r2_optimized, best_params = optimized_svr_model(X, y)
        
        # 4. 绘制结果对比图
        plot_results(y_test, y_pred_basic, y_pred_optimized, mse_basic, mse_optimized, r2_basic, r2_optimized)
        
        # 5. 保存结果
        save_results(mse_basic, rmse_basic, r2_basic, mse_optimized, rmse_optimized, r2_optimized, best_params)
        
        print("\n" + "="*50)
        print("SVR模型训练完成!")
        print("="*50)
        
    except Exception as e:
        print(f"训练过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
