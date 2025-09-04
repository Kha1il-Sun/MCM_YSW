使用 **支持向量回归（SVR）** 来进行建模，预测  **Y染色体浓度** 。

### **步骤 1：导入库并加载数据**

首先，我们需要导入必要的库，并加载特征数据和目标变量数据。

```python
import pandas as pd
import numpy as np
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# 读取特征数据和目标变量数据
X = pd.read_csv('/mnt/data/X_features.csv')  # 特征数据
y = pd.read_csv('/mnt/data/y_target.csv')    # 目标变量数据

# 查看数据结构
print(X.head())
print(y.head())
```

### **步骤 2：数据分割**

将数据分割为训练集和测试集，通常按照 **80% 训练集和 20% 测试集** 的比例进行分割。

```python
# 数据拆分：80%训练数据，20%测试数据
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 检查分割结果
print(X_train.shape, X_test.shape)
```

### **步骤 3：训练支持向量回归（SVR）模型**

创建**SVR**模型，并进行训练。

```python
# 创建支持向量回归（SVR）模型
svr_model = SVR(kernel='rbf')  # 使用RBF核

# 训练模型
svr_model.fit(X_train, y_train)

# 输出训练完成提示
print("SVR模型训练完成!")
```

### **步骤 4：模型预测与评估**

使用训练好的模型进行预测，并计算 **评估指标** （如  **MSE** 、 **RMSE** 、 **R²** ）来评估模型的性能。

```python
# 使用测试集进行预测
y_pred_svr = svr_model.predict(X_test)

# 计算评估指标
mse_svr = mean_squared_error(y_test, y_pred_svr)  # 均方误差
rmse_svr = np.sqrt(mse_svr)  # 均方根误差
r2_svr = r2_score(y_test, y_pred_svr)  # R²值

# 输出评估结果
print(f'SVR - MSE: {mse_svr:.4f}')
print(f'SVR - RMSE: {rmse_svr:.4f}')
print(f'SVR - R²: {r2_svr:.4f}')
```

### **步骤 5：结果分析**

#### **a. MSE（均方误差）** ：

* **MSE**越小，模型的预测误差越小，说明模型拟合较好。

#### **b. RMSE（均方根误差）** ：

* **RMSE**提供了误差的尺度，越小表示模型预测越准确。

#### **c. R²（决定系数）** ：

* **R²**反映了模型对数据的拟合能力，值越接近1表示模型越好。如果R²较低，说明模型的拟合效果较差，可能需要进行模型优化。

### **步骤 6：调整模型参数**

1. **调整C参数** ：

* **C**控制着模型的复杂度，较大的C值可能导致过拟合，较小的C值则可能导致欠拟合。你可以尝试不同的C值来优化模型。

1. **调整epsilon参数** ：

* **epsilon**控制着模型的拟合精度。较大的epsilon值会使模型更加宽松，导致欠拟合；较小的epsilon值会使模型更加严格，可能导致过拟合。

1. **使用GridSearchCV进行参数优化** ：

* 使用**GridSearchCV**或**RandomizedSearchCV**来搜索最优的超参数组合，以便获得最佳模型。

### **代码示例：使用GridSearchCV调优SVR参数**

```python
from sklearn.model_selection import GridSearchCV

# 定义参数网格
param_grid = {
    'C': [0.1, 1, 10],        # 惩罚参数
    'epsilon': [0.1, 0.2, 0.5],  # epsilon参数
    'kernel': ['rbf', 'linear'],  # 核函数选择
}

# 创建GridSearchCV对象
grid_search = GridSearchCV(estimator=SVR(), param_grid=param_grid, cv=5, verbose=2, n_jobs=-1)

# 训练模型
grid_search.fit(X_train, y_train)

# 输出最佳参数
print("最佳参数组合:", grid_search.best_params_)

# 使用最佳参数进行预测
best_model = grid_search.best_estimator_
y_pred_best = best_model.predict(X_test)

# 计算评估指标
mse_best = mean_squared_error(y_test, y_pred_best)
rmse_best = np.sqrt(mse_best)
r2_best = r2_score(y_test, y_pred_best)

# 输出评估结果
print(f'Optimized SVR - MSE: {mse_best:.4f}')
print(f'Optimized SVR - RMSE: {rmse_best:.4f}')
print(f'Optimized SVR - R²: {r2_best:.4f}')
```

### **总结**

1. **训练SVR模型** ：通过上述步骤，我们使用**支持向量回归（SVR）**对数据进行了建模和预测。
2. **评估模型性能** ：通过MSE、RMSE和R²值来评估模型的效果。
3. **调优SVR模型** ：你可以通过调整 **C** 、**epsilon**和**kernel**等参数，进一步优化模型。
