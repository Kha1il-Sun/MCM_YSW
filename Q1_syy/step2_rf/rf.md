### **1. 数据加载和基本处理**

首先，我们加载特征数据和目标变量数据，并检查数据结构。以下是代码：

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# 读取特征数据和目标变量数据
X = pd.read_csv('/mnt/data/X_features.csv')  # 特征数据
y = pd.read_csv('/mnt/data/y_target.csv')    # 目标变量数据

# 查看数据结构
print(X.head())
print(y.head())
```

### **2. 数据拆分**

我们将数据分为 **训练集** 和  **测试集** ，通常按照 **80% 训练数据和 20% 测试数据** 的比例进行分割。

```python
# 数据拆分：80%训练数据，20%测试数据
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 检查分割结果
print(X_train.shape, X_test.shape)
```

### **3. 创建并训练随机森林模型**

在训练 **随机森林回归模型** 时，我们可以设置一些参数来控制模型的复杂度。以下是模型的创建和训练过程：

```python
# 创建随机森林回归模型
rf_model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=6)

# 训练模型
rf_model.fit(X_train, y_train)

# 输出训练完成提示
print("随机森林模型训练完成!")
```

### **4. 预测与评估**

接下来，我们使用 **测试集** 对训练好的模型进行预测，并计算评估指标。

```python
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
```

### **5. 特征重要性分析**

**随机森林**能够给出每个特征的重要性。我们可以绘制特征重要性图，查看每个特征对模型预测的贡献。

```python
# 获取特征重要性
feature_importances = rf_model.feature_importances_

# 绘制特征重要性图
plt.bar(X.columns, feature_importances, color='skyblue', edgecolor='navy', alpha=0.7)
plt.xlabel('Feature')
plt.ylabel('Importance')
plt.title('Feature Importance in Random Forest Model')
plt.xticks(rotation=90)
plt.show()
```

### **6. 模型调优**

#### **参数调整建议：**

* **n_estimators** ：树的数量，默认100。增加树的数量通常能提高模型的稳定性，但会增加计算量。可以从100增加到200或300。
* **max_depth** ：树的最大深度。较深的树可能导致过拟合。可以从6增加到8或10，看看效果如何。
* **min_samples_split** ：控制每个内部节点的最小样本数，较高的值可以避免过拟合。
* **min_samples_leaf** ：每个叶子节点的最小样本数。增加该值可以减少模型复杂度，从而避免过拟合。

#### **GridSearchCV调优参数**

使用 **GridSearchCV** 来优化超参数。通过不同的参数组合进行搜索，并选择最佳参数。

```python
from sklearn.model_selection import GridSearchCV

# 定义参数网格
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [6, 8, 10],
    'min_samples_split': [2, 4, 6],
    'min_samples_leaf': [1, 2, 4]
}

# 创建GridSearchCV对象
grid_search = GridSearchCV(estimator=RandomForestRegressor(random_state=42), param_grid=param_grid, cv=3, verbose=2)

# 训练模型
grid_search.fit(X_train, y_train)

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
```

### **7. 总结与建议**

#### **模型效果评估** ：

* 使用  **MSE** 、**RMSE** 和 **R²** 来评估模型的预测效果。如果 **R²** 值较低，可以尝试调整  **n_estimators** 、**max_depth** 等参数，或使用更多的特征。

#### **特征重要性** ：

* **随机森林回归**不仅可以提供预测结果，还能评估每个特征的贡献，帮助我们识别哪些特征对 **Y染色体浓度** 的预测影响较大。

#### **调优模型参数** ：

* 使用 **GridSearchCV** 自动调整超参数，能够得到最佳的模型性能。
