# XGBoost 回归模型 - Y染色体浓度预测

## 项目简介

本项目使用 XGBoost 算法构建回归模型，用于预测 Y染色体浓度。模型基于孕周数和BMI两个特征进行训练和预测。

## 文件结构

```
step2_xgboost/
├── xgboost_model.py          # 主要的XGBoost模型实现
├── run_xgboost.py            # 运行脚本
├── run_xgboost.bat           # Windows批处理运行文件
├── requirements.txt          # 依赖包列表
├── README.md                 # 项目说明文档
└── xgboost.md               # 原始说明文档
```

## 环境要求

- Python 3.7+
- conda环境: pytorch

## 安装依赖

在运行程序前，请确保已激活pytorch环境并安装所需依赖：

```bash
conda activate pytorch
pip install -r requirements.txt
```

## 运行方式

### 方式1: 使用批处理文件（推荐）
双击运行 `run_xgboost.bat` 文件，程序会自动：
1. 激活pytorch环境
2. 安装依赖包
3. 运行XGBoost模型

### 方式2: 手动运行
```bash
conda activate pytorch
python run_xgboost.py
```

## 输出文件

程序运行完成后会生成以下文件：

1. **xgboost_feature_importance.png** - 特征重要性可视化图
2. **xgboost_predictions.png** - 预测结果对比图
3. **xgboost_results.txt** - 详细的模型评估结果

## 模型评估指标

- **MSE (均方误差)**: 衡量预测值与真实值之间的误差，误差越小，模型预测越准确
- **RMSE (均方根误差)**: 是MSE的平方根，越小表示误差越小
- **R² (决定系数)**: 评估模型的拟合能力，值越接近1说明模型越好

## 特征重要性分析

XGBoost模型提供内置的特征重要性分析功能，可以评估每个特征对模型预测的贡献程度：
- 孕周数
- BMI

## 模型参数

当前模型使用以下参数：
- n_estimators: 100 (树的数量)
- learning_rate: 0.1 (学习率)
- max_depth: 6 (最大深度)
- subsample: 0.8 (子采样比例)
- colsample_bytree: 0.8 (特征采样比例)

## 数据说明

- 训练集: 80%的数据用于模型训练
- 测试集: 20%的数据用于模型评估
- 特征: 孕周数、BMI
- 目标变量: Y染色体浓度

## 注意事项

1. 确保step1目录中存在预处理好的数据文件：
   - X_features.csv (特征数据)
   - y_target.csv (目标变量数据)

2. 程序会自动处理中文显示，确保系统支持中文字体

3. 如果遇到依赖包安装问题，可以手动安装：
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn xgboost
   ```

## 结果解释

- 较低的MSE和RMSE值表示模型预测准确性较高
- 较高的R²值表示模型拟合效果较好
- 特征重要性分析帮助理解哪些因素对Y染色体浓度预测影响最大
