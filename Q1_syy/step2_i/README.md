# 特征工程和XGBoost模型训练

本项目基于i.md中的说明，实现了特征交互和标准化处理，并使用XGBoost模型进行Y染色体浓度预测。

## 功能特点

### 1. 特征工程
- **特征交互项创建**：
  - 孕周数 × BMI
  - BMI × 年龄  
  - 孕周数 × 年龄
- **数据标准化**：对BMI和孕周数进行标准化处理

### 2. 模型训练
- 使用XGBoost回归模型
- 80%训练集，20%测试集
- 完整的模型评估指标

### 3. 结果分析
- 特征重要性分析
- 预测结果可视化
- 详细的性能评估报告

## 文件说明

- `feature_engineering_xgboost.py`: 主要代码文件
- `requirements.txt`: 依赖包列表
- `run_feature_engineering.bat`: Windows运行脚本
- `README.md`: 项目说明文档

## 运行方法

### 方法1：使用批处理文件（推荐）
```bash
run_feature_engineering.bat
```

### 方法2：手动运行
```bash
# 激活conda环境
conda activate pytorch

# 安装依赖
pip install -r requirements.txt

# 运行代码
python feature_engineering_xgboost.py
```

## 输出文件

运行完成后会生成以下文件：
- `feature_importance.png`: 特征重要性可视化图
- `train_predictions.png`: 训练集预测结果对比图
- `test_predictions.png`: 测试集预测结果对比图
- `xgboost_results.txt`: 详细的模型评估结果

## 评估指标

- **MSE**: 均方误差
- **RMSE**: 均方根误差
- **R²**: 决定系数

## 特征说明

### 原始特征
- 孕周数：孕妇的孕周
- BMI：身体质量指数
- 年龄：孕妇年龄

### 交互特征
- 孕周数_BMI：孕周数与BMI的交互项
- BMI_年龄：BMI与年龄的交互项
- 孕周数_年龄：孕周数与年龄的交互项

## 注意事项

1. 确保已安装conda并创建了pytorch环境
2. 数据文件路径为`../step1/processed_data.csv`
3. 代码会自动处理中文字体显示问题
4. 所有图表和结果文件会保存在当前目录
