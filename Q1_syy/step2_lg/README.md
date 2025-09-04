# 线性回归模型：Y染色体浓度分析

## 项目描述
本项目使用线性回归模型分析Y染色体浓度与孕周数、BMI之间的关系，基于step1预处理的数据进行建模和评估。

## 文件说明
- `linear_regression_model.py`: 主要的线性回归模型代码
- `run_linear_regression.bat`: Windows批处理脚本，自动激活pytorch环境并运行模型
- `requirements.txt`: Python依赖包列表
- `2.md`: 原始说明文档

## 运行方法

### 方法1：使用批处理脚本（推荐）
1. 双击运行 `run_linear_regression.bat`
2. 脚本会自动激活pytorch环境并运行模型

### 方法2：手动运行
1. 打开命令行
2. 激活conda环境：`conda activate pytorch`
3. 运行模型：`python linear_regression_model.py`

## 输出结果
运行完成后会生成以下文件：
- `linear_regression_analysis.png`: 模型分析可视化图表
- `linear_regression_results.txt`: 详细的模型结果和评估指标

## 模型特点
- 使用孕周数和BMI作为特征预测Y染色体浓度
- 80%数据用于训练，20%用于测试
- 提供完整的模型评估指标（MSE、RMSE、MAE、R²）
- 包含特征重要性分析和结果可视化

## 依赖环境
- Python 3.7+
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn

## 注意事项
- 确保step1的数据预处理已完成
- 需要正确安装conda和pytorch环境
- 运行前请确保所有依赖包已安装
