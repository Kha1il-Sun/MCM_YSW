# SVR模型训练

本目录包含使用支持向量回归（SVR）进行Y染色体浓度预测的完整代码。

## 文件说明

- `svr_model.py`: 主要的SVR模型训练脚本
- `requirements.txt`: Python依赖包列表
- `run_svr.bat`: Windows批处理运行脚本
- `run_svr.py`: Python运行脚本
- `SVR.md`: 详细的SVR模型说明文档

## 使用方法

### 方法1: 使用批处理脚本（推荐）
```bash
run_svr.bat
```

### 方法2: 使用Python脚本
```bash
python run_svr.py
```

### 方法3: 直接运行
```bash
# 激活pytorch环境
conda activate pytorch

# 安装依赖
pip install -r requirements.txt

# 运行模型训练
python svr_model.py
```

## 输出文件

运行完成后会生成以下文件：
- `svr_results.txt`: 模型训练结果和性能指标
- `svr_analysis.png`: 模型预测结果可视化图表

## 模型说明

本脚本实现了两种SVR模型：
1. **基本SVR模型**: 使用默认参数的RBF核SVR
2. **优化SVR模型**: 使用GridSearchCV进行参数优化的SVR

## 评估指标

- MSE (均方误差): 越小越好
- RMSE (均方根误差): 越小越好  
- R² (决定系数): 越接近1越好

## 数据要求

确保以下数据文件存在于 `../step1/` 目录中：
- `X_features.csv`: 特征数据（孕周数、BMI等）
- `y_target.csv`: 目标变量（Y染色体浓度）

