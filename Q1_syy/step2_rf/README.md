# 随机森林回归模型

本项目使用随机森林回归模型来预测Y染色体浓度，基于孕周数和BMI特征。

## 文件说明

- `rf_model.py`: 主要的随机森林模型训练脚本
- `requirements.txt`: 项目依赖包列表
- `run_rf.bat`: Windows批处理运行脚本
- `rf_results.txt`: 模型训练结果（运行后生成）
- `rf_feature_importance.png`: 特征重要性图（运行后生成）
- `rf_predictions.png`: 预测结果对比图（运行后生成）

## 环境要求

- Python 3.7+
- 依赖包见 `requirements.txt`

## 安装依赖

```bash
pip install -r requirements.txt
```

## 运行方法

### 方法1：使用批处理文件（推荐）
```bash
run_rf.bat
```

### 方法2：直接运行Python脚本
```bash
python rf_model.py
```

## 模型特点

1. **数据分割**: 80%训练数据，20%测试数据
2. **基础模型**: 100棵树，最大深度6
3. **模型优化**: 使用网格搜索优化超参数
4. **评估指标**: MSE、RMSE、R²
5. **特征分析**: 特征重要性排序和可视化

## 输出结果

运行完成后会生成以下文件：

1. **rf_results.txt**: 包含详细的模型评估结果和特征重要性
2. **rf_feature_importance.png**: 特征重要性柱状图
3. **rf_predictions.png**: 包含三个子图：
   - 原始模型预测 vs 实际值
   - 优化模型预测 vs 实际值  
   - 残差图

## 模型调优

代码包含网格搜索优化，会尝试以下参数组合：
- n_estimators: [100, 200, 300]
- max_depth: [6, 8, 10]
- min_samples_split: [2, 4, 6]

## 注意事项

- 确保 `../step1/X_features.csv` 和 `../step1/y_target.csv` 文件存在
- 运行前请激活pytorch环境：`conda activate pytorch`
- 模型训练可能需要几分钟时间，请耐心等待
