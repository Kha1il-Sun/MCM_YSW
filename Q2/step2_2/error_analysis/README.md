# 检测误差影响分析

## 概述

本模块分析不同σ（检测误差）水平对BMI分组和时点推荐的影响，回答论文中的"检测误差对结果的影响"问题。

## 功能特性

- **σ情景设定**: 支持0.5×, 0.75×, 1.0×, 1.25×, 1.5×倍数的σ分析
- **时点推荐计算**: 基于`EmpiricalDetectionModel`计算不同σ下的推荐时点
- **风险函数修正**: 考虑σ对风险函数的影响
- **敏感性分析**: 量化各BMI组对σ变化的敏感性
- **可视化输出**: 生成4种分析图表
- **对比表格**: 生成详细的数值对比表

## 文件结构

```
error_analysis/
├── src/
│   └── error_impact_analyzer.py    # 主分析器
├── outputs/                        # 输出结果目录
├── run_error_analysis.py          # Python运行脚本
├── run_error_analysis.bat         # Windows批处理脚本
└── README.md                      # 说明文档
```

## 使用方法

### 方法1: 使用批处理脚本（推荐）
```bash
# 在error_analysis目录下运行
run_error_analysis.bat
```

### 方法2: 使用Python脚本
```bash
# 在error_analysis目录下运行
python run_error_analysis.py --data-dir ../../step2_1 --config ../../config/step2_config.yaml
```

### 方法3: 在代码中调用
```python
from src.error_impact_analyzer import ErrorImpactAnalyzer

# 创建分析器
analyzer = ErrorImpactAnalyzer()

# 加载数据和模型
analyzer.load_data_and_model()

# 运行分析
results = analyzer.run_error_impact_analysis()

# 生成报告
report = analyzer.generate_summary_report()
```

## 输出结果

### 1. 数值结果
- `error_impact_comparison.csv`: 推荐时点对比表
- `error_impact_report.txt`: 详细分析报告

### 2. 可视化结果
- `time_sensitivity.png`: σ倍数 vs 推荐时点图
- `risk_sensitivity.png`: σ倍数 vs 风险值图
- `sensitivity_coefficients.png`: 敏感性系数对比图
- `time_change_range.png`: 时点变化范围图

### 3. 分析内容
- **时点敏感性**: σ倍数 vs 推荐时点（单独图表）
- **风险敏感性**: σ倍数 vs 风险值（单独图表）
- **敏感性系数**: 各BMI组的敏感性对比（单独图表）
- **变化范围**: 时点变化范围分析（单独图表）

## 核心算法

### 1. σ情景设定
```python
sigma_multipliers = [0.5, 0.75, 1.0, 1.25, 1.5]
sigma = baseline_sigma * multiplier
sigma = max(sigma, 0.001)  # 设置下限避免除零
```

### 2. 时点推荐计算
```python
optimal_time = empirical_model.predict_optimal_time(bmi_median, sigma)
```

### 3. 风险函数修正
```python
# 考虑σ的成功概率
success_prob = 0.5 + 0.5 * np.tanh((optimal_time - 12) / 2)
early_risk = 1 - success_prob
delay_risk = max(0, (optimal_time - 15) / 10)
total_risk = early_risk + 0.3 * delay_risk
```

### 4. 敏感性分析
```python
time_sensitivity = np.std(times) / np.mean(times)
risk_sensitivity = np.std(risks) / np.mean(risks)
max_time_change = max(times) - min(times)
```

## 配置参数

### 默认配置
- **σ倍数范围**: [0.5, 0.75, 1.0, 1.25, 1.5]
- **BMI分组**: [20.0, 30.5, 32.7, 34.4, 50.0]
- **σ下限**: 0.001（避免除零）

### 自定义配置
可以通过修改`run_error_analysis.py`中的参数来自定义分析设置。

## 结果解释

### 1. 时点变化
- **变化范围**: 各BMI组在不同σ下的时点变化范围
- **敏感性**: 时点对σ变化的敏感程度

### 2. 风险变化
- **风险值**: 考虑σ后的风险函数值
- **变化比例**: 相对于基准风险的变化比例

### 3. 稳健性评估
- **最大变化**: 所有σ倍数下的最大时点变化
- **敏感性系数**: 量化的敏感性指标

## 注意事项

1. **σ下限设置**: 为避免除零错误，σ最小值设为0.001
2. **数据依赖**: 需要Step1的输出数据（long_records.csv等）
3. **模型依赖**: 需要`EmpiricalDetectionModel`类
4. **输出目录**: 确保`outputs`目录存在且有写入权限

## 故障排除

### 常见问题
1. **数据加载失败**: 检查数据路径是否正确
2. **模型导入失败**: 确保`empirical_model.py`在正确位置
3. **输出权限错误**: 检查`outputs`目录权限

### 调试模式
在代码中添加`print`语句来查看中间结果：
```python
print(f"当前σ: {sigma}")
print(f"推荐时点: {optimal_time}")
```

## 扩展功能

### 1. 添加新的σ倍数
```python
analyzer.sigma_multipliers = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
```

### 2. 自定义风险函数
```python
def custom_risk_function(self, bmi, sigma):
    # 实现自定义风险函数
    pass
```

### 3. 添加新的可视化
```python
def plot_custom_analysis(self, ax, data):
    # 实现自定义图表
    pass
```

---

**版本**: 1.0.0  
**作者**: MCM Team  
**更新时间**: 2024-01-XX
