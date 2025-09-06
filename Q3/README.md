# MCM Q3: 区间删失AFT + 多重插补 + 风险最小化 + BMI分段优化

本项目实现了基于AFT模型的生存分析管道，包含多重插补、风险最小化和BMI分段优化。

## 核心功能

- **区间删失AFT模型**：支持log-normal、log-logistic、Weibull三种分布族
- **多重插补(MI)**：处理观测误差到区间删失数据的转换
- **风险最小化**：寻找最优推荐时点w*
- **BMI分段优化**：动态规划或CART方法进行分组
- **模型验证**：交叉验证、Bootstrap、敏感性分析

## 快速开始

```bash
# 安装依赖
pip install -r requirements.txt

# 运行完整管道
python scripts/run_pipeline.py --config configs/q3_default.yaml

# 运行敏感性分析
python scripts/run_sigma_sweep.py

# 运行分组分析
python scripts/run_grouping_dp.py
```

## 项目结构

```
Q3/
├── configs/          # 配置文件
├── data/            # 数据文件和schema
├── src/             # 核心代码模块
├── scripts/         # 运行脚本
├── outputs/         # 输出结果
├── logs/           # 日志文件
├── reports/        # 分析报告
└── tests/          # 测试代码
```

## 数学原理

本项目基于AFT(Accelerated Failure Time)模型：

$$\ln T^* = \mathbf{x}^T\beta + \sigma_\varepsilon \varepsilon$$

其中$T^*$为首次达标时间，$\mathbf{x}$为协变量向量。

详细的数学公式和实现原理请参考源码注释和技术文档。

## 配置说明

主要配置文件为`configs/q3_default.yaml`，包含：
- MI参数：插补次数M、误差率q等
- AFT参数：分布族选择、集成方法等  
- 分组参数：分组数K、最小组大小等
- 性能参数：并行数、内存限制等

## 输出说明

- `outputs/curves/`：生存曲线和风险曲线
- `outputs/tables/`：统计结果表格
- `outputs/figures/`：可视化图表
- `outputs/models/`：训练好的模型
- `reports/`：分析报告

## 开发说明

本项目采用模块化设计，主要模块包括：
- `io_utils.py`：数据读写和校验
- `mi_interval_imputer.py`：多重插补
- `aft_models.py`：AFT模型拟合
- `risk_objective.py`：风险函数优化
- `grouping.py`：分段优化算法

## 测试

```bash
# 运行所有测试
pytest tests/

# 运行单元测试
pytest tests/unit/

# 运行集成测试  
pytest tests/integration/
```

## 容器化

```bash
# 构建镜像
docker build -t mcm-q3 .

# 运行容器
docker run -v $(pwd)/data:/app/data -v $(pwd)/outputs:/app/outputs mcm-q3
```