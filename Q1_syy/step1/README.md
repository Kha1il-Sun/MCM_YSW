# 数据预处理脚本

本目录包含用于处理孕妇检测数据的Python脚本，根据step1.md中的要求实现完整的数据预处理流程。

## 文件说明

- `data_preprocessing.py` - 主要的数据预处理脚本
- `run_preprocessing.py` - 运行脚本，包含环境检查
- `run_preprocessing.bat` - Windows批处理文件，方便运行
- `requirements.txt` - 依赖包列表
- `step1.md` - 原始需求说明文件
- `../appendix.xlsx` - 原始数据文件（位于上级目录）

## 运行前准备

### 1. 激活pytorch环境
```bash
conda activate pytorch
```

### 2. 安装依赖包
```bash
pip install -r requirements.txt
```

## 运行方式

### 方式1：使用Python脚本（推荐）
```bash
python run_preprocessing.py
```

### 方式2：使用批处理文件（Windows）
双击 `run_preprocessing.bat` 文件

### 方式3：直接运行主脚本
```bash
python data_preprocessing.py
```

## 数据处理流程

脚本按照以下步骤处理数据：

1. **筛选男胎数据** - 只保留有Y染色体浓度的记录
2. **清洗数据** - 去除缺失值和异常值
3. **转换孕周格式** - 将"w+d"格式转换为数值
4. **处理重复数据** - 对同一天多次检测的数据取平均值
5. **提取特征** - 选择孕周数和BMI作为自变量，Y染色体浓度作为因变量
6. **数据可视化** - 生成分布图和相关性分析
7. **保存结果** - 输出处理后的数据文件

## 输出文件

- `X_features.csv` - 特征数据（孕周数、BMI）
- `y_target.csv` - 目标变量（Y染色体浓度）
- `processed_data.csv` - 完整处理后的数据
- `X_features.npy` - 特征数据（numpy格式）
- `y_target.npy` - 目标变量（numpy格式）
- `data_preprocessing_visualization.png` - 数据可视化图表

## 处理结果

- 原始数据：1082条记录
- 最终数据：1062条记录
- 特征维度：2维（孕周数、BMI）
- 发现19个孕妇在同一天进行了多次检测，已通过取平均值处理

## 注意事项

1. 确保在pytorch环境中运行
2. 原始数据文件`appendix.xlsx`必须存在于同一目录下
3. 脚本会自动检查环境依赖并给出提示
4. 生成的可视化图表会保存为PNG文件

## 错误处理

如果遇到问题：
1. 检查是否已激活pytorch环境
2. 确认所有依赖包已安装
3. 检查原始数据文件是否存在
4. 查看控制台输出的错误信息
