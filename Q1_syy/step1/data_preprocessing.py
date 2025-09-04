#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
数据预处理脚本
根据step1.md中的要求处理孕妇检测数据
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

def load_data(file_path='../appendix.xlsx'):
    """
    加载数据
    """
    print("正在加载数据...")
    data = pd.read_excel(file_path)
    print(f"原始数据形状: {data.shape}")
    print(f"列名: {data.columns.tolist()}")
    return data

def filter_male_fetus_data(data):
    """
    1. 筛选男胎数据：只保留有Y染色体浓度的记录
    """
    print("\n=== 步骤1: 筛选男胎数据 ===")
    
    # 筛选有Y染色体浓度的数据
    male_data = data[data['Y染色体浓度'].notnull()].copy()
    
    print(f"筛选前数据量: {len(data)}")
    print(f"筛选后数据量: {len(male_data)}")
    print(f"男胎数据占比: {len(male_data)/len(data)*100:.2f}%")
    
    return male_data

def clean_data(data):
    """
    2. 清洗数据：去除缺失值和异常值
    """
    print("\n=== 步骤2: 清洗数据 ===")
    
    # 去除关键特征的缺失值
    key_columns = ['Y染色体浓度', '检测孕周', '孕妇BMI']
    data_cleaned = data.dropna(subset=key_columns).copy()
    
    print(f"去除缺失值前: {len(data)} 条记录")
    print(f"去除缺失值后: {len(data_cleaned)} 条记录")
    
    # 检查异常值
    print("\n数据范围检查:")
    print(f"Y染色体浓度范围: {data_cleaned['Y染色体浓度'].min():.6f} - {data_cleaned['Y染色体浓度'].max():.6f}")
    print(f"BMI范围: {data_cleaned['孕妇BMI'].min():.2f} - {data_cleaned['孕妇BMI'].max():.2f}")
    
    # 去除异常值（可根据实际情况调整阈值）
    # BMI异常值：通常BMI在15-50之间是合理的
    bmi_mask = (data_cleaned['孕妇BMI'] >= 15) & (data_cleaned['孕妇BMI'] <= 50)
    data_cleaned = data_cleaned[bmi_mask]
    
    # Y染色体浓度异常值：去除负值和过大的值
    y_chrom_mask = (data_cleaned['Y染色体浓度'] > 0) & (data_cleaned['Y染色体浓度'] <= 1)
    data_cleaned = data_cleaned[y_chrom_mask]
    
    print(f"去除异常值后: {len(data_cleaned)} 条记录")
    
    return data_cleaned

def convert_gestational_weeks(data):
    """
    3. 转换孕周格式：从"w+d"格式转换为以周为单位的数值
    """
    print("\n=== 步骤3: 转换孕周格式 ===")
    
    def week_to_numeric(week_str):
        """
        将孕周从"w+d"格式转换为以周为单位的数值
        例如："12w+3" 转为 12.43
        """
        if pd.isna(week_str):
            return np.nan
        
        try:
            # 处理不同的格式
            if 'w+' in str(week_str):
                weeks, days = str(week_str).split('w+')
                return int(weeks) + int(days) / 7
            elif 'w' in str(week_str):
                # 处理只有周数的情况，如"12w"
                weeks = str(week_str).replace('w', '')
                return int(weeks)
            else:
                # 如果已经是数值，直接返回
                return float(week_str)
        except:
            return np.nan
    
    # 转换孕周格式
    data['检测孕周_数值'] = data['检测孕周'].apply(week_to_numeric)
    
    # 检查转换结果
    print("孕周转换示例:")
    sample_data = data[['检测孕周', '检测孕周_数值']].head(10)
    print(sample_data)
    
    # 去除转换失败的记录
    data = data.dropna(subset=['检测孕周_数值']).copy()
    print(f"孕周转换后数据量: {len(data)} 条记录")
    
    return data

def handle_duplicate_tests(data):
    """
    4. 处理同一天检测的重复数据
    """
    print("\n=== 步骤4: 处理同一天检测的重复数据 ===")
    
    # 转换检测日期格式
    data['检测日期'] = pd.to_datetime(data['检测日期'], format='%Y%m%d', errors='coerce')
    
    # 检查重复数据
    duplicate_check = data.groupby(['孕妇代码', '检测日期']).size()
    duplicates = duplicate_check[duplicate_check > 1]
    
    print(f"发现 {len(duplicates)} 个孕妇在同一天进行了多次检测")
    if len(duplicates) > 0:
        print("重复检测示例:")
        print(duplicates.head())
    
    # 按孕妇代码和检测日期分组，计算均值
    data_grouped = data.groupby(['孕妇代码', '检测日期']).agg({
        'Y染色体浓度': ['mean', 'std', 'count'],
        '孕妇BMI': 'mean',
        '检测孕周_数值': 'mean',
        '年龄': 'first',  # 年龄通常不变，取第一个值
        '身高': 'first',  # 身高通常不变，取第一个值
        '体重': 'mean',   # 体重可能有变化，取均值
    }).reset_index()
    
    # 展平列名
    data_grouped.columns = ['孕妇代码', '检测日期', 'Y染色体浓度_均值', 'Y染色体浓度_标准差', 
                           'Y染色体浓度_次数', 'BMI_均值', '孕周_均值', '年龄', '身高', '体重_均值']
    
    print(f"分组聚合后数据量: {len(data_grouped)} 条记录")
    
    # 检查标准差，标记可能的异常数据
    high_std_mask = data_grouped['Y染色体浓度_标准差'] > 0.01  # 标准差阈值可调整
    if high_std_mask.sum() > 0:
        print(f"发现 {high_std_mask.sum()} 条记录的同一天检测差异较大（标准差>0.01）")
        print("这些记录可能需要进一步检查:")
        print(data_grouped[high_std_mask][['孕妇代码', '检测日期', 'Y染色体浓度_均值', 'Y染色体浓度_标准差']])
    
    return data_grouped

def extract_features(data):
    """
    5. 提取特征：选择孕周数、BMI作为自变量，Y染色体浓度作为因变量
    """
    print("\n=== 步骤5: 提取特征 ===")
    
    # 重命名列以便后续使用
    feature_data = data.rename(columns={
        '孕周_均值': '孕周数',
        'BMI_均值': 'BMI',
        'Y染色体浓度_均值': 'Y染色体浓度'
    }).copy()
    
    # 提取特征和目标变量
    X = feature_data[['孕周数', 'BMI']].copy()
    y = feature_data['Y染色体浓度'].copy()
    
    print(f"特征矩阵形状: {X.shape}")
    print(f"目标变量形状: {y.shape}")
    print(f"特征列: {X.columns.tolist()}")
    
    # 基本统计信息
    print("\n特征统计信息:")
    print(X.describe())
    print("\n目标变量统计信息:")
    print(y.describe())
    
    return X, y, feature_data

def visualize_data(data, X, y):
    """
    数据可视化
    """
    print("\n=== 数据可视化 ===")
    
    # 创建子图
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('数据预处理结果可视化', fontsize=16)
    
    # 1. Y染色体浓度分布
    axes[0, 0].hist(y, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0, 0].set_title('Y染色体浓度分布')
    axes[0, 0].set_xlabel('Y染色体浓度')
    axes[0, 0].set_ylabel('频数')
    
    # 2. 孕周数分布
    axes[0, 1].hist(X['孕周数'], bins=20, alpha=0.7, color='lightgreen', edgecolor='black')
    axes[0, 1].set_title('孕周数分布')
    axes[0, 1].set_xlabel('孕周数')
    axes[0, 1].set_ylabel('频数')
    
    # 3. BMI分布
    axes[0, 2].hist(X['BMI'], bins=20, alpha=0.7, color='lightcoral', edgecolor='black')
    axes[0, 2].set_title('BMI分布')
    axes[0, 2].set_xlabel('BMI')
    axes[0, 2].set_ylabel('频数')
    
    # 4. 孕周数 vs Y染色体浓度
    axes[1, 0].scatter(X['孕周数'], y, alpha=0.6, color='blue')
    axes[1, 0].set_title('孕周数 vs Y染色体浓度')
    axes[1, 0].set_xlabel('孕周数')
    axes[1, 0].set_ylabel('Y染色体浓度')
    
    # 5. BMI vs Y染色体浓度
    axes[1, 1].scatter(X['BMI'], y, alpha=0.6, color='red')
    axes[1, 1].set_title('BMI vs Y染色体浓度')
    axes[1, 1].set_xlabel('BMI')
    axes[1, 1].set_ylabel('Y染色体浓度')
    
    # 6. 相关性热力图
    corr_data = pd.concat([X, y], axis=1)
    corr_matrix = corr_data.corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=axes[1, 2])
    axes[1, 2].set_title('特征相关性热力图')
    
    plt.tight_layout()
    plt.savefig('data_preprocessing_visualization.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("可视化图表已保存为 'data_preprocessing_visualization.png'")

def save_processed_data(X, y, feature_data):
    """
    保存处理后的数据
    """
    print("\n=== 保存处理后的数据 ===")
    
    # 保存特征和目标变量
    X.to_csv('X_features.csv', index=False)
    y.to_csv('y_target.csv', index=False)
    feature_data.to_csv('processed_data.csv', index=False)
    
    # 保存为numpy格式
    np.save('X_features.npy', X.values)
    np.save('y_target.npy', y.values)
    
    print("数据已保存:")
    print("- X_features.csv: 特征数据")
    print("- y_target.csv: 目标变量")
    print("- processed_data.csv: 完整处理后的数据")
    print("- X_features.npy: 特征数据(numpy格式)")
    print("- y_target.npy: 目标变量(numpy格式)")

def main():
    """
    主函数：执行完整的数据预处理流程
    """
    print("开始数据预处理...")
    print("=" * 50)
    
    try:
        # 1. 加载数据
        data = load_data('../appendix.xlsx')
        
        # 2. 筛选男胎数据
        male_data = filter_male_fetus_data(data)
        
        # 3. 清洗数据
        cleaned_data = clean_data(male_data)
        
        # 4. 转换孕周格式
        converted_data = convert_gestational_weeks(cleaned_data)
        
        # 5. 处理重复数据
        grouped_data = handle_duplicate_tests(converted_data)
        
        # 6. 提取特征
        X, y, feature_data = extract_features(grouped_data)
        
        # 7. 数据可视化
        visualize_data(feature_data, X, y)
        
        # 8. 保存处理后的数据
        save_processed_data(X, y, feature_data)
        
        print("\n" + "=" * 50)
        print("数据预处理完成！")
        print(f"最终数据量: {len(feature_data)} 条记录")
        print(f"特征维度: {X.shape[1]} 维")
        
    except Exception as e:
        print(f"数据处理过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
