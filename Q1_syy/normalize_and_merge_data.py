#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据归一化处理并合并脚本
对X_features.csv和y_target.csv中的数据进行归一化处理，然后合并到一个CSV文件中
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os

def normalize_and_merge_data():
    """
    读取数据，进行归一化处理，然后合并保存
    """
    # 设置文件路径
    x_file = 'step1/X_features.csv'
    y_file = 'step1/y_target.csv'
    output_file = 'step1/normalized_merged_data.csv'
    
    print("开始读取数据...")
    
    # 读取特征数据
    X_data = pd.read_csv(x_file)
    print(f"特征数据形状: {X_data.shape}")
    print(f"特征列名: {X_data.columns.tolist()}")
    
    # 读取目标变量数据
    y_data = pd.read_csv(y_file)
    print(f"目标变量数据形状: {y_data.shape}")
    print(f"目标变量列名: {y_data.columns.tolist()}")
    
    # 检查数据长度是否一致
    if len(X_data) != len(y_data):
        print(f"警告: 特征数据长度({len(X_data)})与目标变量长度({len(y_data)})不一致!")
        # 取较小的长度
        min_len = min(len(X_data), len(y_data))
        X_data = X_data.iloc[:min_len]
        y_data = y_data.iloc[:min_len]
        print(f"已调整数据长度为: {min_len}")
    
    print("\n开始归一化处理...")
    
    # 创建归一化器
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    
    # 对特征数据进行归一化
    X_normalized = scaler_X.fit_transform(X_data)
    X_normalized_df = pd.DataFrame(X_normalized, columns=X_data.columns)
    
    # 对目标变量进行归一化
    y_normalized = scaler_y.fit_transform(y_data)
    y_normalized_df = pd.DataFrame(y_normalized, columns=y_data.columns)
    
    print("归一化完成!")
    print(f"特征数据归一化后范围: [{X_normalized.min():.6f}, {X_normalized.max():.6f}]")
    print(f"目标变量归一化后范围: [{y_normalized.min():.6f}, {y_normalized.max():.6f}]")
    
    # 合并归一化后的数据
    print("\n开始合并数据...")
    merged_data = pd.concat([X_normalized_df, y_normalized_df], axis=1)
    
    print(f"合并后数据形状: {merged_data.shape}")
    print(f"合并后列名: {merged_data.columns.tolist()}")
    
    # 保存合并后的数据
    print(f"\n保存数据到: {output_file}")
    merged_data.to_csv(output_file, index=False, encoding='utf-8')
    
    # 显示前几行数据
    print("\n合并后数据预览:")
    print(merged_data.head(10))
    
    # 显示数据统计信息
    print("\n数据统计信息:")
    print(merged_data.describe())
    
    # 保存归一化参数信息
    scaler_info = {
        'X_features_min': scaler_X.data_min_.tolist(),
        'X_features_max': scaler_X.data_max_.tolist(),
        'y_target_min': scaler_y.data_min_.tolist(),
        'y_target_max': scaler_y.data_max_.tolist(),
        'feature_columns': X_data.columns.tolist(),
        'target_columns': y_data.columns.tolist()
    }
    
    # 保存归一化参数到文件
    import json
    with open('step1/normalization_params.json', 'w', encoding='utf-8') as f:
        json.dump(scaler_info, f, ensure_ascii=False, indent=2)
    
    print(f"\n归一化参数已保存到: step1/normalization_params.json")
    print("数据处理完成!")
    
    return merged_data, scaler_info

if __name__ == "__main__":
    try:
        merged_data, scaler_info = normalize_and_merge_data()
        print("\n✅ 数据归一化和合并处理成功完成!")
    except Exception as e:
        print(f"\n❌ 处理过程中出现错误: {str(e)}")
        import traceback
        traceback.print_exc()
