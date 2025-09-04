#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Y染色体浓度预测系统 - 示例运行脚本
演示如何使用不同的参数配置运行模型
"""

import os
import sys
import subprocess

def run_command(cmd, description):
    """运行命令并显示结果"""
    print(f"\n{'='*60}")
    print(f"运行: {description}")
    print(f"命令: {cmd}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ 执行成功！")
            if result.stdout:
                print("输出:")
                print(result.stdout)
        else:
            print("❌ 执行失败！")
            if result.stderr:
                print("错误信息:")
                print(result.stderr)
    except Exception as e:
        print(f"❌ 执行异常: {e}")

def main():
    """主函数 - 演示不同的运行方式"""
    
    print("Y染色体浓度预测系统 - 示例运行")
    print("本脚本将演示不同的运行方式")
    
    # 检查是否在正确的目录
    if not os.path.exists('appendix.xlsx'):
        print("❌ 错误：找不到数据文件 appendix.xlsx")
        print("请确保在Q1目录下运行此脚本")
        return
    
    # 示例1：快速开始（使用默认参数）
    print("\n示例1：快速开始模式")
    run_command("python main.py", "使用默认参数运行完整流程")
    
    # 示例2：仅数据预处理
    print("\n示例2：仅数据预处理")
    run_command("python main.py --mode preprocess", "仅进行数据预处理")
    
    # 示例3：自定义模型结构
    print("\n示例3：自定义模型结构")
    run_command("python main.py --mode full --hidden_layers 128 64 32 16 --epochs 150", 
                "使用更深的网络结构和较少的训练轮数")
    
    # 示例4：高批次大小训练
    print("\n示例4：高批次大小训练")
    run_command("python main.py --mode train --batch_size 64 --epochs 100", 
                "使用较大的批次大小进行训练")
    
    # 示例5：低Dropout率
    print("\n示例5：低Dropout率")
    run_command("python main.py --mode train --dropout_rate 0.1 --epochs 250", 
                "使用较低的Dropout率")
    
    print(f"\n{'='*60}")
    print("所有示例运行完成！")
    print("请查看生成的文件：")
    print("- training_history.png: 训练历史图表")
    print("- predictions.png: 预测结果对比图")
    print("- evaluation_results.txt: 模型评估结果")
    print("- y_chromosome_model.h5: 训练好的模型")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
