#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
运行数据预处理脚本
确保在pytorch环境中运行
"""

import sys
import os
import subprocess

def check_environment():
    """
    检查当前环境
    """
    print("检查当前环境...")
    print(f"Python版本: {sys.version}")
    print(f"当前工作目录: {os.getcwd()}")
    
    # 检查是否在pytorch环境中
    try:
        import torch
        print(f"PyTorch版本: {torch.__version__}")
        print("✓ PyTorch环境已激活")
    except ImportError:
        print("✗ PyTorch未安装或环境未激活")
        print("请运行: conda activate pytorch")
        return False
    
    # 检查必要的包
    required_packages = ['pandas', 'numpy', 'matplotlib', 'seaborn', 'openpyxl']
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✓ {package} 已安装")
        except ImportError:
            missing_packages.append(package)
            print(f"✗ {package} 未安装")
    
    if missing_packages:
        print(f"\n缺少以下包: {missing_packages}")
        print("请安装缺少的包:")
        for package in missing_packages:
            if package == 'openpyxl':
                print(f"  pip install {package}")
            else:
                print(f"  conda install {package}")
        return False
    
    return True

def main():
    """
    主函数
    """
    print("=" * 60)
    print("数据预处理运行脚本")
    print("=" * 60)
    
    # 检查环境
    if not check_environment():
        print("\n环境检查失败，请先解决环境问题")
        return
    
    print("\n环境检查通过，开始运行数据预处理...")
    print("-" * 60)
    
    try:
        # 导入并运行数据预处理脚本
        from data_preprocessing import main as run_preprocessing
        run_preprocessing()
        
    except Exception as e:
        print(f"运行过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
