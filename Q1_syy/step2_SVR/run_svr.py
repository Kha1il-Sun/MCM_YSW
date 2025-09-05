#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SVR模型运行脚本
"""

import subprocess
import sys
import os

def run_command(command, description):
    """运行命令并处理错误"""
    print(f"\n{description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✓ {description} 成功")
        if result.stdout:
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ {description} 失败")
        print(f"错误信息: {e.stderr}")
        return False

def main():
    """主函数"""
    print("SVR模型训练脚本")
    print("="*50)
    
    # 检查当前目录
    current_dir = os.getcwd()
    print(f"当前工作目录: {current_dir}")
    
    # 检查必要文件是否存在
    required_files = ['svr_model.py', 'requirements.txt']
    for file in required_files:
        if not os.path.exists(file):
            print(f"错误: 找不到文件 {file}")
            return False
    
    # 安装依赖
    if not run_command("pip install -r requirements.txt", "安装依赖包"):
        return False
    
    # 运行SVR模型
    if not run_command("python svr_model.py", "运行SVR模型训练"):
        return False
    
    print("\n" + "="*50)
    print("SVR模型训练完成!")
    print("="*50)
    
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)

