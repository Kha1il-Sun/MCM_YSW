#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Y染色体浓度预测主程序
使用深度学习分析Y染色体浓度与其他指标之间的关系

作者：MCM团队
日期：2025年
"""

import os
import sys
import argparse
import warnings
warnings.filterwarnings('ignore')

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_preprocessing import load_and_preprocess_data, save_preprocessed_data
from model_building import YChromosomePredictor, load_preprocessed_data

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='Y染色体浓度预测系统')
    parser.add_argument('--data_file', type=str, default='appendix.xlsx',
                       help='数据文件路径 (默认: appendix.xlsx)')
    parser.add_argument('--mode', type=str, choices=['preprocess', 'train', 'full'],
                       default='full', help='运行模式: preprocess(仅预处理), train(仅训练), full(完整流程)')
    parser.add_argument('--epochs', type=int, default=200,
                       help='训练轮数 (默认: 200)')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='批次大小 (默认: 32)')
    parser.add_argument('--hidden_layers', type=int, nargs='+', default=[64, 32, 16],
                       help='隐藏层神经元数量 (默认: [64, 32, 16])')
    parser.add_argument('--dropout_rate', type=float, default=0.2,
                       help='Dropout比率 (默认: 0.2)')
    parser.add_argument('--output_dir', type=str, default='.',
                       help='输出目录 (默认: 当前目录)')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Y染色体浓度预测系统")
    print("=" * 60)
    print(f"数据文件: {args.data_file}")
    print(f"运行模式: {args.mode}")
    print(f"输出目录: {args.output_dir}")
    print("=" * 60)
    
    # 检查数据文件是否存在
    if not os.path.exists(args.data_file):
        print(f"错误：找不到数据文件 {args.data_file}")
        return
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 步骤1：数据预处理
    if args.mode in ['preprocess', 'full']:
        print("\n步骤1：数据预处理")
        print("-" * 40)
        
        X_train, X_test, y_train, y_test, scaler, feature_columns = load_and_preprocess_data(args.data_file)
        
        if X_train is None:
            print("数据预处理失败，程序退出！")
            return
        
        # 保存预处理后的数据
        save_preprocessed_data(X_train, X_test, y_train, y_test, 
                             feature_columns, scaler, args.output_dir)
        print("数据预处理完成！")
    
    # 步骤2：模型训练
    if args.mode in ['train', 'full']:
        print("\n步骤2：模型训练")
        print("-" * 40)
        
        # 加载预处理后的数据
        X_train, X_test, y_train, y_test, scaler, feature_columns = load_preprocessed_data(args.output_dir)
        
        if X_train is None:
            print("无法加载预处理数据，请先运行数据预处理！")
            return
        
        # 创建模型实例
        predictor = YChromosomePredictor()
        
        # 构建模型
        print("构建神经网络模型...")
        predictor.build_model(
            input_dim=X_train.shape[1],
            hidden_layers=args.hidden_layers,
            dropout_rate=args.dropout_rate
        )
        
        # 训练模型
        print("开始训练模型...")
        predictor.train_model(
            X_train, y_train, X_test, y_test,
            epochs=args.epochs,
            batch_size=args.batch_size
        )
        
        # 评估模型
        print("\n模型评估：")
        metrics, y_pred = predictor.evaluate_model(X_test, y_test)
        
        # 绘制训练历史
        history_path = os.path.join(args.output_dir, 'training_history.png')
        predictor.plot_training_history(history_path)
        
        # 绘制预测结果
        predictions_path = os.path.join(args.output_dir, 'predictions.png')
        predictor.plot_predictions(y_test, y_pred, predictions_path)
        
        # 保存模型
        predictor.scaler = scaler
        predictor.feature_columns = feature_columns
        model_path = os.path.join(args.output_dir, 'y_chromosome_model.h5')
        scaler_path = os.path.join(args.output_dir, 'scaler.pkl')
        features_path = os.path.join(args.output_dir, 'feature_columns.txt')
        
        predictor.save_model(model_path, scaler_path, features_path)
        
        # 保存评估结果
        results_path = os.path.join(args.output_dir, 'evaluation_results.txt')
        with open(results_path, 'w', encoding='utf-8') as f:
            f.write("Y染色体浓度预测模型评估结果\n")
            f.write("=" * 40 + "\n")
            for metric, value in metrics.items():
                f.write(f"{metric}: {value:.6f}\n")
            f.write(f"\n模型参数：\n")
            f.write(f"隐藏层: {args.hidden_layers}\n")
            f.write(f"Dropout比率: {args.dropout_rate}\n")
            f.write(f"训练轮数: {args.epochs}\n")
            f.write(f"批次大小: {args.batch_size}\n")
        
        print(f"评估结果已保存到：{results_path}")
        print("模型训练完成！")
    
    print("\n" + "=" * 60)
    print("程序执行完成！")
    print("=" * 60)

def quick_start():
    """快速开始函数，使用默认参数"""
    print("快速开始模式 - 使用默认参数")
    
    # 检查是否存在预处理数据
    if os.path.exists('X_train.npy'):
        print("发现预处理数据，直接开始训练...")
        args = argparse.Namespace(
            data_file='appendix.xlsx',
            mode='train',
            epochs=200,
            batch_size=32,
            hidden_layers=[64, 32, 16],
            dropout_rate=0.2,
            output_dir='.'
        )
    else:
        print("未发现预处理数据，开始完整流程...")
        args = argparse.Namespace(
            data_file='appendix.xlsx',
            mode='full',
            epochs=200,
            batch_size=32,
            hidden_layers=[64, 32, 16],
            dropout_rate=0.2,
            output_dir='.'
        )
    
    # 手动设置参数
    sys.argv = ['main.py']
    for key, value in vars(args).items():
        if key == 'hidden_layers':
            sys.argv.extend([f'--{key}'] + [str(v) for v in value])
        else:
            sys.argv.extend([f'--{key}', str(value)])
    
    main()

if __name__ == "__main__":
    if len(sys.argv) == 1:
        # 如果没有命令行参数，使用快速开始模式
        quick_start()
    else:
        # 使用命令行参数
        main()
