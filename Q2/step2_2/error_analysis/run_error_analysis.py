"""
检测误差影响分析运行脚本
"""

import os
import sys
import argparse
from datetime import datetime

# 添加src目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from error_impact_analyzer import ErrorImpactAnalyzer


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='检测误差影响分析')
    parser.add_argument('--data-dir', type=str, default='../../step2_1',
                       help='Step1数据目录路径')
    parser.add_argument('--config', type=str, default='../../config/step2_config.yaml',
                       help='配置文件路径')
    parser.add_argument('--output-dir', type=str, default='outputs',
                       help='输出目录路径')
    
    args = parser.parse_args()
    
    print("🚀 开始检测误差影响分析...")
    print(f"📁 数据目录: {args.data_dir}")
    print(f"⚙️ 配置文件: {args.config}")
    print(f"📤 输出目录: {args.output_dir}")
    print("-" * 50)
    
    try:
        # 创建分析器
        analyzer = ErrorImpactAnalyzer(config_path=args.config)
        
        # 加载数据和模型
        analyzer.load_data_and_model(data_dir=args.data_dir)
        
        # 运行分析
        results = analyzer.run_error_impact_analysis()
        
        # 生成报告
        report = analyzer.generate_summary_report()
        
        print("\n" + "="*60)
        print("📊 分析结果摘要")
        print("="*60)
        
        # 显示对比表格
        if 'comparison_table' in results:
            print("\n📋 推荐时点对比表:")
            print(results['comparison_table'].to_string(index=False))
        
        # 显示关键发现
        print(f"\n🔍 关键发现:")
        print(f"- 基准σ: {results['baseline_sigma']:.6f}")
        print(f"- 分析范围: σ倍数 {results['sigma_multipliers'][0]}x - {results['sigma_multipliers'][-1]}x")
        
        # 计算最大变化
        max_time_change = 0
        for group_name in analyzer.bmi_groups.keys():
            sensitivity = results['sensitivity_analysis'][group_name]
            max_time_change = max(max_time_change, sensitivity['max_time_change'])
        
        print(f"- 最大时点变化: {max_time_change:.2f} 周")
        
        print("\n" + "="*60)
        print("✅ 检测误差影响分析完成！")
        print("="*60)
        
    except Exception as e:
        print(f"❌ 分析过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
