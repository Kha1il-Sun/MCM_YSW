#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
问题3运行脚本
提供便捷的运行接口和参数检查
"""

import os
import sys
import argparse
import logging
from pathlib import Path


def setup_logging(level=logging.INFO):
    """设置日志"""
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('q3_run.log'),
            logging.StreamHandler()
        ]
    )


def check_environment():
    """检查运行环境"""
    logger = logging.getLogger(__name__)
    
    # 检查Python版本
    if sys.version_info < (3, 8):
        logger.error("Python版本需要3.8或更高")
        return False
    
    # 检查依赖
    required_packages = ['pandas', 'numpy', 'scikit-learn', 'scipy', 'xgboost']
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        logger.error(f"缺少依赖包: {', '.join(missing_packages)}")
        logger.info("请运行: pip install -r requirements.txt")
        return False
    
    logger.info("环境检查通过")
    return True


def check_data_availability():
    """检查数据可用性"""
    logger = logging.getLogger(__name__)
    
    data_dir = Path("../step2_1")
    required_files = [
        "step1_long_records.csv",
        "step1_surv_dat_fit.csv", 
        "step1_report.csv",
        "step1_config.yaml"
    ]
    
    missing_files = []
    for file in required_files:
        if not (data_dir / file).exists():
            missing_files.append(str(data_dir / file))
    
    if missing_files:
        logger.error(f"缺少数据文件: {missing_files}")
        logger.info("请先运行Q2/step2_1/data_processing.py生成数据")
        return False
    
    logger.info("数据文件检查通过")
    return True


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='问题3：综合多因素的NIPT最佳时点优化',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  python run_analysis.py                    # 基础运行
  python run_analysis.py --verbose         # 详细输出  
  python run_analysis.py --debug          # 调试模式
  python run_analysis.py --check-only     # 仅检查环境
  python run_analysis.py --config custom.yaml  # 自定义配置
        """
    )
    
    parser.add_argument('--config', type=str, default='config/step3_config.yaml',
                       help='配置文件路径 (默认: config/step3_config.yaml)')
    parser.add_argument('--data-dir', type=str, default='../step2_1',
                       help='数据目录路径 (默认: ../step2_1)')
    parser.add_argument('--output-dir', type=str, default='outputs',
                       help='输出目录路径 (默认: outputs)')
    parser.add_argument('--verbose', action='store_true',
                       help='详细输出')
    parser.add_argument('--debug', action='store_true',
                       help='调试模式')
    parser.add_argument('--check-only', action='store_true',
                       help='仅检查环境和数据，不运行分析')
    parser.add_argument('--validate-only', action='store_true',
                       help='仅验证配置，不运行分析')
    
    args = parser.parse_args()
    
    # 设置日志级别
    if args.debug:
        setup_logging(logging.DEBUG)
    elif args.verbose:
        setup_logging(logging.INFO)
    else:
        setup_logging(logging.WARNING)
    
    logger = logging.getLogger(__name__)
    
    print("=" * 60)
    print("问题3：综合多因素的NIPT最佳时点优化")
    print("=" * 60)
    
    # 环境检查
    print("1. 检查运行环境...")
    if not check_environment():
        sys.exit(1)
    
    print("2. 检查数据可用性...")
    if not check_data_availability():
        sys.exit(1)
    
    # 配置文件检查
    config_path = Path(args.config)
    if not config_path.exists():
        logger.error(f"配置文件不存在: {config_path}")
        print("请检查配置文件路径或使用默认配置文件")
        sys.exit(1)
    
    print(f"3. 配置文件: {config_path}")
    
    if args.check_only:
        print("✓ 环境和数据检查完成，所有检查通过")
        return
    
    if args.validate_only:
        print("4. 验证配置文件...")
        try:
            import yaml
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            print("✓ 配置文件格式正确")
            
            # 简单配置验证
            required_sections = ['model_params', 'optimization', 'grouping']
            for section in required_sections:
                if section not in config:
                    logger.warning(f"配置文件缺少节: {section}")
            
        except Exception as e:
            logger.error(f"配置文件验证失败: {e}")
            sys.exit(1)
        
        print("✓ 配置验证完成")
        return
    
    # 运行主分析
    print("4. 开始运行分析...")
    try:
        # 导入主程序
        from p3 import main as p3_main
        
        # 准备参数
        sys.argv = [
            'p3.py',
            '--config', str(config_path),
            '--data-dir', args.data_dir,
            '--output-dir', args.output_dir
        ]
        
        if args.verbose:
            sys.argv.append('--verbose')
        if args.debug:
            sys.argv.append('--debug')
        
        # 运行分析
        p3_main()
        
        print("=" * 60)
        print("✓ 分析完成！")
        print(f"✓ 结果保存在: {Path(args.output_dir).absolute()}")
        print("=" * 60)
        
        # 显示主要输出文件
        output_dir = Path(args.output_dir)
        if output_dir.exists():
            print("\n主要输出文件:")
            key_files = [
                'q3_bmi_groups_optimal.csv',
                'q3_comprehensive_report.txt',
                'q3_executive_summary.txt'
            ]
            
            for file in key_files:
                file_path = output_dir / file
                if file_path.exists():
                    size = file_path.stat().st_size / 1024  # KB
                    print(f"  ✓ {file} ({size:.1f} KB)")
                else:
                    print(f"  ✗ {file} (未生成)")
        
    except KeyboardInterrupt:
        print("\n用户中断，分析停止")
        sys.exit(1)
    except Exception as e:
        logger.error(f"分析过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()