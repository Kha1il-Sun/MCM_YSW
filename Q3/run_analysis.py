#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Q3 NIPT BMI分组与最佳时点优化 - 主运行脚本
"""

import os
import sys
import yaml
import argparse
import logging
from pathlib import Path
from typing import Dict, Any
import pandas as pd
import numpy as np
from datetime import datetime

# 添加src目录到路径
sys.path.append(str(Path(__file__).parent / "src"))

def setup_logging(config: Dict[str, Any]) -> logging.Logger:
    """设置日志系统"""
    log_config = config.get("logging", {})
    
    # 创建日志目录
    log_file = log_config.get("file", "outputs/logs/analysis.log")
    log_dir = Path(log_file).parent
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # 配置日志
    logging.basicConfig(
        level=getattr(logging, log_config.get("level", "INFO")),
        format=log_config.get("format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s"),
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler(sys.stdout) if log_config.get("console", True) else logging.NullHandler()
        ]
    )
    
    logger = logging.getLogger("Q3_Analysis")
    logger.info("="*60)
    logger.info("Q3 NIPT BMI分组与最佳时点优化分析开始")
    logger.info("="*60)
    
    return logger

def load_config(config_path: str) -> Dict[str, Any]:
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

def validate_config(config: Dict[str, Any], logger: logging.Logger) -> bool:
    """验证配置文件完整性"""
    logger.info("验证配置文件...")
    
    required_sections = ['data', 'modeling', 'grouping', 'optimization', 'output']
    
    for section in required_sections:
        if section not in config:
            logger.error(f"缺少配置段: {section}")
            return False
    
    # 验证数据配置
    data_config = config['data']
    if 'threshold' not in data_config or not (3.0 <= data_config['threshold'] <= 5.0):
        logger.error("阈值配置无效，应在3.0-5.0%之间")
        return False
    
    # 验证孕周范围
    gest_range = data_config.get('gestational_range', [10, 25])
    if len(gest_range) != 2 or gest_range[0] >= gest_range[1]:
        logger.error("孕周范围配置无效")
        return False
    
    logger.info("配置文件验证通过")
    return True

def create_output_directories(config: Dict[str, Any], logger: logging.Logger) -> None:
    """创建输出目录结构"""
    logger.info("创建输出目录结构...")
    
    base_dir = Path(config['output']['base_dir'])
    
    # 创建主要目录
    directories = [
        base_dir,
        base_dir / "figures",
        base_dir / "logs",
        base_dir / "intermediate",
        base_dir / "reports"
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
        logger.debug(f"创建目录: {directory}")

def log_system_info(logger: logging.Logger) -> None:
    """记录系统信息"""
    logger.info("系统信息:")
    logger.info(f"  Python版本: {sys.version}")
    logger.info(f"  工作目录: {os.getcwd()}")
    logger.info(f"  运行时间: {datetime.now()}")
    
    # 记录关键包版本
    try:
        import pandas as pd
        import numpy as np
        import sklearn
        logger.info(f"  Pandas版本: {pd.__version__}")
        logger.info(f"  NumPy版本: {np.__version__}")
        logger.info(f"  Scikit-learn版本: {sklearn.__version__}")
    except ImportError as e:
        logger.warning(f"无法导入包: {e}")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="Q3 NIPT BMI分组与最佳时点优化分析",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  python run_analysis.py --config config.yaml --data data/q3_preprocessed.csv
  python run_analysis.py --config config.yaml --data data/q3_preprocessed.csv --outdir results
  python run_analysis.py --config config.yaml --data data/q3_preprocessed.csv --debug
        """
    )
    
    parser.add_argument(
        "--config", 
        type=str, 
        default="config.yaml",
        help="配置文件路径 (默认: config.yaml)"
    )
    
    parser.add_argument(
        "--data", 
        type=str, 
        required=True,
        help="输入数据文件路径"
    )
    
    parser.add_argument(
        "--outdir", 
        type=str,
        help="输出目录（覆盖配置文件中的设置）"
    )
    
    parser.add_argument(
        "--debug", 
        action="store_true",
        help="启用调试模式"
    )
    
    parser.add_argument(
        "--dry-run", 
        action="store_true",
        help="干运行模式（仅验证配置，不执行分析）"
    )
    
    parser.add_argument(
        "--steps", 
        nargs="+",
        choices=["preprocess", "model", "optimize", "group", "validate", "sensitivity", "report"],
        help="指定要运行的步骤"
    )
    
    args = parser.parse_args()
    
    try:
        # 加载配置
        if not Path(args.config).exists():
            print(f"错误: 配置文件 {args.config} 不存在")
            sys.exit(1)
        
        config = load_config(args.config)
        
        # 覆盖输出目录设置
        if args.outdir:
            config['output']['base_dir'] = args.outdir
        
        # 启用调试模式
        if args.debug:
            config['debug']['enabled'] = True
            config['logging']['level'] = 'DEBUG'
        
        # 设置日志
        logger = setup_logging(config)
        
        # 记录系统信息
        log_system_info(logger)
        
        # 验证配置
        if not validate_config(config, logger):
            logger.error("配置验证失败")
            sys.exit(1)
        
        # 创建输出目录
        create_output_directories(config, logger)
        
        # 验证数据文件
        if not Path(args.data).exists():
            logger.error(f"数据文件不存在: {args.data}")
            sys.exit(1)
        
        # 记录配置和参数
        logger.info("运行参数:")
        logger.info(f"  配置文件: {args.config}")
        logger.info(f"  数据文件: {args.data}")
        logger.info(f"  输出目录: {config['output']['base_dir']}")
        logger.info(f"  调试模式: {args.debug}")
        logger.info(f"  干运行模式: {args.dry_run}")
        
        if args.steps:
            logger.info(f"  指定步骤: {', '.join(args.steps)}")
        
        # 保存使用的配置文件副本
        config_backup_path = Path(config['output']['base_dir']) / "config_used.yaml"
        with open(config_backup_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
        logger.info(f"配置文件副本已保存: {config_backup_path}")
        
        if args.dry_run:
            logger.info("干运行模式 - 配置验证完成，未执行实际分析")
            return
        
        # 导入并运行分析模块
        logger.info("开始执行分析...")
        
        # 这里应该导入实际的分析模块
        # 由于目前没有实现具体的分析代码，我们只是做一个框架展示
        logger.info("注意: 实际分析模块尚未实现")
        logger.info("请实现以下模块:")
        logger.info("  - src/dataio.py: 数据读取与预处理")
        logger.info("  - src/models/: 建模模块")
        logger.info("  - src/optimize.py: 优化算法")
        logger.info("  - src/grouping.py: 分组算法")
        logger.info("  - src/sensitivity.py: 敏感性分析")
        logger.info("  - src/plots.py: 可视化")
        
        # 示例：读取数据基本信息
        logger.info("读取数据基本信息...")
        try:
            df = pd.read_csv(args.data)
            logger.info(f"数据形状: {df.shape}")
            logger.info(f"列名: {list(df.columns)}")
            
            if 'BMI' in df.columns:
                logger.info(f"BMI范围: [{df['BMI'].min():.1f}, {df['BMI'].max():.1f}]")
            if 'gest_week' in df.columns:
                logger.info(f"孕周范围: [{df['gest_week'].min():.1f}, {df['gest_week'].max():.1f}]")
            if 'Y_pct' in df.columns:
                logger.info(f"Y浓度范围: [{df['Y_pct'].min():.3f}, {df['Y_pct'].max():.3f}]")
                
        except Exception as e:
            logger.error(f"读取数据失败: {e}")
            sys.exit(1)
        
        logger.info("="*60)
        logger.info("分析框架执行完成")
        logger.info("请根据README_problem3.md实现具体的分析模块")
        logger.info("="*60)
        
    except KeyboardInterrupt:
        print("\n分析被用户中断")
        sys.exit(1)
    except Exception as e:
        print(f"发生错误: {e}")
        if 'logger' in locals():
            logger.exception("运行时发生异常")
        sys.exit(1)

if __name__ == "__main__":
    main()