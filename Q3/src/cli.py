"""
命令行接口模块

提供Q3项目的命令行交互接口。
"""

import argparse
import sys
import yaml
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List

# 导入项目模块
from . import dataio
from . import utils

logger = logging.getLogger(__name__)

def create_parser() -> argparse.ArgumentParser:
    """创建命令行参数解析器"""
    parser = argparse.ArgumentParser(
        prog="q3-analysis",
        description="Q3 NIPT BMI分组与最佳时点优化分析工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  %(prog)s --config config.yaml --data data/q3_preprocessed.csv
  %(prog)s --config config.yaml --data data.csv --outdir results --debug
  %(prog)s --data data.csv --steps preprocess model optimize --verbose
        """
    )
    
    # 必需参数
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="输入数据文件路径（CSV格式）"
    )
    
    # 可选参数
    parser.add_argument(
        "--config", 
        type=str,
        default="config.yaml",
        help="配置文件路径 (默认: config.yaml)"
    )
    
    parser.add_argument(
        "--outdir",
        type=str,
        help="输出目录路径（覆盖配置文件设置）"
    )
    
    # 运行模式
    parser.add_argument(
        "--debug",
        action="store_true",
        help="启用调试模式（详细日志输出）"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true", 
        help="启用详细输出"
    )
    
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="静默模式（最少输出）"
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="干运行模式（验证配置但不执行分析）"
    )
    
    # 步骤控制
    parser.add_argument(
        "--steps",
        nargs="+",
        choices=["preprocess", "model", "optimize", "group", "validate", "sensitivity", "report"],
        help="指定要执行的分析步骤"
    )
    
    parser.add_argument(
        "--skip-steps",
        nargs="+", 
        choices=["preprocess", "model", "optimize", "group", "validate", "sensitivity", "report"],
        help="指定要跳过的分析步骤"
    )
    
    # 输出控制
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="禁用图表生成"
    )
    
    parser.add_argument(
        "--no-reports", 
        action="store_true",
        help="禁用报告生成"
    )
    
    parser.add_argument(
        "--format",
        choices=["txt", "html", "pdf"],
        default="txt",
        help="报告输出格式 (默认: txt)"
    )
    
    # 算法参数覆盖
    parser.add_argument(
        "--threshold",
        type=float,
        help="Y染色体浓度阈值（覆盖配置文件）"
    )
    
    parser.add_argument(
        "--bmi-method",
        choices=["hybrid", "tree", "dp", "custom"],
        help="BMI分组方法（覆盖配置文件）"
    )
    
    parser.add_argument(
        "--cv-folds",
        type=int,
        help="交叉验证折数（覆盖配置文件）"
    )
    
    # 计算资源
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=-1,
        help="并行作业数 (默认: -1, 使用所有CPU)"
    )
    
    parser.add_argument(
        "--memory-limit",
        type=str,
        help="内存限制 (如: '8GB', '4000MB')"
    )
    
    # 版本和帮助
    parser.add_argument(
        "--version",
        action="version",
        version="%(prog)s 1.0.0"
    )
    
    return parser

def validate_arguments(args: argparse.Namespace) -> bool:
    """验证命令行参数"""
    
    # 验证数据文件存在
    if not Path(args.data).exists():
        logger.error(f"数据文件不存在: {args.data}")
        return False
    
    # 验证配置文件存在  
    if not Path(args.config).exists():
        logger.error(f"配置文件不存在: {args.config}")
        return False
    
    # 验证参数冲突
    if args.quiet and args.verbose:
        logger.error("不能同时使用 --quiet 和 --verbose")
        return False
    
    if args.steps and args.skip_steps:
        overlap = set(args.steps) & set(args.skip_steps)
        if overlap:
            logger.error(f"步骤冲突: {overlap} 同时在包含和排除列表中")
            return False
    
    # 验证数值参数范围
    if args.threshold and not (3.0 <= args.threshold <= 5.0):
        logger.error("阈值应在3.0-5.0%之间")
        return False
        
    if args.cv_folds and not (2 <= args.cv_folds <= 10):
        logger.error("交叉验证折数应在2-10之间")
        return False
    
    return True

def override_config_from_args(config: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    """根据命令行参数覆盖配置"""
    
    # 输出目录
    if args.outdir:
        config['output']['base_dir'] = args.outdir
    
    # 算法参数
    if args.threshold:
        config['data']['threshold'] = args.threshold
    
    if args.bmi_method:
        config['grouping']['method'] = args.bmi_method
        
    if args.cv_folds:
        config['modeling']['cross_validation']['n_folds'] = args.cv_folds
    
    # 计算资源
    if args.n_jobs:
        config['computing']['parallel']['n_jobs'] = args.n_jobs
    
    # 输出控制
    if args.no_plots:
        config['output']['figures']['enabled'] = False
    
    if args.no_reports:
        config['output']['reports']['enabled'] = False
        
    if args.format:
        config['output']['reports']['format'] = [args.format]
    
    # 调试模式
    if args.debug:
        config['debug']['enabled'] = True
        config['logging']['level'] = 'DEBUG'
    
    if args.verbose:
        config['logging']['level'] = 'INFO'
    elif args.quiet:
        config['logging']['level'] = 'WARNING'
    
    return config

def setup_logging_from_config(config: Dict[str, Any]) -> logging.Logger:
    """根据配置设置日志系统"""
    log_config = config.get("logging", {})
    
    # 创建日志目录
    log_file = log_config.get("file", "outputs/logs/analysis.log")
    log_dir = Path(log_file).parent
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # 配置日志
    level = getattr(logging, log_config.get("level", "INFO"))
    format_str = log_config.get("format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    
    # 清除现有处理器
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # 配置根日志器
    logging.basicConfig(
        level=level,
        format=format_str,
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler(sys.stdout) if log_config.get("console", True) else logging.NullHandler()
        ]
    )
    
    return logging.getLogger("Q3_CLI")

def print_banner(logger: logging.Logger) -> None:
    """打印项目横幅"""
    banner = """
    ╔═══════════════════════════════════════════════════════════╗
    ║                    Q3 NIPT 优化分析                       ║  
    ║            基于多因素的BMI分组与最佳时点优化                ║
    ║                      Version 1.0.0                       ║
    ╚═══════════════════════════════════════════════════════════╝
    """
    logger.info(banner)

def print_config_summary(config: Dict[str, Any], logger: logging.Logger) -> None:
    """打印配置摘要"""
    logger.info("配置摘要:")
    logger.info(f"  数据阈值: {config['data'].get('threshold', 4.0)}%")
    logger.info(f"  孕周范围: {config['data'].get('gestational_range', [10, 25])}周")
    logger.info(f"  分组方法: {config['grouping'].get('method', 'hybrid')}")
    logger.info(f"  建模方法: {config['modeling']['attain_model'].get('type', 'xgboost')}")
    logger.info(f"  交叉验证: {config['modeling']['cross_validation'].get('n_folds', 5)}折")
    logger.info(f"  输出目录: {config['output']['base_dir']}")

def determine_steps_to_run(args: argparse.Namespace) -> List[str]:
    """确定要运行的步骤"""
    all_steps = ["preprocess", "model", "optimize", "group", "validate", "sensitivity", "report"]
    
    if args.steps:
        # 用户指定了要运行的步骤
        steps = args.steps
    else:
        # 默认运行所有步骤
        steps = all_steps
    
    # 移除要跳过的步骤
    if args.skip_steps:
        steps = [step for step in steps if step not in args.skip_steps]
    
    return steps

def run_analysis_pipeline(config: Dict[str, Any], data_path: str, steps: List[str], 
                         dry_run: bool = False) -> bool:
    """
    运行分析流水线
    
    Parameters
    ----------
    config : Dict[str, Any]
        配置字典
    data_path : str  
        数据文件路径
    steps : List[str]
        要执行的步骤列表
    dry_run : bool
        是否为干运行模式
        
    Returns
    -------
    bool
        是否成功完成
    """
    logger = logging.getLogger("Q3_Pipeline")
    
    if dry_run:
        logger.info("=== 干运行模式 - 仅验证配置 ===")
        logger.info(f"将要执行的步骤: {', '.join(steps)}")
        logger.info("配置验证完成，未执行实际分析")
        return True
    
    logger.info("=== 开始分析流水线 ===")
    logger.info(f"执行步骤: {', '.join(steps)}")
    
    try:
        # 步骤1: 数据预处理
        if "preprocess" in steps:
            logger.info("Step 1/7: 数据预处理")
            # TODO: 实现数据预处理逻辑
            logger.info("数据预处理完成")
        
        # 步骤2: 模型训练
        if "model" in steps:
            logger.info("Step 2/7: 模型训练")
            # TODO: 实现模型训练逻辑
            logger.info("模型训练完成")
        
        # 步骤3: 时点优化
        if "optimize" in steps:
            logger.info("Step 3/7: 时点优化")
            # TODO: 实现时点优化逻辑
            logger.info("时点优化完成")
        
        # 步骤4: BMI分组
        if "group" in steps:
            logger.info("Step 4/7: BMI分组")
            # TODO: 实现BMI分组逻辑
            logger.info("BMI分组完成")
        
        # 步骤5: 模型验证
        if "validate" in steps:
            logger.info("Step 5/7: 模型验证")
            # TODO: 实现模型验证逻辑
            logger.info("模型验证完成")
        
        # 步骤6: 敏感性分析
        if "sensitivity" in steps:
            logger.info("Step 6/7: 敏感性分析")
            # TODO: 实现敏感性分析逻辑
            logger.info("敏感性分析完成")
        
        # 步骤7: 结果报告
        if "report" in steps:
            logger.info("Step 7/7: 结果报告")
            # TODO: 实现结果报告逻辑
            logger.info("结果报告完成")
        
        logger.info("=== 分析流水线完成 ===")
        return True
        
    except Exception as e:
        logger.exception(f"分析流水线执行失败: {e}")
        return False

def main(args: Optional[List[str]] = None) -> int:
    """
    主入口函数
    
    Parameters
    ----------  
    args : Optional[List[str]]
        命令行参数列表，如果为None则从sys.argv获取
        
    Returns
    -------
    int
        退出状态码（0表示成功，非0表示失败）
    """
    
    # 创建参数解析器
    parser = create_parser()
    
    # 解析参数
    parsed_args = parser.parse_args(args)
    
    try:
        # 验证参数
        if not validate_arguments(parsed_args):
            return 1
        
        # 加载配置
        with open(parsed_args.config, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # 根据命令行参数覆盖配置
        config = override_config_from_args(config, parsed_args)
        
        # 设置日志
        logger = setup_logging_from_config(config)
        
        # 打印横幅和配置摘要
        print_banner(logger)
        print_config_summary(config, logger)
        
        # 确定要运行的步骤
        steps = determine_steps_to_run(parsed_args)
        
        # 运行分析
        success = run_analysis_pipeline(
            config=config,
            data_path=parsed_args.data,
            steps=steps,
            dry_run=parsed_args.dry_run
        )
        
        if success:
            logger.info("分析成功完成！")
            return 0
        else:
            logger.error("分析执行失败！")
            return 1
            
    except KeyboardInterrupt:
        print("\n分析被用户中断")
        return 2
    except Exception as e:
        print(f"发生未预期错误: {e}")
        if 'logger' in locals():
            logger.exception("CLI执行过程中发生异常")
        return 3

if __name__ == "__main__":
    sys.exit(main())