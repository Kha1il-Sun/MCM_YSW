"""
I/O 工具模块
读取和校验 Step1 产物与可选QC
"""

import pandas as pd
import numpy as np
import yaml
import os
from typing import Tuple, Dict, Any, Optional
import logging

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_step1_products(datadir: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    读取 Step1 的所有产物
    
    Parameters:
    -----------
    datadir : str
        数据目录路径
        
    Returns:
    --------
    long_df : pd.DataFrame
        逐次检测长表
    surv_df : pd.DataFrame
        对齐版区间删失表（含 L_fit/R_fit）
    report : pd.DataFrame
        质量统计与阈值邻域 σ 摘要
    cfg1 : dict
        Step1 配置参数
    """
    logger.info(f"从 {datadir} 读取 Step1 产物...")
    
    # 读取长表
    long_path = os.path.join(datadir, "step1_long_records.csv")
    if not os.path.exists(long_path):
        raise FileNotFoundError(f"未找到长表文件: {long_path}")
    long_df = pd.read_csv(long_path)
    logger.info(f"读取长表: {len(long_df)} 条记录, {long_df['id'].nunique()} 个个体")
    
    # 读取生存数据
    surv_path = os.path.join(datadir, "step1_surv_dat_fit.csv")
    if not os.path.exists(surv_path):
        raise FileNotFoundError(f"未找到生存数据文件: {surv_path}")
    surv_df = pd.read_csv(surv_path)
    logger.info(f"读取生存数据: {len(surv_df)} 个个体")
    
    # 读取报告
    report_path = os.path.join(datadir, "step1_report.csv")
    if not os.path.exists(report_path):
        raise FileNotFoundError(f"未找到报告文件: {report_path}")
    report = pd.read_csv(report_path)
    logger.info("读取处理报告")
    
    # 读取配置
    config_path = os.path.join(datadir, "step1_config.yaml")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"未找到配置文件: {config_path}")
    with open(config_path, 'r', encoding='utf-8') as f:
        cfg1 = yaml.safe_load(f)
    logger.info("读取 Step1 配置")
    
    return long_df, surv_df, report, cfg1


def validate_data_consistency(long_df: pd.DataFrame, surv_df: pd.DataFrame, 
                            report: pd.DataFrame, cfg1: Dict[str, Any]) -> bool:
    """
    校验数据一致性
    
    Parameters:
    -----------
    long_df : pd.DataFrame
        长表数据
    surv_df : pd.DataFrame
        生存数据
    report : pd.DataFrame
        报告数据
    cfg1 : dict
        Step1 配置
        
    Returns:
    --------
    bool
        数据是否一致
    """
    logger.info("开始数据一致性校验...")
    
    # 检查个体ID一致性
    long_ids = set(long_df['id'].unique())
    surv_ids = set(surv_df['id'].unique())
    
    if long_ids != surv_ids:
        logger.warning(f"个体ID不一致: 长表{len(long_ids)}个, 生存表{len(surv_ids)}个")
        missing_in_surv = long_ids - surv_ids
        missing_in_long = surv_ids - long_ids
        if missing_in_surv:
            logger.warning(f"生存表中缺失的ID: {list(missing_in_surv)[:5]}...")
        if missing_in_long:
            logger.warning(f"长表中缺失的ID: {list(missing_in_long)[:5]}...")
        return False
    
    # 检查BMI字段一致性
    if 'BMI_base' in surv_df.columns and 'BMI_used' in long_df.columns:
        # 检查首检BMI是否一致
        first_bmi_long = long_df.groupby('id')['BMI_used'].first()
        bmi_surv = surv_df.set_index('id')['BMI_base']
        
        # 允许小的数值误差
        bmi_diff = np.abs(first_bmi_long - bmi_surv)
        if bmi_diff.max() > 0.01:  # 0.01 BMI单位的容差
            logger.warning(f"BMI不一致: 最大差异 {bmi_diff.max():.4f}")
            return False
    
    # 检查孕周范围合理性
    week_range = long_df['week'].agg(['min', 'max'])
    if week_range['min'] < 8 or week_range['max'] > 35:
        logger.warning(f"孕周范围异常: {week_range['min']:.1f} - {week_range['max']:.1f}")
        return False
    
    # 检查Y浓度范围
    y_range = long_df['Y_frac'].agg(['min', 'max'])
    if y_range['min'] < 0 or y_range['max'] > 1:
        logger.warning(f"Y浓度范围异常: {y_range['min']:.4f} - {y_range['max']:.4f}")
        return False
    
    # 检查删失类型分布
    censor_counts = surv_df['censor_type'].value_counts()
    expected_types = {'left', 'right', 'interval'}
    actual_types = set(censor_counts.index)
    if not expected_types.issubset(actual_types):
        logger.warning(f"删失类型不完整: 期望{expected_types}, 实际{actual_types}")
        return False
    
    logger.info("数据一致性校验通过")
    return True


def extract_sigma_info(report: pd.DataFrame) -> Dict[str, Any]:
    """
    从报告中提取σ估计信息
    
    Parameters:
    -----------
    report : pd.DataFrame
        处理报告
        
    Returns:
    --------
    dict
        σ估计信息
    """
    logger.info("提取σ估计信息...")
    
    # 从报告的最后一行提取检测误差建模信息
    error_info_str = report.iloc[-1]['检测误差建模']
    
    # 解析字符串格式的字典
    import ast
    try:
        error_info = ast.literal_eval(error_info_str)
    except:
        logger.warning("无法解析检测误差建模信息，使用默认值")
        error_info = {
            '阈值附近样本数': 0,
            'Y浓度残差标准差': 0.0059,
            'Y浓度残差方差': 3.43e-5
        }
    
    sigma_info = {
        'global_sigma': error_info.get('Y浓度残差标准差', 0.0059),
        'global_variance': error_info.get('Y浓度残差方差', 3.43e-5),
        'threshold_nearby_n': error_info.get('阈值附近样本数', 0)
    }
    
    logger.info(f"全局σ: {sigma_info['global_sigma']:.6f}")
    logger.info(f"阈值附近样本数: {sigma_info['threshold_nearby_n']}")
    
    return sigma_info


def load_step2_config(config_path: str) -> Dict[str, Any]:
    """
    读取 Step2 配置
    
    Parameters:
    -----------
    config_path : str
        配置文件路径
        
    Returns:
    --------
    dict
        配置参数
    """
    logger.info(f"读取 Step2 配置: {config_path}")
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"未找到配置文件: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        cfg2 = yaml.safe_load(f)
    
    logger.info("Step2 配置读取完成")
    return cfg2


def save_results(results: Dict[str, Any], output_dir: str, config: Dict[str, Any]) -> None:
    """
    保存结果到输出目录
    
    Parameters:
    -----------
    results : dict
        结果字典
    output_dir : str
        输出目录
    config : dict
        配置参数
    """
    logger.info(f"保存结果到 {output_dir}")
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存CSV文件
    if 'wstar_curve' in results:
        wstar_path = os.path.join(output_dir, "p2_wstar_curve.csv")
        results['wstar_curve'].to_csv(wstar_path, index=False)
        logger.info(f"保存w*曲线: {wstar_path}")
    
    if 'groups' in results:
        group_path = os.path.join(output_dir, "p2_group_recommendation.csv")
        results['groups'].to_csv(group_path, index=False)
        logger.info(f"保存分组推荐: {group_path}")
    
    # 保存报告
    if 'report' in results:
        report_path = os.path.join(output_dir, "p2_report.txt")
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(results['report'])
        logger.info(f"保存报告: {report_path}")
    
    # 保存配置回填
    config_path = os.path.join(output_dir, "step2_config_used.yaml")
    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
    logger.info(f"保存使用配置: {config_path}")


def get_data_summary(long_df: pd.DataFrame, surv_df: pd.DataFrame) -> Dict[str, Any]:
    """
    获取数据摘要信息
    
    Parameters:
    -----------
    long_df : pd.DataFrame
        长表数据
    surv_df : pd.DataFrame
        生存数据
        
    Returns:
    --------
    dict
        数据摘要
    """
    summary = {
        'n_records': len(long_df),
        'n_individuals': long_df['id'].nunique(),
        'week_range': [long_df['week'].min(), long_df['week'].max()],
        'bmi_range': [long_df['BMI_used'].min(), long_df['BMI_used'].max()],
        'y_range': [long_df['Y_frac'].min(), long_df['Y_frac'].max()],
        'censor_distribution': surv_df['censor_type'].value_counts().to_dict(),
        'avg_records_per_person': len(long_df) / long_df['id'].nunique()
    }
    
    return summary


def check_data_quality(long_df: pd.DataFrame, surv_df: pd.DataFrame) -> Dict[str, Any]:
    """
    检查数据质量
    
    Parameters:
    -----------
    long_df : pd.DataFrame
        长表数据
    surv_df : pd.DataFrame
        生存数据
        
    Returns:
    --------
    dict
        质量检查结果
    """
    quality_checks = {}
    
    # 检查缺失值
    quality_checks['missing_values_long'] = long_df.isnull().sum().to_dict()
    quality_checks['missing_values_surv'] = surv_df.isnull().sum().to_dict()
    
    # 检查重复记录
    quality_checks['duplicate_records'] = long_df.duplicated().sum()
    
    # 检查孕周递增性
    week_increasing = long_df.groupby('id')['week'].apply(lambda x: x.is_monotonic_increasing)
    quality_checks['non_increasing_weeks'] = (~week_increasing).sum()
    
    # 检查BMI合理性
    bmi_outliers = (long_df['BMI_used'] < 15) | (long_df['BMI_used'] > 50)
    quality_checks['bmi_outliers'] = bmi_outliers.sum()
    
    # 检查Y浓度合理性
    y_outliers = (long_df['Y_frac'] < 0) | (long_df['Y_frac'] > 1)
    quality_checks['y_outliers'] = y_outliers.sum()
    
    return quality_checks
