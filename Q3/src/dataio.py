"""
数据读取与预处理模块

负责数据加载、清洗、特征工程等预处理任务。
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

def load_data(file_path: str, **kwargs) -> pd.DataFrame:
    """
    加载数据文件
    
    Parameters
    ----------
    file_path : str
        数据文件路径
    **kwargs
        传递给pandas.read_csv的额外参数
        
    Returns
    -------
    pd.DataFrame
        加载的数据框
        
    Raises
    ------
    FileNotFoundError
        当文件不存在时
    ValueError
        当数据格式不符合要求时
    """
    logger.info(f"加载数据文件: {file_path}")
    
    if not Path(file_path).exists():
        raise FileNotFoundError(f"数据文件不存在: {file_path}")
    
    # 读取数据
    df = pd.read_csv(file_path, **kwargs)
    logger.info(f"数据加载完成，形状: {df.shape}")
    
    return df

def validate_data_format(df: pd.DataFrame, required_columns: List[str]) -> bool:
    """
    验证数据格式
    
    Parameters
    ----------
    df : pd.DataFrame
        待验证的数据框
    required_columns : List[str]
        必需的列名列表
        
    Returns
    -------
    bool
        验证是否通过
    """
    logger.info("验证数据格式...")
    
    # 检查必需列
    missing_cols = set(required_columns) - set(df.columns)
    if missing_cols:
        logger.error(f"缺少必需列: {missing_cols}")
        return False
    
    # 检查数据类型
    numeric_cols = ['gest_week', 'BMI', 'Y_pct', 'GC', 'readcount', 'age', 'height', 'weight']
    for col in numeric_cols:
        if col in df.columns and not pd.api.types.is_numeric_dtype(df[col]):
            logger.error(f"列 {col} 应为数值型")
            return False
    
    logger.info("数据格式验证通过")
    return True

def preprocess_data(df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    """
    数据预处理主函数
    
    Parameters
    ----------
    df : pd.DataFrame
        原始数据框
    config : Dict[str, Any]
        预处理配置
        
    Returns
    -------
    pd.DataFrame
        预处理后的数据框
    """
    logger.info("开始数据预处理...")
    
    # 复制数据，避免修改原始数据
    df_processed = df.copy()
    
    # 1. 基础清洗
    df_processed = clean_basic_data(df_processed, config)
    
    # 2. 特征工程
    df_processed = engineer_features(df_processed, config)
    
    # 3. 异常值处理
    df_processed = handle_outliers(df_processed, config)
    
    logger.info(f"数据预处理完成，最终形状: {df_processed.shape}")
    return df_processed

def clean_basic_data(df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    """基础数据清洗"""
    logger.info("执行基础数据清洗...")
    
    df_clean = df.copy()
    
    # 仅保留男胎数据（如果有性别信息）
    if 'gender' in df_clean.columns and config['data'].get('male_only', True):
        df_clean = df_clean[df_clean['gender'] == 'M']
        logger.info(f"筛选男胎后数据量: {len(df_clean)}")
    
    # 孕周范围筛选
    gest_range = config['data'].get('gestational_range', [10, 25])
    mask = (df_clean['gest_week'] >= gest_range[0]) & (df_clean['gest_week'] <= gest_range[1])
    df_clean = df_clean[mask]
    logger.info(f"孕周筛选后数据量: {len(df_clean)}")
    
    # 处理缺失值
    before_count = len(df_clean)
    df_clean = df_clean.dropna(subset=['ID', 'gest_week', 'BMI', 'Y_pct'])
    after_count = len(df_clean)
    if before_count > after_count:
        logger.info(f"删除核心字段缺失记录: {before_count - after_count}条")
    
    return df_clean

def engineer_features(df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    """特征工程"""
    logger.info("执行特征工程...")
    
    df_feat = df.copy()
    
    # 获取特征工程配置
    feat_config = config.get('features', {})
    
    # 标准化连续变量
    if feat_config.get('standardize', True):
        continuous_vars = feat_config.get('continuous_vars', ['age', 'GC', 'readcount'])
        for var in continuous_vars:
            if var in df_feat.columns:
                df_feat[f'{var}_std'] = (df_feat[var] - df_feat[var].mean()) / df_feat[var].std()
    
    # 创建交互项
    interactions = feat_config.get('interaction_terms', [])
    for interaction in interactions:
        if len(interaction) == 2 and all(col in df_feat.columns for col in interaction):
            new_col = f"{interaction[0]}_{interaction[1]}_interaction"
            df_feat[new_col] = df_feat[interaction[0]] * df_feat[interaction[1]]
    
    # 时间变换
    time_transforms = feat_config.get('time_transforms', [])
    for transform in time_transforms:
        if transform == 'log':
            df_feat['gest_week_log'] = np.log(df_feat['gest_week'])
        elif transform == 'sqrt':
            df_feat['gest_week_sqrt'] = np.sqrt(df_feat['gest_week'])
    
    logger.info(f"特征工程完成，新特征数: {df_feat.shape[1] - df.shape[1]}")
    return df_feat

def handle_outliers(df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    """异常值处理"""
    logger.info("处理异常值...")
    
    df_clean = df.copy()
    
    # 获取异常值处理配置
    outlier_config = config['data']
    method = outlier_config.get('outlier_method', 'iqr')
    
    if method == 'iqr':
        factor = outlier_config.get('outlier_factor', 1.5)
        df_clean = remove_outliers_iqr(df_clean, factor)
    elif method == 'quantile':
        upper_q = outlier_config.get('y_pct_upper_quantile', 99.5)
        df_clean = remove_outliers_quantile(df_clean, upper_q)
    
    return df_clean

def remove_outliers_iqr(df: pd.DataFrame, factor: float = 1.5) -> pd.DataFrame:
    """使用IQR方法移除异常值"""
    numeric_cols = ['Y_pct', 'BMI', 'age', 'GC', 'readcount']
    df_clean = df.copy()
    
    for col in numeric_cols:
        if col in df_clean.columns:
            Q1 = df_clean[col].quantile(0.25)
            Q3 = df_clean[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - factor * IQR
            upper = Q3 + factor * IQR
            
            before_count = len(df_clean)
            df_clean = df_clean[(df_clean[col] >= lower) & (df_clean[col] <= upper)]
            after_count = len(df_clean)
            
            if before_count > after_count:
                logger.info(f"列 {col} 移除异常值: {before_count - after_count}条")
    
    return df_clean

def remove_outliers_quantile(df: pd.DataFrame, upper_q: float = 99.5) -> pd.DataFrame:
    """使用分位数方法移除异常值"""
    df_clean = df.copy()
    
    # 主要针对Y_pct
    if 'Y_pct' in df_clean.columns:
        upper_bound = np.percentile(df_clean['Y_pct'], upper_q)
        before_count = len(df_clean)
        df_clean = df_clean[df_clean['Y_pct'] <= upper_bound]
        after_count = len(df_clean)
        
        if before_count > after_count:
            logger.info(f"Y_pct 分位数截断移除: {before_count - after_count}条")
    
    return df_clean

def create_target_variables(df: pd.DataFrame, threshold: float = 0.04) -> pd.DataFrame:
    """创建目标变量"""
    logger.info(f"创建目标变量，阈值: {threshold}")
    
    df_target = df.copy()
    
    # 达标指示变量
    df_target['attain_target'] = (df_target['Y_pct'] >= threshold).astype(int)
    
    # 成功指示变量（如果有draw_success列）
    if 'draw_success' in df_target.columns:
        df_target['success_target'] = df_target['draw_success'].astype(int)
    else:
        # 如果没有成功标签，基于其他指标创建代理变量
        logger.warning("未找到draw_success列，将创建代理变量")
        # 这里可以基于readcount、GC等指标创建成功率代理
        df_target['success_target'] = 1  # 临时设为全部成功
    
    return df_target

def split_data_by_id(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    按ID分割数据为训练集和测试集
    
    避免同一ID的数据同时出现在训练集和测试集中
    """
    logger.info(f"按ID分割数据，测试集比例: {test_size}")
    
    # 获取唯一ID
    unique_ids = df['ID'].unique()
    np.random.seed(random_state)
    
    # 随机分割ID
    n_test = int(len(unique_ids) * test_size)
    test_ids = np.random.choice(unique_ids, size=n_test, replace=False)
    train_ids = np.setdiff1d(unique_ids, test_ids)
    
    # 分割数据
    train_df = df[df['ID'].isin(train_ids)]
    test_df = df[df['ID'].isin(test_ids)]
    
    logger.info(f"训练集: {len(train_df)}条记录, {len(train_ids)}个ID")
    logger.info(f"测试集: {len(test_df)}条记录, {len(test_ids)}个ID")
    
    return train_df, test_df

def get_data_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """获取数据摘要统计"""
    summary = {
        'n_records': len(df),
        'n_unique_ids': df['ID'].nunique() if 'ID' in df.columns else None,
        'gestational_week_range': [df['gest_week'].min(), df['gest_week'].max()] if 'gest_week' in df.columns else None,
        'bmi_range': [df['BMI'].min(), df['BMI'].max()] if 'BMI' in df.columns else None,
        'y_pct_range': [df['Y_pct'].min(), df['Y_pct'].max()] if 'Y_pct' in df.columns else None,
        'attain_rate': df['attain_target'].mean() if 'attain_target' in df.columns else None,
        'success_rate': df['success_target'].mean() if 'success_target' in df.columns else None
    }
    
    return summary