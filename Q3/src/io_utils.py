"""
数据读写和校验模块

提供数据加载、保存和schema验证功能。
"""

import os
import pickle
import yaml
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import logging

from .exceptions import DataValidationError, MCMError
from .utils import ensure_numpy, timer

logger = logging.getLogger(__name__)


class DataSchema:
    """数据schema管理类"""
    
    def __init__(self, schema_path: Union[str, Path]):
        """
        初始化schema
        
        Args:
            schema_path: schema文件路径
        """
        self.schema_path = Path(schema_path)
        with open(self.schema_path, 'r', encoding='utf-8') as f:
            self.schema = yaml.safe_load(f)
        
        self.required_columns = self.schema.get('required_columns', [])
        self.optional_columns = self.schema.get('optional_columns', [])
        self.column_types = self.schema.get('column_types', {})
        self.validation_rules = self.schema.get('validation_rules', [])
        
        logger.info(f"加载schema: {self.schema_path.name}")
    
    def validate_dataframe(self, df: pd.DataFrame, 
                          strict: bool = True) -> List[DataValidationError]:
        """
        验证DataFrame是否符合schema
        
        Args:
            df: 要验证的DataFrame
            strict: 是否严格模式（缺少可选列也报错）
            
        Returns:
            验证错误列表
        """
        errors = []
        
        # 检查必需列
        missing_required = set(self.required_columns) - set(df.columns)
        if missing_required:
            errors.append(DataValidationError(
                f"缺少必需列: {list(missing_required)}",
                failed_rules=["missing_required_columns"],
                column=None
            ))
        
        # 检查可选列（严格模式）
        if strict:
            missing_optional = set(self.optional_columns) - set(df.columns)
            if missing_optional:
                errors.append(DataValidationError(
                    f"缺少可选列: {list(missing_optional)}",
                    failed_rules=["missing_optional_columns"],
                    column=None
                ))
        
        # 检查未知列
        known_columns = set(self.required_columns + self.optional_columns)
        unknown_columns = set(df.columns) - known_columns
        if unknown_columns:
            logger.warning(f"发现未知列: {list(unknown_columns)}")
        
        # 验证列类型和约束
        for col_name, col_spec in self.column_types.items():
            if col_name not in df.columns:
                continue
            
            col_errors = self._validate_column(df[col_name], col_name, col_spec)
            errors.extend(col_errors)
        
        # 验证业务规则
        for rule in self.validation_rules:
            rule_errors = self._validate_rule(df, rule)
            errors.extend(rule_errors)
        
        return errors
    
    def _validate_column(self, series: pd.Series, col_name: str, 
                        col_spec: Dict[str, Any]) -> List[DataValidationError]:
        """验证单个列"""
        errors = []
        
        # 检查空值
        if not col_spec.get('nullable', True):
            null_mask = series.isnull()
            if null_mask.any():
                null_indices = series.index[null_mask].tolist()
                errors.append(DataValidationError(
                    f"列 {col_name} 包含空值",
                    failed_rules=["not_nullable"],
                    invalid_rows=null_indices,
                    column=col_name
                ))
        
        # 检查数据类型
        expected_type = col_spec.get('type')
        if expected_type and not series.empty:
            type_errors = self._check_data_type(series, col_name, expected_type)
            errors.extend(type_errors)
        
        # 检查数值范围
        if expected_type in ['float', 'integer']:
            range_errors = self._check_numeric_range(series, col_name, col_spec)
            errors.extend(range_errors)
        
        # 检查允许值
        if 'allowed_values' in col_spec:
            value_errors = self._check_allowed_values(series, col_name, col_spec['allowed_values'])
            errors.extend(value_errors)
        
        return errors
    
    def _check_data_type(self, series: pd.Series, col_name: str, 
                        expected_type: str) -> List[DataValidationError]:
        """检查数据类型"""
        errors = []
        
        if expected_type == 'integer':
            # 检查是否为整数（允许浮点形式的整数）
            non_null = series.dropna()
            if not non_null.empty:
                try:
                    int_values = pd.to_numeric(non_null, errors='coerce')
                    invalid_mask = int_values.isnull()
                    if invalid_mask.any():
                        invalid_indices = non_null.index[invalid_mask].tolist()
                        errors.append(DataValidationError(
                            f"列 {col_name} 包含非整数值",
                            failed_rules=["invalid_integer"],
                            invalid_rows=invalid_indices,
                            column=col_name
                        ))
                except Exception:
                    errors.append(DataValidationError(
                        f"列 {col_name} 类型转换失败",
                        failed_rules=["type_conversion_error"],
                        column=col_name
                    ))
        
        elif expected_type == 'float':
            non_null = series.dropna()
            if not non_null.empty:
                try:
                    float_values = pd.to_numeric(non_null, errors='coerce')
                    invalid_mask = float_values.isnull()
                    if invalid_mask.any():
                        invalid_indices = non_null.index[invalid_mask].tolist()
                        errors.append(DataValidationError(
                            f"列 {col_name} 包含非数值",
                            failed_rules=["invalid_float"],
                            invalid_rows=invalid_indices,
                            column=col_name
                        ))
                except Exception:
                    errors.append(DataValidationError(
                        f"列 {col_name} 类型转换失败",
                        failed_rules=["type_conversion_error"],
                        column=col_name
                    ))
        
        elif expected_type == 'string':
            # 字符串类型检查相对宽松
            pass
        
        return errors
    
    def _check_numeric_range(self, series: pd.Series, col_name: str, 
                           col_spec: Dict[str, Any]) -> List[DataValidationError]:
        """检查数值范围"""
        errors = []
        non_null = pd.to_numeric(series.dropna(), errors='coerce')
        
        if 'min_value' in col_spec:
            min_val = col_spec['min_value']
            invalid_mask = non_null < min_val
            if invalid_mask.any():
                invalid_indices = non_null.index[invalid_mask].tolist()
                errors.append(DataValidationError(
                    f"列 {col_name} 包含小于最小值 {min_val} 的数据",
                    failed_rules=["below_min_value"],
                    invalid_rows=invalid_indices,
                    column=col_name
                ))
        
        if 'max_value' in col_spec:
            max_val = col_spec['max_value']
            invalid_mask = non_null > max_val
            if invalid_mask.any():
                invalid_indices = non_null.index[invalid_mask].tolist()
                errors.append(DataValidationError(
                    f"列 {col_name} 包含大于最大值 {max_val} 的数据",
                    failed_rules=["above_max_value"],
                    invalid_rows=invalid_indices,
                    column=col_name
                ))
        
        return errors
    
    def _check_allowed_values(self, series: pd.Series, col_name: str, 
                            allowed_values: List[Any]) -> List[DataValidationError]:
        """检查允许值"""
        errors = []
        non_null = series.dropna()
        
        if not non_null.empty:
            invalid_mask = ~non_null.isin(allowed_values)
            if invalid_mask.any():
                invalid_indices = non_null.index[invalid_mask].tolist()
                invalid_values = non_null[invalid_mask].unique().tolist()
                errors.append(DataValidationError(
                    f"列 {col_name} 包含不允许的值: {invalid_values}",
                    failed_rules=["invalid_values"],
                    invalid_rows=invalid_indices,
                    column=col_name,
                    invalid_values=invalid_values,
                    allowed_values=allowed_values
                ))
        
        return errors
    
    def _validate_rule(self, df: pd.DataFrame, rule: Dict[str, Any]) -> List[DataValidationError]:
        """验证业务规则"""
        errors = []
        rule_name = rule.get('rule')
        
        if rule_name == 'time_monotonic':
            errors.extend(self._check_time_monotonic(df))
        elif rule_name == 'no_duplicate_times':
            errors.extend(self._check_no_duplicate_times(df))
        elif rule_name == 'min_observations_per_id':
            min_count = rule.get('min_count', 3)
            errors.extend(self._check_min_observations(df, min_count))
        elif rule_name == 'interval_consistency':
            errors.extend(self._check_interval_consistency(df))
        elif rule_name == 'censor_type_consistency':
            errors.extend(self._check_censor_type_consistency(df))
        elif rule_name == 'unique_ids':
            errors.extend(self._check_unique_ids(df))
        # 可以添加更多规则
        
        return errors
    
    def _check_time_monotonic(self, df: pd.DataFrame) -> List[DataValidationError]:
        """检查时间单调性"""
        errors = []
        
        if 'id' in df.columns and 't' in df.columns:
            for patient_id, group in df.groupby('id'):
                time_values = group['t'].values
                if len(time_values) > 1:
                    if not np.all(np.diff(time_values) > 0):
                        invalid_indices = group.index.tolist()
                        errors.append(DataValidationError(
                            f"患者 {patient_id} 的时间序列不是严格递增的",
                            failed_rules=["time_not_monotonic"],
                            invalid_rows=invalid_indices,
                            patient_id=patient_id
                        ))
        
        return errors
    
    def _check_no_duplicate_times(self, df: pd.DataFrame) -> List[DataValidationError]:
        """检查重复时间点"""
        errors = []
        
        if 'id' in df.columns and 't' in df.columns:
            duplicates = df.duplicated(['id', 't'])
            if duplicates.any():
                duplicate_indices = df.index[duplicates].tolist()
                errors.append(DataValidationError(
                    "存在重复的(id, t)组合",
                    failed_rules=["duplicate_times"],
                    invalid_rows=duplicate_indices
                ))
        
        return errors
    
    def _check_min_observations(self, df: pd.DataFrame, min_count: int) -> List[DataValidationError]:
        """检查最小观测数"""
        errors = []
        
        if 'id' in df.columns:
            counts = df['id'].value_counts()
            insufficient_ids = counts[counts < min_count].index.tolist()
            
            if insufficient_ids:
                errors.append(DataValidationError(
                    f"{len(insufficient_ids)} 个患者的观测数少于 {min_count}",
                    failed_rules=["insufficient_observations"],
                    insufficient_ids=insufficient_ids,
                    min_count=min_count
                ))
        
        return errors
    
    def _check_interval_consistency(self, df: pd.DataFrame) -> List[DataValidationError]:
        """检查区间一致性"""
        errors = []
        
        if 'L' in df.columns and 'R' in df.columns:
            # L <= R 对于非空R值
            valid_R = df['R'].notna()
            invalid_intervals = (df.loc[valid_R, 'L'] > df.loc[valid_R, 'R'])
            
            if invalid_intervals.any():
                invalid_indices = df.index[valid_R][invalid_intervals].tolist()
                errors.append(DataValidationError(
                    "存在 L > R 的无效区间",
                    failed_rules=["invalid_intervals"],
                    invalid_rows=invalid_indices
                ))
        
        return errors
    
    def _check_censor_type_consistency(self, df: pd.DataFrame) -> List[DataValidationError]:
        """检查删失类型一致性"""
        errors = []
        
        if all(col in df.columns for col in ['L', 'R', 'censor_type']):
            for idx, row in df.iterrows():
                censor_type = row['censor_type']
                L, R = row['L'], row['R']
                
                is_valid = True
                if censor_type == 'interval':
                    is_valid = pd.notna(L) and pd.notna(R) and L <= R
                elif censor_type == 'left':
                    is_valid = pd.notna(R) and pd.isna(L)
                elif censor_type == 'right':
                    is_valid = pd.notna(L) and pd.isna(R)
                elif censor_type == 'exact':
                    is_valid = pd.notna(L) and (pd.isna(R) or L == R)
                
                if not is_valid:
                    errors.append(DataValidationError(
                        f"行 {idx}: 删失类型 '{censor_type}' 与 L={L}, R={R} 不一致",
                        failed_rules=["censor_type_inconsistent"],
                        invalid_rows=[idx],
                        censor_type=censor_type,
                        L=L,
                        R=R
                    ))
        
        return errors
    
    def _check_unique_ids(self, df: pd.DataFrame) -> List[DataValidationError]:
        """检查ID唯一性"""
        errors = []
        
        if 'id' in df.columns:
            duplicates = df.duplicated('id')
            if duplicates.any():
                duplicate_indices = df.index[duplicates].tolist()
                duplicate_ids = df.loc[duplicates, 'id'].tolist()
                errors.append(DataValidationError(
                    f"存在重复ID: {duplicate_ids}",
                    failed_rules=["duplicate_ids"],
                    invalid_rows=duplicate_indices,
                    duplicate_ids=duplicate_ids
                ))
        
        return errors


def load_data(file_path: Union[str, Path], 
              schema_path: Optional[Union[str, Path]] = None,
              validate: bool = True,
              **kwargs) -> pd.DataFrame:
    """
    加载数据文件并可选地进行schema验证
    
    Args:
        file_path: 数据文件路径
        schema_path: schema文件路径
        validate: 是否进行验证
        **kwargs: pandas读取参数
        
    Returns:
        加载的DataFrame
        
    Raises:
        DataValidationError: 数据验证失败
        MCMError: 文件读取失败
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise MCMError(f"数据文件不存在: {file_path}")
    
    try:
        with timer(f"加载数据文件 {file_path.name}", logger):
            if file_path.suffix.lower() == '.csv':
                df = pd.read_csv(file_path, **kwargs)
            elif file_path.suffix.lower() in ['.xlsx', '.xls']:
                df = pd.read_excel(file_path, **kwargs)
            elif file_path.suffix.lower() == '.parquet':
                df = pd.read_parquet(file_path, **kwargs)
            elif file_path.suffix.lower() == '.pkl':
                df = pd.read_pickle(file_path)
            else:
                raise MCMError(f"不支持的文件格式: {file_path.suffix}")
        
        logger.info(f"成功加载数据: {df.shape[0]} 行, {df.shape[1]} 列")
        
        # Schema验证
        if validate and schema_path:
            schema = DataSchema(schema_path)
            errors = schema.validate_dataframe(df, strict=False)
            
            if errors:
                error_summary = f"数据验证失败，发现 {len(errors)} 个错误"
                logger.error(error_summary)
                
                # 记录详细错误信息
                for error in errors:
                    logger.error(f"  - {error.message}")
                
                # 可以选择是否抛出异常
                if any(rule in error.failed_rules 
                      for error in errors 
                      for rule in ["missing_required_columns", "type_conversion_error"]):
                    raise DataValidationError(error_summary, failed_rules=[e.failed_rules for e in errors])
                else:
                    logger.warning("存在数据质量问题，但不影响主要功能")
        
        return df
        
    except Exception as e:
        if isinstance(e, (DataValidationError, MCMError)):
            raise
        else:
            raise MCMError(f"数据加载失败: {e}", file_path=str(file_path))


def save_data(df: pd.DataFrame, 
              file_path: Union[str, Path],
              create_dirs: bool = True,
              **kwargs) -> None:
    """
    保存DataFrame到文件
    
    Args:
        df: 要保存的DataFrame
        file_path: 保存路径
        create_dirs: 是否创建目录
        **kwargs: pandas保存参数
    """
    file_path = Path(file_path)
    
    if create_dirs:
        file_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        with timer(f"保存数据到 {file_path.name}", logger):
            if file_path.suffix.lower() == '.csv':
                df.to_csv(file_path, index=False, **kwargs)
            elif file_path.suffix.lower() in ['.xlsx', '.xls']:
                df.to_excel(file_path, index=False, **kwargs)
            elif file_path.suffix.lower() == '.parquet':
                df.to_parquet(file_path, index=False, **kwargs)
            elif file_path.suffix.lower() == '.pkl':
                df.to_pickle(file_path)
            else:
                raise MCMError(f"不支持的保存格式: {file_path.suffix}")
        
        logger.info(f"成功保存数据: {df.shape[0]} 行, {df.shape[1]} 列")
        
    except Exception as e:
        raise MCMError(f"数据保存失败: {e}", file_path=str(file_path))


def load_model(file_path: Union[str, Path]) -> Any:
    """
    加载模型文件
    
    Args:
        file_path: 模型文件路径
        
    Returns:
        加载的模型对象
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise MCMError(f"模型文件不存在: {file_path}")
    
    try:
        with open(file_path, 'rb') as f:
            model = pickle.load(f)
        
        logger.info(f"成功加载模型: {file_path.name}")
        return model
        
    except Exception as e:
        raise MCMError(f"模型加载失败: {e}", file_path=str(file_path))


def save_model(model: Any, file_path: Union[str, Path],
               create_dirs: bool = True) -> None:
    """
    保存模型到文件
    
    Args:
        model: 要保存的模型对象
        file_path: 保存路径
        create_dirs: 是否创建目录
    """
    file_path = Path(file_path)
    
    if create_dirs:
        file_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        with open(file_path, 'wb') as f:
            pickle.dump(model, f)
        
        logger.info(f"成功保存模型: {file_path.name}")
        
    except Exception as e:
        raise MCMError(f"模型保存失败: {e}", file_path=str(file_path))


def batch_process_data(df: pd.DataFrame, 
                      process_func: callable,
                      chunk_size: int = 1000,
                      **kwargs) -> pd.DataFrame:
    """
    分批处理大型DataFrame
    
    Args:
        df: 输入DataFrame
        process_func: 处理函数
        chunk_size: 批次大小
        **kwargs: 传递给处理函数的参数
        
    Returns:
        处理后的DataFrame
    """
    if len(df) <= chunk_size:
        return process_func(df, **kwargs)
    
    results = []
    n_chunks = (len(df) + chunk_size - 1) // chunk_size
    
    logger.info(f"分批处理数据: {len(df)} 行分为 {n_chunks} 批")
    
    for i in range(0, len(df), chunk_size):
        chunk = df.iloc[i:i+chunk_size].copy()
        
        try:
            processed_chunk = process_func(chunk, **kwargs)
            results.append(processed_chunk)
            
            if (i // chunk_size + 1) % 10 == 0:
                logger.info(f"已处理 {i // chunk_size + 1}/{n_chunks} 批")
                
        except Exception as e:
            logger.error(f"处理第 {i//chunk_size + 1} 批时出错: {e}")
            raise
    
    return pd.concat(results, ignore_index=True)


def validate_data_consistency(long_df: pd.DataFrame, 
                            surv_df: pd.DataFrame) -> List[DataValidationError]:
    """
    验证长表和生存数据的一致性
    
    Args:
        long_df: 长格式数据
        surv_df: 生存数据
        
    Returns:
        一致性验证错误列表
    """
    errors = []
    
    # 检查ID一致性
    long_ids = set(long_df['id'].unique()) if 'id' in long_df.columns else set()
    surv_ids = set(surv_df['id'].unique()) if 'id' in surv_df.columns else set()
    
    missing_in_surv = long_ids - surv_ids
    missing_in_long = surv_ids - long_ids
    
    if missing_in_surv:
        errors.append(DataValidationError(
            f"长表中的 {len(missing_in_surv)} 个ID在生存数据中缺失",
            failed_rules=["missing_ids_in_survival"],
            missing_ids=list(missing_in_surv)
        ))
    
    if missing_in_long:
        errors.append(DataValidationError(
            f"生存数据中的 {len(missing_in_long)} 个ID在长表中缺失",
            failed_rules=["missing_ids_in_longitudinal"],
            missing_ids=list(missing_in_long)
        ))
    
    # 检查BMI一致性（如果都有BMI相关列）
    if 'BMI' in long_df.columns and 'BMI_base' in surv_df.columns:
        common_ids = long_ids & surv_ids
        
        for patient_id in common_ids:
            long_bmi = long_df[long_df['id'] == patient_id]['BMI'].iloc[0]
            surv_bmi = surv_df[surv_df['id'] == patient_id]['BMI_base'].iloc[0]
            
            if abs(long_bmi - surv_bmi) > 0.1:  # 允许小的数值误差
                errors.append(DataValidationError(
                    f"患者 {patient_id} 的BMI不一致: 长表={long_bmi:.2f}, 生存数据={surv_bmi:.2f}",
                    failed_rules=["bmi_inconsistent"],
                    patient_id=patient_id,
                    long_bmi=long_bmi,
                    surv_bmi=surv_bmi
                ))
    
    return errors


def create_data_summary(df: pd.DataFrame, name: str = "数据") -> Dict[str, Any]:
    """
    创建数据摘要信息
    
    Args:
        df: 输入DataFrame
        name: 数据名称
        
    Returns:
        数据摘要字典
    """
    summary = {
        "name": name,
        "shape": df.shape,
        "columns": list(df.columns),
        "dtypes": df.dtypes.to_dict(),
        "missing_values": df.isnull().sum().to_dict(),
        "memory_usage_mb": df.memory_usage(deep=True).sum() / 1024 / 1024
    }
    
    # 数值列的基本统计
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        summary["numeric_summary"] = df[numeric_cols].describe().to_dict()
    
    # 分类列的基本信息
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    if len(categorical_cols) > 0:
        summary["categorical_summary"] = {}
        for col in categorical_cols:
            summary["categorical_summary"][col] = {
                "unique_count": df[col].nunique(),
                "top_values": df[col].value_counts().head().to_dict()
            }
    
    return summary