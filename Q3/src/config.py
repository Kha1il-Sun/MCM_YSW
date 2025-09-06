"""
配置管理模块

使用pydantic进行配置验证和类型检查。
"""

import os
import yaml
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field, validator, root_validator
import logging.config

from .exceptions import ConfigError


class WeightsConfig(BaseModel):
    """风险权重配置"""
    w1: float = Field(1.0, ge=0, description="未达标惩罚权重")
    w2: float = Field(0.6, ge=0, description="延迟惩罚权重") 
    w3: float = Field(0.3, ge=0, description="不确定性惩罚权重")


class DelayConfig(BaseModel):
    """延迟惩罚配置"""
    pref_week: float = Field(15.0, gt=0, description="偏好时间点")
    scale: float = Field(10.0, gt=0, description="延迟惩罚尺度参数")


class MIConfig(BaseModel):
    """多重插补配置"""
    M: int = Field(20, ge=1, le=100, description="插补次数")
    q: float = Field(0.02, ge=0, le=1, description="观测误差率")
    use_local_sigma: bool = Field(True, description="是否使用局部sigma")
    deterministic_by_id: bool = Field(True, description="按ID确定性播种")


class EnsembleConfig(BaseModel):
    """集成模型配置"""
    enable: bool = Field(True, description="是否启用集成")
    method: str = Field("stacking", regex="^(avg|stacking)$", description="集成方法")
    penalty: float = Field(1e-3, ge=0, description="正则化惩罚参数")


class AFTConfig(BaseModel):
    """AFT模型配置"""
    families: List[str] = Field(
        ["lognormal", "loglogistic", "weibull"],
        description="分布族列表"
    )
    select_by: str = Field("AIC", regex="^(AIC|BIC)$", description="模型选择准则")
    ic_cox_check: bool = Field(True, description="是否进行IC-Cox检查")
    ensemble: EnsembleConfig = Field(default_factory=EnsembleConfig)
    
    @validator('families')
    def validate_families(cls, v):
        valid_families = {"lognormal", "loglogistic", "weibull", "exponential", "gamma"}
        invalid = set(v) - valid_families
        if invalid:
            raise ValueError(f"无效的分布族: {invalid}")
        return v


class GroupingConfig(BaseModel):
    """分组配置"""
    method: str = Field("dp", regex="^(dp|cart)$", description="分组方法")
    K: int = Field(4, ge=2, le=10, description="分组数量")
    min_group_size: int = Field(40, ge=10, description="最小组大小")


class SigmaSweepConfig(BaseModel):
    """Sigma扫描配置"""
    factors: List[float] = Field(
        [0.5, 1.0, 1.5, 2.0, 5.0],
        description="sigma倍数因子"
    )
    base_sigma: float = Field(1.0, gt=0, description="基础sigma值")
    sweep_mode: str = Field("multiplicative", regex="^(multiplicative|additive)$")
    
    @validator('factors')
    def validate_factors(cls, v):
        if any(f <= 0 for f in v):
            raise ValueError("所有sigma因子必须为正数")
        return sorted(v)


class PerformanceConfig(BaseModel):
    """性能配置"""
    chunk_size: int = Field(2000, ge=100, description="数据块大小")
    max_workers: int = Field(4, ge=1, le=32, description="最大工作线程数")
    memory_limit_gb: float = Field(8.0, gt=0, description="内存限制(GB)")
    cache_dir: str = Field(".cache", description="缓存目录")
    use_joblib: bool = Field(True, description="是否使用joblib并行")
    
    @validator('max_workers')
    def validate_max_workers(cls, v):
        import multiprocessing
        max_cpu = multiprocessing.cpu_count()
        if v > max_cpu:
            raise ValueError(f"max_workers ({v}) 超过CPU核心数 ({max_cpu})")
        return v


class LoggingConfig(BaseModel):
    """日志配置"""
    level: str = Field("INFO", regex="^(DEBUG|INFO|WARNING|ERROR|CRITICAL)$")
    file: str = Field("logs/mcm_q3.log", description="日志文件路径")
    rotate_max_mb: int = Field(100, ge=1, description="日志文件最大大小(MB)")
    backup_count: int = Field(5, ge=1, description="日志备份数量")


class ValidationConfig(BaseModel):
    """验证配置"""
    cv_folds: int = Field(5, ge=2, le=10, description="交叉验证折数")
    bootstrap_samples: int = Field(100, ge=10, description="Bootstrap采样次数")
    test_size: float = Field(0.2, gt=0, lt=1, description="测试集比例")
    stratify_by: str = Field("BMI_quartile", description="分层变量")


class PlotsConfig(BaseModel):
    """绘图配置"""
    dpi: int = Field(300, ge=72, description="图像DPI")
    figsize: List[float] = Field([10, 8], description="图像大小")
    style: str = Field("seaborn-v0_8", description="绘图风格")
    save_formats: List[str] = Field(["png", "pdf"], description="保存格式")
    
    @validator('figsize')
    def validate_figsize(cls, v):
        if len(v) != 2 or any(x <= 0 for x in v):
            raise ValueError("figsize必须是两个正数")
        return v
    
    @validator('save_formats')
    def validate_formats(cls, v):
        valid_formats = {"png", "pdf", "svg", "eps", "jpg", "jpeg"}
        invalid = set(v) - valid_formats
        if invalid:
            raise ValueError(f"无效的保存格式: {invalid}")
        return v


class MCMConfig(BaseModel):
    """主配置类"""
    seed: int = Field(42, ge=0, description="随机种子")
    threshold: float = Field(0.04, gt=0, lt=1, description="达标阈值")
    clinical_bounds: List[float] = Field([12.0, 25.0], description="临床范围")
    tau: float = Field(0.90, gt=0, lt=1, description="概率门槛")
    
    weights: WeightsConfig = Field(default_factory=WeightsConfig)
    delay: DelayConfig = Field(default_factory=DelayConfig)
    mi: MIConfig = Field(default_factory=MIConfig)
    aft: AFTConfig = Field(default_factory=AFTConfig)
    grouping: GroupingConfig = Field(default_factory=GroupingConfig)
    sigma_sweep: SigmaSweepConfig = Field(default_factory=SigmaSweepConfig)
    performance: PerformanceConfig = Field(default_factory=PerformanceConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    validation: ValidationConfig = Field(default_factory=ValidationConfig)
    plots: PlotsConfig = Field(default_factory=PlotsConfig)
    
    @validator('clinical_bounds')
    def validate_clinical_bounds(cls, v):
        if len(v) != 2 or v[0] >= v[1]:
            raise ValueError("clinical_bounds必须是[min, max]格式，且min < max")
        return v
    
    @root_validator
    def validate_consistency(cls, values):
        """验证配置项之间的一致性"""
        # 检查tau和threshold的合理性
        tau = values.get('tau', 0.9)
        threshold = values.get('threshold', 0.04)
        
        if tau > 0.99:
            raise ValueError("tau过高可能导致无解")
        
        if threshold > 0.1:
            raise ValueError("threshold过高可能不合理")
        
        return values
    
    class Config:
        """Pydantic配置"""
        extra = "forbid"  # 禁止额外字段
        validate_assignment = True  # 赋值时验证


def load_config(config_path: Union[str, Path]) -> MCMConfig:
    """
    加载配置文件
    
    Args:
        config_path: 配置文件路径
        
    Returns:
        验证后的配置对象
        
    Raises:
        ConfigError: 配置文件错误
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise ConfigError(f"配置文件不存在: {config_path}")
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            if config_path.suffix.lower() in ['.yaml', '.yml']:
                config_data = yaml.safe_load(f)
            else:
                raise ConfigError(f"不支持的配置文件格式: {config_path.suffix}")
        
        return MCMConfig(**config_data)
        
    except yaml.YAMLError as e:
        raise ConfigError(f"YAML解析错误: {e}", config_path=str(config_path))
    except Exception as e:
        raise ConfigError(f"配置验证失败: {e}", config_path=str(config_path))


def setup_logging(config: Union[MCMConfig, LoggingConfig, str, Path]) -> None:
    """
    设置日志系统
    
    Args:
        config: 配置对象、日志配置或配置文件路径
    """
    if isinstance(config, (str, Path)):
        # 如果是路径，尝试加载专门的logging配置
        config_path = Path(config)
        if config_path.name == "logging.yaml":
            with open(config_path, 'r', encoding='utf-8') as f:
                logging_config = yaml.safe_load(f)
            logging.config.dictConfig(logging_config)
            return
        else:
            # 加载主配置文件
            main_config = load_config(config_path)
            logging_config = main_config.logging
    elif isinstance(config, MCMConfig):
        logging_config = config.logging
    elif isinstance(config, LoggingConfig):
        logging_config = config
    else:
        raise ConfigError(f"无效的配置类型: {type(config)}")
    
    # 创建日志目录
    log_file = Path(logging_config.file)
    log_file.parent.mkdir(parents=True, exist_ok=True)
    
    # 配置根日志记录器
    logging.basicConfig(
        level=getattr(logging, logging_config.level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # 添加文件处理器
    from logging.handlers import RotatingFileHandler
    file_handler = RotatingFileHandler(
        logging_config.file,
        maxBytes=logging_config.rotate_max_mb * 1024 * 1024,
        backupCount=logging_config.backup_count,
        encoding='utf-8'
    )
    file_handler.setFormatter(
        logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(module)s:%(funcName)s:%(lineno)d - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
    )
    
    root_logger = logging.getLogger()
    root_logger.addHandler(file_handler)


def get_config_summary(config: MCMConfig) -> Dict[str, Any]:
    """
    获取配置摘要信息
    
    Args:
        config: 配置对象
        
    Returns:
        配置摘要字典
    """
    return {
        "seed": config.seed,
        "threshold": config.threshold,
        "tau": config.tau,
        "mi_samples": config.mi.M,
        "aft_families": config.aft.families,
        "grouping_method": config.grouping.method,
        "grouping_K": config.grouping.K,
        "max_workers": config.performance.max_workers,
        "memory_limit_gb": config.performance.memory_limit_gb,
    }


def validate_paths(config: MCMConfig, base_dir: Optional[Path] = None) -> None:
    """
    验证配置中的路径是否有效
    
    Args:
        config: 配置对象
        base_dir: 基础目录，用于相对路径解析
    """
    base_dir = base_dir or Path.cwd()
    
    # 检查缓存目录
    cache_dir = base_dir / config.performance.cache_dir
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        raise ConfigError(f"无法创建缓存目录 {cache_dir}: {e}")
    
    # 检查日志目录
    log_file = base_dir / config.logging.file
    try:
        log_file.parent.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        raise ConfigError(f"无法创建日志目录 {log_file.parent}: {e}")


def merge_configs(base_config: MCMConfig, override_dict: Dict[str, Any]) -> MCMConfig:
    """
    合并配置，支持嵌套字典更新
    
    Args:
        base_config: 基础配置
        override_dict: 覆盖配置字典
        
    Returns:
        合并后的配置
    """
    def deep_update(base_dict: dict, update_dict: dict) -> dict:
        """深度更新字典"""
        result = base_dict.copy()
        for key, value in update_dict.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = deep_update(result[key], value)
            else:
                result[key] = value
        return result
    
    base_dict = base_config.dict()
    merged_dict = deep_update(base_dict, override_dict)
    
    return MCMConfig(**merged_dict)