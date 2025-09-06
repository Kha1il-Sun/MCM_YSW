"""
通用工具函数模块

包含随机种子管理、计时、数值计算等通用功能。
"""

import time
import hashlib
import warnings
from contextlib import contextmanager
from typing import Any, Dict, Generator, Optional, Union
import numpy as np
import pandas as pd
from functools import wraps


def set_random_seeds(seed: int = 42) -> None:
    """
    设置全局随机种子，确保结果可复现
    
    Args:
        seed: 随机种子值
    """
    np.random.seed(seed)
    try:
        import random
        random.seed(seed)
    except ImportError:
        pass
    
    # 如果有其他随机数生成器，也在这里设置
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def rng_for_id(global_seed: int, uid: Union[str, int]) -> np.random.Generator:
    """
    为特定ID生成稳定的随机数生成器
    
    Args:
        global_seed: 全局种子
        uid: 用户/样本ID
    
    Returns:
        numpy随机数生成器
    """
    # 使用哈希函数确保ID映射的一致性
    uid_hash = int(hashlib.md5(str(uid).encode()).hexdigest()[:8], 16)
    combined_seed = (uid_hash ^ global_seed) & 0xFFFFFFFF
    return np.random.default_rng(combined_seed)


def logdiffexp(a: float, b: float) -> float:
    """
    数值稳定的 log(exp(a) - exp(b)) 计算
    
    Args:
        a: 较大的对数值
        b: 较小的对数值
    
    Returns:
        log(exp(a) - exp(b))
    
    Raises:
        ValueError: 如果 b > a
    """
    if b > a:
        raise ValueError(f"要求 a > b，但得到 a={a}, b={b}")
    
    if np.isinf(b) and b < 0:  # b = -inf
        return a
    
    if a == b:
        return -np.inf
    
    return a + np.log1p(-np.exp(b - a))


def log_sum_exp(x: np.ndarray, axis: Optional[int] = None) -> Union[float, np.ndarray]:
    """
    数值稳定的 log(sum(exp(x))) 计算
    
    Args:
        x: 输入数组
        axis: 求和的轴
    
    Returns:
        log(sum(exp(x)))
    """
    x_max = np.max(x, axis=axis, keepdims=True)
    if axis is not None:
        x_max = np.squeeze(x_max, axis=axis)
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = x_max + np.log(np.sum(np.exp(x - np.expand_dims(x_max, axis)), axis=axis))
    
    return result


@contextmanager
def timer(description: str = "Operation", logger=None) -> Generator[Dict[str, Any], None, None]:
    """
    计时上下文管理器
    
    Args:
        description: 操作描述
        logger: 日志记录器
    
    Yields:
        包含计时信息的字典
    """
    start_time = time.time()
    timing_info = {"start_time": start_time}
    
    try:
        yield timing_info
    finally:
        end_time = time.time()
        elapsed = end_time - start_time
        timing_info.update({
            "end_time": end_time,
            "elapsed": elapsed,
            "elapsed_str": format_time(elapsed)
        })
        
        message = f"{description} completed in {timing_info['elapsed_str']}"
        if logger:
            logger.info(message)
        else:
            print(message)


def format_time(seconds: float) -> str:
    """
    格式化时间显示
    
    Args:
        seconds: 秒数
    
    Returns:
        格式化的时间字符串
    """
    if seconds < 1:
        return f"{seconds*1000:.1f}ms"
    elif seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}m{secs:.1f}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        return f"{hours}h{minutes}m{secs:.1f}s"


def memory_usage() -> Dict[str, float]:
    """
    获取当前内存使用情况
    
    Returns:
        内存使用信息字典
    """
    try:
        import psutil
        process = psutil.Process()
        memory_info = process.memory_info()
        return {
            "rss_mb": memory_info.rss / 1024 / 1024,  # 物理内存
            "vms_mb": memory_info.vms / 1024 / 1024,  # 虚拟内存
            "percent": process.memory_percent()        # 内存使用百分比
        }
    except ImportError:
        return {"rss_mb": 0, "vms_mb": 0, "percent": 0}


def check_memory_limit(limit_gb: float, logger=None) -> bool:
    """
    检查内存使用是否超过限制
    
    Args:
        limit_gb: 内存限制（GB）
        logger: 日志记录器
    
    Returns:
        是否超过限制
    """
    usage = memory_usage()
    current_gb = usage["rss_mb"] / 1024
    
    if current_gb > limit_gb:
        message = f"内存使用超过限制: {current_gb:.2f}GB > {limit_gb}GB"
        if logger:
            logger.warning(message)
        else:
            warnings.warn(message)
        return True
    
    return False


def safe_divide(numerator: np.ndarray, denominator: np.ndarray, 
                fill_value: float = 0.0) -> np.ndarray:
    """
    安全除法，避免除零错误
    
    Args:
        numerator: 分子
        denominator: 分母
        fill_value: 除零时的填充值
    
    Returns:
        除法结果
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        result = np.divide(numerator, denominator)
        result = np.where(np.isfinite(result), result, fill_value)
    return result


def ensure_numpy(x: Any) -> np.ndarray:
    """
    确保输入转换为numpy数组
    
    Args:
        x: 输入数据
    
    Returns:
        numpy数组
    """
    if isinstance(x, np.ndarray):
        return x
    elif isinstance(x, (list, tuple)):
        return np.array(x)
    elif isinstance(x, pd.Series):
        return x.values
    elif np.isscalar(x):
        return np.array([x])
    else:
        return np.array(x)


def clip_to_bounds(x: np.ndarray, bounds: tuple, 
                  warn: bool = True, logger=None) -> np.ndarray:
    """
    将数值限制在指定范围内
    
    Args:
        x: 输入数组
        bounds: (min_val, max_val) 边界
        warn: 是否在截断时发出警告
        logger: 日志记录器
    
    Returns:
        截断后的数组
    """
    min_val, max_val = bounds
    x_clipped = np.clip(x, min_val, max_val)
    
    if warn:
        n_clipped = np.sum((x < min_val) | (x > max_val))
        if n_clipped > 0:
            message = f"截断了 {n_clipped} 个值到范围 [{min_val}, {max_val}]"
            if logger:
                logger.warning(message)
            else:
                warnings.warn(message)
    
    return x_clipped


def retry_on_failure(max_attempts: int = 3, delay: float = 1.0, 
                    exceptions: tuple = (Exception,)):
    """
    失败重试装饰器
    
    Args:
        max_attempts: 最大尝试次数
        delay: 重试间隔（秒）
        exceptions: 需要重试的异常类型
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_attempts - 1:
                        time.sleep(delay * (2 ** attempt))  # 指数退避
                        continue
                    break
            
            raise last_exception
        return wrapper
    return decorator


class ProgressTracker:
    """简单的进度跟踪器"""
    
    def __init__(self, total: int, description: str = "Processing", 
                 update_interval: int = 100):
        self.total = total
        self.description = description
        self.update_interval = update_interval
        self.current = 0
        self.start_time = time.time()
        self.last_update = 0
    
    def update(self, step: int = 1) -> None:
        """更新进度"""
        self.current += step
        
        if (self.current - self.last_update >= self.update_interval or 
            self.current >= self.total):
            self._print_progress()
            self.last_update = self.current
    
    def _print_progress(self) -> None:
        """打印进度信息"""
        elapsed = time.time() - self.start_time
        if self.current > 0:
            eta = elapsed * (self.total - self.current) / self.current
            eta_str = format_time(eta)
        else:
            eta_str = "Unknown"
        
        percentage = 100.0 * self.current / self.total
        print(f"\r{self.description}: {self.current}/{self.total} "
              f"({percentage:.1f}%) ETA: {eta_str}", end="", flush=True)
        
        if self.current >= self.total:
            print()  # 换行


def validate_probability(p: np.ndarray, name: str = "probability") -> np.ndarray:
    """
    验证概率值的有效性
    
    Args:
        p: 概率数组
        name: 变量名称，用于错误消息
    
    Returns:
        验证后的概率数组
    
    Raises:
        ValueError: 如果概率值无效
    """
    p = ensure_numpy(p)
    
    if np.any(p < 0) or np.any(p > 1):
        raise ValueError(f"{name} 必须在 [0, 1] 范围内")
    
    if np.any(np.isnan(p)):
        raise ValueError(f"{name} 包含 NaN 值")
    
    return p


def create_grid(bounds: tuple, n_points: int, 
               log_scale: bool = False) -> np.ndarray:
    """
    创建网格点
    
    Args:
        bounds: (min_val, max_val) 边界
        n_points: 网格点数量
        log_scale: 是否使用对数尺度
    
    Returns:
        网格点数组
    """
    min_val, max_val = bounds
    
    if log_scale:
        if min_val <= 0:
            raise ValueError("对数尺度要求最小值 > 0")
        return np.logspace(np.log10(min_val), np.log10(max_val), n_points)
    else:
        return np.linspace(min_val, max_val, n_points)