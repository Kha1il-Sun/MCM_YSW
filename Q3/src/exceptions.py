"""
统一异常处理模块

定义了项目中使用的所有自定义异常类，提供统一的错误处理机制。
"""

from typing import Any, Dict, List, Optional


class MCMError(Exception):
    """项目基础异常类"""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        self.message = message
        self.details = details or {}
        super().__init__(self.message)
    
    def __str__(self) -> str:
        if self.details:
            details_str = ", ".join([f"{k}={v}" for k, v in self.details.items()])
            return f"{self.message} (Details: {details_str})"
        return self.message


class DataValidationError(MCMError):
    """数据校验异常"""
    
    def __init__(
        self, 
        message: str, 
        failed_rules: Optional[List[str]] = None,
        invalid_rows: Optional[List[int]] = None,
        column: Optional[str] = None,
        **kwargs
    ):
        details = {
            "failed_rules": failed_rules or [],
            "invalid_rows": invalid_rows or [],
            "column": column,
            **kwargs
        }
        super().__init__(message, details)
        self.failed_rules = failed_rules or []
        self.invalid_rows = invalid_rows or []
        self.column = column


class ModelFittingError(MCMError):
    """模型拟合异常"""
    
    def __init__(
        self, 
        message: str, 
        model_type: Optional[str] = None,
        convergence_info: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        details = {
            "model_type": model_type,
            "convergence_info": convergence_info,
            **kwargs
        }
        super().__init__(message, details)
        self.model_type = model_type
        self.convergence_info = convergence_info


class ConfigError(MCMError):
    """配置文件异常"""
    
    def __init__(
        self, 
        message: str, 
        config_path: Optional[str] = None,
        invalid_keys: Optional[List[str]] = None,
        **kwargs
    ):
        details = {
            "config_path": config_path,
            "invalid_keys": invalid_keys or [],
            **kwargs
        }
        super().__init__(message, details)
        self.config_path = config_path
        self.invalid_keys = invalid_keys or []


class ComputationError(MCMError):
    """计算过程异常"""
    
    def __init__(
        self, 
        message: str, 
        operation: Optional[str] = None,
        input_shape: Optional[tuple] = None,
        **kwargs
    ):
        details = {
            "operation": operation,
            "input_shape": input_shape,
            **kwargs
        }
        super().__init__(message, details)
        self.operation = operation
        self.input_shape = input_shape


class OptimizationError(MCMError):
    """优化算法异常"""
    
    def __init__(
        self, 
        message: str, 
        optimizer: Optional[str] = None,
        iteration: Optional[int] = None,
        objective_value: Optional[float] = None,
        **kwargs
    ):
        details = {
            "optimizer": optimizer,
            "iteration": iteration,
            "objective_value": objective_value,
            **kwargs
        }
        super().__init__(message, details)
        self.optimizer = optimizer
        self.iteration = iteration
        self.objective_value = objective_value


class ValidationError(MCMError):
    """模型验证异常"""
    
    def __init__(
        self, 
        message: str, 
        validation_type: Optional[str] = None,
        fold: Optional[int] = None,
        metric_values: Optional[Dict[str, float]] = None,
        **kwargs
    ):
        details = {
            "validation_type": validation_type,
            "fold": fold,
            "metric_values": metric_values,
            **kwargs
        }
        super().__init__(message, details)
        self.validation_type = validation_type
        self.fold = fold
        self.metric_values = metric_values


class ResourceError(MCMError):
    """资源相关异常（内存、磁盘、CPU等）"""
    
    def __init__(
        self, 
        message: str, 
        resource_type: Optional[str] = None,
        current_usage: Optional[float] = None,
        limit: Optional[float] = None,
        **kwargs
    ):
        details = {
            "resource_type": resource_type,
            "current_usage": current_usage,
            "limit": limit,
            **kwargs
        }
        super().__init__(message, details)
        self.resource_type = resource_type
        self.current_usage = current_usage
        self.limit = limit


class ParallelizationError(MCMError):
    """并行处理异常"""
    
    def __init__(
        self, 
        message: str, 
        worker_id: Optional[int] = None,
        n_workers: Optional[int] = None,
        task_info: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        details = {
            "worker_id": worker_id,
            "n_workers": n_workers,
            "task_info": task_info,
            **kwargs
        }
        super().__init__(message, details)
        self.worker_id = worker_id
        self.n_workers = n_workers
        self.task_info = task_info


# 异常处理工具函数
def format_validation_errors(errors: List[DataValidationError]) -> str:
    """格式化多个数据验证错误为可读字符串"""
    if not errors:
        return "No validation errors."
    
    formatted_errors = []
    for i, error in enumerate(errors, 1):
        error_info = [f"Error {i}: {error.message}"]
        
        if error.failed_rules:
            error_info.append(f"  Failed rules: {', '.join(error.failed_rules)}")
        
        if error.invalid_rows:
            row_count = len(error.invalid_rows)
            if row_count <= 10:
                error_info.append(f"  Invalid rows: {error.invalid_rows}")
            else:
                sample_rows = error.invalid_rows[:5] + ["..."] + error.invalid_rows[-5:]
                error_info.append(f"  Invalid rows ({row_count} total): {sample_rows}")
        
        if error.column:
            error_info.append(f"  Column: {error.column}")
        
        formatted_errors.append("\n".join(error_info))
    
    return "\n\n".join(formatted_errors)


def handle_exception(
    exception: Exception, 
    logger=None, 
    reraise: bool = True,
    context: Optional[Dict[str, Any]] = None
) -> None:
    """
    统一异常处理函数
    
    Args:
        exception: 要处理的异常
        logger: 日志记录器
        reraise: 是否重新抛出异常
        context: 额外的上下文信息
    """
    context = context or {}
    
    # 构造错误消息
    error_msg = f"Exception occurred: {type(exception).__name__}: {str(exception)}"
    
    if context:
        context_str = ", ".join([f"{k}={v}" for k, v in context.items()])
        error_msg += f" (Context: {context_str})"
    
    # 记录日志
    if logger:
        if isinstance(exception, MCMError):
            logger.error(error_msg, extra={"exception_details": exception.details})
        else:
            logger.error(error_msg, exc_info=True)
    
    # 重新抛出异常
    if reraise:
        raise exception