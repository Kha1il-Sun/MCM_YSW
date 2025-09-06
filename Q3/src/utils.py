"""
工具函数模块
包含随机种子设置、日志配置、配置读取等通用功能
"""

import os
import yaml
import logging
import numpy as np
import random
from datetime import datetime
from pathlib import Path


def set_random_seed(seed=42):
    """设置随机种子确保结果可复现"""
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def setup_logging(log_level=logging.INFO, log_file=None):
    """设置日志配置"""
    if log_file is None:
        log_file = f"outputs/logs/run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    # 确保日志目录存在
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def load_config(config_path="config.yaml"):
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def ensure_dir(path):
    """确保目录存在"""
    Path(path).mkdir(parents=True, exist_ok=True)


def winsorize(data, limits=(0.05, 0.05)):
    """Winsorize数据，处理异常值"""
    from scipy.stats import mstats
    return mstats.winsorize(data, limits=limits)


def safe_logit(x, eps=1e-15):
    """安全的logit变换，避免数值问题"""
    x = np.clip(x, eps, 1 - eps)
    return np.log(x / (1 - x))


def safe_expit(x):
    """安全的expit变换（sigmoid的逆）"""
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))


def calculate_bmi(weight, height):
    """计算BMI"""
    return weight / (height / 100) ** 2


def format_results(results, decimal_places=4):
    """格式化结果输出"""
    if isinstance(results, dict):
        return {k: round(v, decimal_places) if isinstance(v, (int, float)) else v 
                for k, v in results.items()}
    elif isinstance(results, (list, tuple)):
        return [round(x, decimal_places) if isinstance(x, (int, float)) else x 
                for x in results]
    else:
        return round(results, decimal_places) if isinstance(results, (int, float)) else results


class Timer:
    """简单的计时器类"""
    def __init__(self):
        self.start_time = None
        self.end_time = None
    
    def start(self):
        self.start_time = datetime.now()
    
    def stop(self):
        self.end_time = datetime.now()
        return self.elapsed()
    
    def elapsed(self):
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return None
