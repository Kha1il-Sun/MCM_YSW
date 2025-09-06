"""
Q3 NIPT BMI分组与最佳时点优化项目

本包实现了基于多因素的NIPT最佳检测时点优化算法，
采用双通道交叉验证方法和BMI分组策略。
"""

__version__ = "1.0.0"
__author__ = "Q3项目组"
__email__ = "q3-project@example.com"

# 导入核心模块
from . import dataio
from . import utils  
from . import optimize
from . import grouping
from . import sensitivity
from . import plots

__all__ = [
    "dataio",
    "utils", 
    "optimize",
    "grouping", 
    "sensitivity",
    "plots"
]