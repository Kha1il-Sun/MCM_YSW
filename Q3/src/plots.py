"""
可视化模块

提供生存分析、风险曲线、分组结果等各类图表的绘制功能。
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging

from .exceptions import ComputationError
from .utils import ensure_numpy

logger = logging.getLogger(__name__)

# 设置绘图风格
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


class PlotManager:
    """绘图管理器"""
    
    def __init__(self, output_dir: Union[str, Path] = "outputs/figures",
                 dpi: int = 300, figsize: Tuple[float, float] = (10, 8),
                 save_formats: List[str] = ["png", "pdf"]):
        """
        初始化绘图管理器
        
        Args:
            output_dir: 输出目录
            dpi: 图像分辨率
            figsize: 图像大小
            save_formats: 保存格式列表
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.dpi = dpi
        self.figsize = figsize
        self.save_formats = save_formats
        
        logger.info(f"初始化绘图管理器: 输出到 {self.output_dir}")
    
    def save_figure(self, fig, filename: str, **kwargs) -> None:
        """保存图形到多种格式"""
        for fmt in self.save_formats:
            filepath = self.output_dir / f"{filename}.{fmt}"
            fig.savefig(filepath, dpi=self.dpi, bbox_inches='tight', **kwargs)
        
        logger.debug(f"保存图形: {filename}")
        plt.close(fig)
    
    def plot_survival_curves(self, t_grid: np.ndarray, 
                           F_curves: np.ndarray,
                           labels: Optional[List[str]] = None,
                           title: str = "生存曲线",
                           filename: str = "survival_curves") -> None:
        """
        绘制生存曲线
        
        Args:
            t_grid: 时间网格
            F_curves: 累积分布函数值 (n_times, n_curves)
            labels: 曲线标签
            title: 图标题
            filename: 保存文件名
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        n_curves = F_curves.shape[1]
        
        if labels is None:
            labels = [f"曲线 {i+1}" for i in range(n_curves)]
        
        # 如果曲线太多，只显示部分
        if n_curves > 20:
            indices = np.linspace(0, n_curves-1, 20, dtype=int)
            F_curves = F_curves[:, indices]
            labels = [labels[i] for i in indices]
            n_curves = 20
        
        colors = plt.cm.tab10(np.linspace(0, 1, min(n_curves, 10)))
        
        for i in range(n_curves):
            color = colors[i % 10]
            ax.plot(t_grid, F_curves[:, i], 
                   label=labels[i], color=color, linewidth=2)
        
        ax.set_xlabel("时间 (周)")
        ax.set_ylabel("累积达标概率 F(t)")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, np.max(t_grid))
        ax.set_ylim(0, 1)
        
        if n_curves <= 10:
            ax.legend()
        
        self.save_figure(fig, filename)
    
    def plot_group_survival_curves(self, group_curves: Dict[Any, Tuple[np.ndarray, np.ndarray]],
                                 title: str = "分组生存曲线",
                                 filename: str = "group_survival_curves") -> None:
        """
        绘制分组生存曲线
        
        Args:
            group_curves: 分组曲线字典 {group_id: (t_grid, mean_curve)}
            title: 图标题
            filename: 保存文件名
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        colors = plt.cm.Set1(np.linspace(0, 1, len(group_curves)))
        
        for i, (group_id, (t_grid, mean_curve)) in enumerate(group_curves.items()):
            ax.plot(t_grid, mean_curve, 
                   label=f"组 {group_id}", 
                   color=colors[i], 
                   linewidth=3)
        
        ax.set_xlabel("时间 (周)")
        ax.set_ylabel("平均累积达标概率 F̄(t)")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_xlim(0, np.max([np.max(t) for t, _ in group_curves.values()]))
        ax.set_ylim(0, 1)
        
        self.save_figure(fig, filename)
    
    def plot_risk_curves(self, risk_df: pd.DataFrame,
                        title: str = "风险曲线",
                        filename: str = "risk_curves") -> None:
        """
        绘制风险曲线
        
        Args:
            risk_df: 包含w和风险组件的DataFrame
            title: 图标题
            filename: 保存文件名
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(self.figsize[0], self.figsize[1]*1.2))
        
        # 上图：总风险和各组件
        ax1.plot(risk_df['w'], risk_df['total_risk'], 
                label='总风险', linewidth=3, color='red')
        ax1.plot(risk_df['w'], risk_df['failure_risk'], 
                label='未达标风险', linewidth=2, alpha=0.8)
        ax1.plot(risk_df['w'], risk_df['delay_penalty'], 
                label='延迟惩罚', linewidth=2, alpha=0.8)
        ax1.plot(risk_df['w'], risk_df['uncertainty_penalty'], 
                label='不确定性惩罚', linewidth=2, alpha=0.8)
        
        ax1.set_xlabel("推荐时间点 w (周)")
        ax1.set_ylabel("风险值")
        ax1.set_title(f"{title} - 风险组件")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 下图：达标概率
        ax2.plot(risk_df['w'], risk_df['mean_success_prob'], 
                linewidth=3, color='green')
        ax2.axhline(y=0.9, color='red', linestyle='--', alpha=0.7, label='门槛 τ=0.9')
        
        ax2.set_xlabel("推荐时间点 w (周)")
        ax2.set_ylabel("平均达标概率")
        ax2.set_title(f"{title} - 达标概率")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 1)
        
        plt.tight_layout()
        self.save_figure(fig, filename)
    
    def plot_bmi_segmentation(self, segmentation_info: Dict[str, Any],
                            BMI_data: Optional[np.ndarray] = None,
                            title: str = "BMI分段结果",
                            filename: str = "bmi_segmentation") -> None:
        """
        绘制BMI分段结果
        
        Args:
            segmentation_info: 分段信息
            BMI_data: BMI数据（用于显示分布）
            title: 图标题
            filename: 保存文件名
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(self.figsize[0], self.figsize[1]*1.2))
        
        breakpoints = segmentation_info['breakpoints']
        segments = segmentation_info['segments']
        
        # 上图：BMI分布和分段
        if BMI_data is not None:
            ax1.hist(BMI_data, bins=50, alpha=0.6, density=True, 
                    color='lightblue', label='BMI分布')
        
        # 绘制分段线
        for bp in breakpoints:
            ax1.axvline(x=bp, color='red', linestyle='--', alpha=0.8)
        
        # 标注分段区间
        for i, segment in enumerate(segments):
            bmi_range = segment['bmi_range']
            mid_point = (bmi_range[0] + bmi_range[1]) / 2
            ax1.text(mid_point, ax1.get_ylim()[1] * 0.8, 
                    f"段{i}\n[{bmi_range[0]:.1f}, {bmi_range[1]:.1f}]",
                    ha='center', va='center', 
                    bbox=dict(boxstyle="round,pad=0.3", facecolor='yellow', alpha=0.7))
        
        ax1.set_xlabel("BMI")
        ax1.set_ylabel("密度")
        ax1.set_title(f"{title} - 分段划分")
        ax1.grid(True, alpha=0.3)
        if BMI_data is not None:
            ax1.legend()
        
        # 下图：各段的最优w*
        segment_ids = []
        optimal_ws = []
        bmi_ranges = []
        
        for segment in segments:
            if segment['optimal_w'] is not None:
                segment_ids.append(segment['segment_id'])
                optimal_ws.append(segment['optimal_w'])
                bmi_range = segment['bmi_range']
                bmi_ranges.append(f"[{bmi_range[0]:.1f}, {bmi_range[1]:.1f}]")
        
        if optimal_ws:
            bars = ax2.bar(range(len(segment_ids)), optimal_ws, 
                          color=plt.cm.Set3(np.linspace(0, 1, len(segment_ids))))
            
            # 添加数值标签
            for bar, w in zip(bars, optimal_ws):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                        f'{w:.2f}', ha='center', va='bottom')
            
            ax2.set_xlabel("分段")
            ax2.set_ylabel("最优推荐时间 w* (周)")
            ax2.set_title(f"{title} - 各段最优w*")
            ax2.set_xticks(range(len(segment_ids)))
            ax2.set_xticklabels([f"段{i}\n{r}" for i, r in zip(segment_ids, bmi_ranges)])
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        self.save_figure(fig, filename)
    
    def plot_calibration(self, calibration_result: Dict[str, Any],
                        title: str = "预测校准图",
                        filename: str = "calibration_plot") -> None:
        """
        绘制预测校准图
        
        Args:
            calibration_result: 校准结果
            title: 图标题
            filename: 保存文件名
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(self.figsize[0]*1.5, self.figsize[1]))
        
        bin_centers = calibration_result['bin_centers']
        observed_freqs = calibration_result['observed_frequencies']
        predicted_freqs = calibration_result['predicted_frequencies']
        bin_counts = calibration_result['bin_counts']
        
        # 左图：校准曲线
        ax1.plot([0, 1], [0, 1], 'k--', alpha=0.7, label='完美校准')
        ax1.scatter(predicted_freqs, observed_freqs, 
                   s=bin_counts*10, alpha=0.7, label='观测点')
        
        ax1.set_xlabel("预测概率")
        ax1.set_ylabel("观测频率")
        ax1.set_title("校准曲线")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(0, 1)
        ax1.set_ylim(0, 1)
        
        # 右图：各bin的样本数
        valid_bins = np.array(bin_counts) > 0
        if np.any(valid_bins):
            ax2.bar(np.array(bin_centers)[valid_bins], 
                   np.array(bin_counts)[valid_bins],
                   width=0.08, alpha=0.7)
        
        ax2.set_xlabel("预测概率区间")
        ax2.set_ylabel("样本数")
        ax2.set_title("样本分布")
        ax2.grid(True, alpha=0.3)
        
        # 添加指标文本
        brier_score = calibration_result['brier_score']
        ece = calibration_result['expected_calibration_error']
        
        fig.suptitle(f"{title}\nBrier分数: {brier_score:.4f}, ECE: {ece:.4f}")
        
        plt.tight_layout()
        self.save_figure(fig, filename)
    
    def plot_sensitivity_analysis(self, sensitivity_results: Dict[str, pd.DataFrame],
                                title: str = "敏感性分析",
                                filename: str = "sensitivity_analysis") -> None:
        """
        绘制敏感性分析结果
        
        Args:
            sensitivity_results: 敏感性分析结果
            title: 图标题
            filename: 保存文件名
        """
        n_params = len(sensitivity_results)
        
        if n_params == 0:
            return
        
        fig, axes = plt.subplots(2, n_params, 
                               figsize=(self.figsize[0]*n_params, self.figsize[1]*1.5))
        
        if n_params == 1:
            axes = axes.reshape(2, 1)
        
        for i, (param_name, results_df) in enumerate(sensitivity_results.items()):
            if param_name not in results_df.columns:
                continue
            
            param_values = results_df[param_name].values
            optimal_ws = results_df['optimal_w'].values
            w_shifts = results_df['w_shift'].values
            
            # 上图：最优w*随参数变化
            axes[0, i].plot(param_values, optimal_ws, 'o-', linewidth=2, markersize=6)
            axes[0, i].set_xlabel(param_name)
            axes[0, i].set_ylabel("最优w*")
            axes[0, i].set_title(f"{param_name} 对 w* 的影响")
            axes[0, i].grid(True, alpha=0.3)
            
            # 下图：w*偏移量
            axes[1, i].plot(param_values, w_shifts, 's-', 
                           linewidth=2, markersize=6, color='orange')
            axes[1, i].axhline(y=0, color='red', linestyle='--', alpha=0.7)
            axes[1, i].set_xlabel(param_name)
            axes[1, i].set_ylabel("w* 偏移量")
            axes[1, i].set_title(f"{param_name} 对 w* 偏移的影响")
            axes[1, i].grid(True, alpha=0.3)
        
        plt.suptitle(title)
        plt.tight_layout()
        self.save_figure(fig, filename)
    
    def plot_model_comparison(self, comparison_df: pd.DataFrame,
                            title: str = "模型比较",
                            filename: str = "model_comparison") -> None:
        """
        绘制模型比较结果
        
        Args:
            comparison_df: 比较结果DataFrame
            title: 图标题
            filename: 保存文件名
        """
        fig, axes = plt.subplots(2, 2, figsize=(self.figsize[0]*1.5, self.figsize[1]*1.5))
        
        models = comparison_df.index if hasattr(comparison_df, 'index') else range(len(comparison_df))
        
        # AIC比较
        if 'AIC' in comparison_df.columns:
            axes[0, 0].bar(models, comparison_df['AIC'], alpha=0.7)
            axes[0, 0].set_title("AIC 比较")
            axes[0, 0].set_ylabel("AIC")
            axes[0, 0].tick_params(axis='x', rotation=45)
        
        # BIC比较
        if 'BIC' in comparison_df.columns:
            axes[0, 1].bar(models, comparison_df['BIC'], alpha=0.7, color='orange')
            axes[0, 1].set_title("BIC 比较")
            axes[0, 1].set_ylabel("BIC")
            axes[0, 1].tick_params(axis='x', rotation=45)
        
        # 对数似然比较
        if 'loglik' in comparison_df.columns:
            axes[1, 0].bar(models, comparison_df['loglik'], alpha=0.7, color='green')
            axes[1, 0].set_title("对数似然比较")
            axes[1, 0].set_ylabel("Log-likelihood")
            axes[1, 0].tick_params(axis='x', rotation=45)
        
        # 其他指标
        if 'optimal_w' in comparison_df.columns:
            axes[1, 1].bar(models, comparison_df['optimal_w'], alpha=0.7, color='purple')
            axes[1, 1].set_title("最优w*比较")
            axes[1, 1].set_ylabel("w*")
            axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.suptitle(title)
        plt.tight_layout()
        self.save_figure(fig, filename)
    
    def plot_feature_importance(self, importance_dict: Dict[str, Dict[str, float]],
                              title: str = "特征重要性",
                              filename: str = "feature_importance") -> None:
        """
        绘制特征重要性
        
        Args:
            importance_dict: 特征重要性字典
            title: 图标题
            filename: 保存文件名
        """
        n_models = len(importance_dict)
        
        if n_models == 0:
            return
        
        fig, axes = plt.subplots(1, n_models, figsize=(self.figsize[0]*n_models, self.figsize[1]))
        
        if n_models == 1:
            axes = [axes]
        
        for i, (model_name, importance) in enumerate(importance_dict.items()):
            features = list(importance.keys())
            values = list(importance.values())
            
            # 按重要性排序
            sorted_indices = np.argsort(values)[::-1]
            features = [features[j] for j in sorted_indices]
            values = [values[j] for j in sorted_indices]
            
            axes[i].barh(range(len(features)), values, alpha=0.7)
            axes[i].set_yticks(range(len(features)))
            axes[i].set_yticklabels(features)
            axes[i].set_xlabel("重要性")
            axes[i].set_title(f"{model_name} 特征重要性")
            axes[i].grid(True, alpha=0.3)
        
        plt.suptitle(title)
        plt.tight_layout()
        self.save_figure(fig, filename)


def create_plot_manager(config: Dict[str, Any]) -> PlotManager:
    """
    创建绘图管理器的工厂函数
    
    Args:
        config: 绘图配置
        
    Returns:
        绘图管理器实例
    """
    plots_config = config.get('plots', {})
    
    return PlotManager(
        output_dir=plots_config.get('output_dir', 'outputs/figures'),
        dpi=plots_config.get('dpi', 300),
        figsize=tuple(plots_config.get('figsize', [10, 8])),
        save_formats=plots_config.get('save_formats', ['png', 'pdf'])
    )


def plot_summary_dashboard(plot_manager: PlotManager,
                         results: Dict[str, Any],
                         title: str = "MCM Q3 分析结果总览") -> None:
    """
    创建结果总览仪表板
    
    Args:
        plot_manager: 绘图管理器
        results: 分析结果字典
        title: 仪表板标题
    """
    logger.info("创建结果总览仪表板")
    
    # 这里可以根据results的内容创建综合性的仪表板
    # 由于结果结构复杂，这里提供一个框架
    
    fig = plt.figure(figsize=(20, 15))
    
    # 创建子图网格
    gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
    
    # 可以根据具体的results内容添加各种图表
    # 例如：
    # - 生存曲线
    # - 风险曲线  
    # - 分组结果
    # - 模型比较
    # - 敏感性分析
    
    plt.suptitle(title, fontsize=16)
    
    plot_manager.save_figure(fig, "summary_dashboard")