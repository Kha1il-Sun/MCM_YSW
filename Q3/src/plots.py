"""
绘图模块
生成各种分析图表
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import logging

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

logger = logging.getLogger(__name__)


class PlotGenerator:
    """绘图生成器"""
    
    def __init__(self, config):
        self.config = config
        self.dpi = config['output']['figure_dpi']
        self.format = config['output']['figure_format']
        
    def plot_eda_analysis(self, data, save_path=None):
        """绘制探索性数据分析图"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 1. Y染色体浓度分布
        axes[0, 0].hist(data['Y染色体浓度'], bins=30, alpha=0.7, edgecolor='black')
        axes[0, 0].axvline(x=0.04, color='red', linestyle='--', label='4%阈值')
        axes[0, 0].set_xlabel('Y染色体浓度')
        axes[0, 0].set_ylabel('频数')
        axes[0, 0].set_title('Y染色体浓度分布')
        axes[0, 0].legend()
        
        # 2. BMI分布
        axes[0, 1].hist(data['BMI'], bins=30, alpha=0.7, edgecolor='black')
        axes[0, 1].set_xlabel('BMI')
        axes[0, 1].set_ylabel('频数')
        axes[0, 1].set_title('BMI分布')
        
        # 3. 孕周分布
        axes[0, 2].hist(data['孕周'], bins=20, alpha=0.7, edgecolor='black')
        axes[0, 2].set_xlabel('孕周')
        axes[0, 2].set_ylabel('频数')
        axes[0, 2].set_title('孕周分布')
        
        # 4. Y染色体浓度 vs 孕周
        scatter = axes[1, 0].scatter(data['孕周'], data['Y染色体浓度'], 
                                   c=data['BMI'], cmap='viridis', alpha=0.6)
        axes[1, 0].axhline(y=0.04, color='red', linestyle='--', label='4%阈值')
        axes[1, 0].set_xlabel('孕周')
        axes[1, 0].set_ylabel('Y染色体浓度')
        axes[1, 0].set_title('Y染色体浓度 vs 孕周（颜色表示BMI）')
        axes[1, 0].legend()
        plt.colorbar(scatter, ax=axes[1, 0], label='BMI')
        
        # 5. 成功率 vs 孕周
        success_by_week = data.groupby('孕周')['是否成功'].mean()
        axes[1, 1].plot(success_by_week.index, success_by_week.values, 'o-')
        axes[1, 1].set_xlabel('孕周')
        axes[1, 1].set_ylabel('成功率')
        axes[1, 1].set_title('成功率 vs 孕周')
        axes[1, 1].grid(True, alpha=0.3)
        
        # 6. 达标率 vs 孕周
        attain_by_week = data.groupby('孕周')['达标'].mean()
        axes[1, 2].plot(attain_by_week.index, attain_by_week.values, 'o-', color='green')
        axes[1, 2].axhline(y=0.9, color='red', linestyle='--', label='90%达标率')
        axes[1, 2].set_xlabel('孕周')
        axes[1, 2].set_ylabel('达标率')
        axes[1, 2].set_title('达标率 vs 孕周')
        axes[1, 2].legend()
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, format=self.format, bbox_inches='tight')
            logger.info(f"EDA分析图已保存: {save_path}")
        
        plt.show()
        return fig
    
    def plot_curves_by_bmi(self, data, group_stats, save_path=None):
        """绘制按BMI分组的孕周-概率曲线"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 按BMI分组
        bmi_groups = pd.cut(data['BMI'], 
                           bins=[0, 18.5, 24, 28, np.inf], 
                           labels=['偏瘦', '正常', '超重', '肥胖'],
                           right=False)
        
        colors = ['blue', 'green', 'orange', 'red']
        
        # 1. 达标概率曲线
        for i, group in enumerate(['偏瘦', '正常', '超重', '肥胖']):
            group_data = data[bmi_groups == group]
            if len(group_data) > 0:
                attain_by_week = group_data.groupby('孕周')['达标'].mean()
                axes[0, 0].plot(attain_by_week.index, attain_by_week.values, 
                               'o-', color=colors[i], label=group, linewidth=2)
        
        axes[0, 0].axhline(y=0.9, color='red', linestyle='--', label='90%达标率')
        axes[0, 0].set_xlabel('孕周')
        axes[0, 0].set_ylabel('达标概率')
        axes[0, 0].set_title('各BMI组达标概率曲线')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. 成功率曲线
        for i, group in enumerate(['偏瘦', '正常', '超重', '肥胖']):
            group_data = data[bmi_groups == group]
            if len(group_data) > 0:
                success_by_week = group_data.groupby('孕周')['是否成功'].mean()
                axes[0, 1].plot(success_by_week.index, success_by_week.values, 
                               'o-', color=colors[i], label=group, linewidth=2)
        
        axes[0, 1].set_xlabel('孕周')
        axes[0, 1].set_ylabel('成功率')
        axes[0, 1].set_title('各BMI组成功率曲线')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. 一次命中概率曲线
        for i, group in enumerate(['偏瘦', '正常', '超重', '肥胖']):
            group_data = data[bmi_groups == group]
            if len(group_data) > 0:
                detect_by_week = group_data.groupby('孕周').apply(
                    lambda x: (x['达标'] * x['是否成功']).mean()
                )
                axes[1, 0].plot(detect_by_week.index, detect_by_week.values, 
                               'o-', color=colors[i], label=group, linewidth=2)
        
        axes[1, 0].set_xlabel('孕周')
        axes[1, 0].set_ylabel('一次命中概率')
        axes[1, 0].set_title('各BMI组一次命中概率曲线')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. 推荐时点对比
        if group_stats is not None:
            groups = group_stats['group'].values
            recommended_weeks = group_stats['recommended_week'].values
            
            bars = axes[1, 1].bar(groups, recommended_weeks, color=colors[:len(groups)])
            axes[1, 1].set_xlabel('BMI组')
            axes[1, 1].set_ylabel('推荐检测时点（周）')
            axes[1, 1].set_title('各BMI组推荐检测时点')
            
            # 添加数值标签
            for bar, week in zip(bars, recommended_weeks):
                axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                               f'{week:.1f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, format=self.format, bbox_inches='tight')
            logger.info(f"BMI分组曲线图已保存: {save_path}")
        
        plt.show()
        return fig
    
    def plot_risk_curves(self, optimization_results, group_stats, save_path=None):
        """绘制风险函数曲线与最优点"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. 个体最优时点分布
        optimal_weeks = [r['optimal_week'] for r in optimization_results 
                        if r['optimal_week'] is not None]
        
        axes[0, 0].hist(optimal_weeks, bins=20, alpha=0.7, edgecolor='black')
        axes[0, 0].set_xlabel('最优检测时点（周）')
        axes[0, 0].set_ylabel('频数')
        axes[0, 0].set_title('个体最优检测时点分布')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. 风险函数示例
        weeks = np.linspace(10, 25, 100)
        time_risks = np.where(weeks <= 12, 0.1, 
                             np.where(weeks <= 27, 1.0, 2.0))
        
        axes[0, 1].plot(weeks, time_risks, 'b-', linewidth=2, label='时间段风险')
        axes[0, 1].set_xlabel('孕周')
        axes[0, 1].set_ylabel('风险值')
        axes[0, 1].set_title('时间段风险函数')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. 检测概率 vs 风险
        detect_probs = [r['optimal_detect_prob'] for r in optimization_results 
                       if r['optimal_detect_prob'] is not None]
        risks = [r['optimal_risk'] for r in optimization_results 
                if r['optimal_risk'] is not None]
        
        axes[1, 0].scatter(detect_probs, risks, alpha=0.6)
        axes[1, 0].set_xlabel('检测概率')
        axes[1, 0].set_ylabel('风险值')
        axes[1, 0].set_title('检测概率 vs 风险')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. 各组推荐时点对比
        if group_stats is not None:
            groups = group_stats['group'].values
            recommended_weeks = group_stats['recommended_week'].values
            n_samples = group_stats['n_samples'].values
            
            # 使用气泡图显示样本数
            scatter = axes[1, 1].scatter(groups, recommended_weeks, s=n_samples*2, 
                                       alpha=0.6, c=range(len(groups)), cmap='viridis')
            axes[1, 1].set_xlabel('BMI组')
            axes[1, 1].set_ylabel('推荐检测时点（周）')
            axes[1, 1].set_title('各组推荐时点（气泡大小表示样本数）')
            
            # 添加数值标签
            for i, (group, week, n) in enumerate(zip(groups, recommended_weeks, n_samples)):
                axes[1, 1].annotate(f'{week:.1f}\n(n={n})', 
                                   (group, week), ha='center', va='center')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, format=self.format, bbox_inches='tight')
            logger.info(f"风险曲线图已保存: {save_path}")
        
        plt.show()
        return fig
    
    def plot_sensitivity_analysis(self, sensitivity_results, save_path=None):
        """绘制敏感性分析图"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. 阈值敏感性
        if 'threshold' in sensitivity_results:
            threshold_df = sensitivity_results['threshold']
            axes[0, 0].plot(threshold_df['threshold'], threshold_df['avg_detect_prob'], 
                           'o-', linewidth=2, markersize=8)
            axes[0, 0].set_xlabel('Y染色体浓度阈值')
            axes[0, 0].set_ylabel('平均检测概率')
            axes[0, 0].set_title('阈值敏感性分析')
            axes[0, 0].grid(True, alpha=0.3)
        
        # 2. 误差敏感性
        if 'error' in sensitivity_results:
            error_df = sensitivity_results['error']
            axes[0, 1].plot(error_df['error_factor'], error_df['avg_detect_prob'], 
                           'o-', linewidth=2, markersize=8)
            axes[0, 1].set_xlabel('误差因子')
            axes[0, 1].set_ylabel('平均检测概率')
            axes[0, 1].set_title('测量误差敏感性分析')
            axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Bootstrap置信区间
        if 'confidence_intervals' in sensitivity_results:
            ci = sensitivity_results['confidence_intervals']
            metrics = list(ci.keys())
            means = [ci[metric]['mean'] for metric in metrics]
            ci_lowers = [ci[metric]['ci_lower'] for metric in metrics]
            ci_uppers = [ci[metric]['ci_upper'] for metric in metrics]
            
            x_pos = np.arange(len(metrics))
            axes[1, 0].bar(x_pos, means, yerr=[np.array(means) - np.array(ci_lowers),
                                              np.array(ci_uppers) - np.array(means)],
                          capsize=5, alpha=0.7)
            axes[1, 0].set_xticks(x_pos)
            axes[1, 0].set_xticklabels(metrics, rotation=45)
            axes[1, 0].set_ylabel('概率值')
            axes[1, 0].set_title('Bootstrap置信区间')
            axes[1, 0].grid(True, alpha=0.3)
        
        # 4. 权重敏感性
        if 'weight' in sensitivity_results:
            weight_df = sensitivity_results['weight']
            axes[1, 1].plot(weight_df['gamma'], weight_df['avg_risk'], 
                           'o-', linewidth=2, markersize=8)
            axes[1, 1].set_xlabel('Gamma权重')
            axes[1, 1].set_ylabel('平均风险')
            axes[1, 1].set_title('权重敏感性分析')
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, format=self.format, bbox_inches='tight')
            logger.info(f"敏感性分析图已保存: {save_path}")
        
        plt.show()
        return fig
    
    def plot_redraw_strategy(self, group_strategies, save_path=None):
        """绘制重采策略分析图"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 提取数据
        groups = list(group_strategies.keys())
        w1_values = [group_strategies[g]['strategy']['w1'] for g in groups]
        delta_values = [group_strategies[g]['strategy']['delta'] for g in groups]
        w2_values = [group_strategies[g]['strategy']['w2'] for g in groups]
        expected_risks = [group_strategies[g]['strategy']['expected_risk'] for g in groups]
        n_samples = [group_strategies[g]['n_samples'] for g in groups]
        
        # 1. 第一次检测时点
        bars1 = axes[0, 0].bar(groups, w1_values, alpha=0.7, color='skyblue')
        axes[0, 0].set_xlabel('BMI组')
        axes[0, 0].set_ylabel('第一次检测时点（周）')
        axes[0, 0].set_title('各BMI组第一次检测时点')
        
        # 添加数值标签
        for bar, w1 in zip(bars1, w1_values):
            axes[0, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                           f'{w1:.1f}', ha='center', va='bottom')
        
        # 2. 重采间隔
        bars2 = axes[0, 1].bar(groups, delta_values, alpha=0.7, color='lightgreen')
        axes[0, 1].set_xlabel('BMI组')
        axes[0, 1].set_ylabel('重采间隔（周）')
        axes[0, 1].set_title('各BMI组重采间隔')
        
        # 添加数值标签
        for bar, delta in zip(bars2, delta_values):
            axes[0, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                           f'{delta:.1f}', ha='center', va='bottom')
        
        # 3. 期望风险
        bars3 = axes[1, 0].bar(groups, expected_risks, alpha=0.7, color='salmon')
        axes[1, 0].set_xlabel('BMI组')
        axes[1, 0].set_ylabel('期望风险')
        axes[1, 0].set_title('各BMI组期望风险')
        
        # 添加数值标签
        for bar, risk in zip(bars3, expected_risks):
            axes[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                           f'{risk:.3f}', ha='center', va='bottom')
        
        # 4. 策略对比（气泡图）
        scatter = axes[1, 1].scatter(w1_values, w2_values, s=np.array(n_samples)*3, 
                                   c=expected_risks, cmap='viridis', alpha=0.7)
        axes[1, 1].set_xlabel('第一次检测时点（周）')
        axes[1, 1].set_ylabel('第二次检测时点（周）')
        axes[1, 1].set_title('重采策略对比（颜色表示期望风险）')
        
        # 添加组标签
        for i, group in enumerate(groups):
            axes[1, 1].annotate(group, (w1_values[i], w2_values[i]), 
                               ha='center', va='center', fontsize=8)
        
        plt.colorbar(scatter, ax=axes[1, 1], label='期望风险')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, format=self.format, bbox_inches='tight')
            logger.info(f"重采策略图已保存: {save_path}")
        
        plt.show()
        return fig
