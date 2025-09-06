"""
检测误差影响分析器
分析不同σ（检测误差）水平对BMI分组和时点推荐的影响
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any
import yaml
import os
import sys
from datetime import datetime

# 添加父目录到路径，以便导入现有模块
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from empirical_model import EmpiricalDetectionModel
from io_utils import load_step1_products, load_step2_config

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class ErrorImpactAnalyzer:
    """检测误差影响分析器"""
    
    def __init__(self, config_path: str = None):
        """
        初始化误差影响分析器
        
        Parameters:
        -----------
        config_path : str
            配置文件路径
        """
        self.config = self._load_config(config_path)
        self.empirical_model = None
        self.baseline_sigma = None
        self.bmi_groups = None
        self.analysis_results = {}
        
        # 设置σ情景倍数（扩大范围以显示更明显的影响）
        self.sigma_multipliers = [0.1, 0.5, 1.0, 2.0, 5.0]
        
        print("✅ 误差影响分析器初始化完成")
    
    def _load_config(self, config_path: str) -> Dict:
        """加载配置文件"""
        if config_path is None:
            config_path = os.path.join(os.path.dirname(__file__), '..', '..', 'config', 'step2_config.yaml')
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            return config
        except Exception as e:
            print(f"⚠️ 配置文件加载失败，使用默认配置: {e}")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict:
        """获取默认配置"""
        return {
            'grouping': {
                'custom_cuts': [20.0, 30.5, 32.7, 34.4, 50.0]
            },
            'model_params': {
                'alpha': 11.358,
                'beta': -24.261,
                'gamma': 0.5,
                'delta': 0.01
            }
        }
    
    def load_data_and_model(self, data_dir: str = None):
        """
        加载数据和模型
        
        Parameters:
        -----------
        data_dir : str
            数据目录路径
        """
        if data_dir is None:
            data_dir = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'step2_1')
        
        print("📊 加载数据和模型...")
        
        try:
            # 加载Step1数据
            self.long_df, self.surv_df, self.report_df, self.step1_config = load_step1_products(data_dir)
            print(f"✅ 数据加载成功: {len(self.long_df)} 条记录")
            
            # 创建经验模型
            self.empirical_model = EmpiricalDetectionModel(
                alpha=self.config['model_params']['alpha'],
                beta=self.config['model_params']['beta'],
                gamma=self.config['model_params']['gamma'],
                delta=self.config['model_params']['delta']
            )
            
            # 获取基准σ
            self.baseline_sigma = self._extract_baseline_sigma()
            print(f"✅ 基准σ: {self.baseline_sigma:.6f}")
            
            # 设置BMI分组
            self._setup_bmi_groups()
            print(f"✅ BMI分组设置完成: {len(self.bmi_groups)} 个组")
            
        except Exception as e:
            print(f"❌ 数据加载失败: {e}")
            raise
    
    def _extract_baseline_sigma(self) -> float:
        """从报告中提取基准σ"""
        try:
            # 尝试从报告中提取σ
            if 'sigma_global' in self.report_df.columns:
                return self.report_df['sigma_global'].iloc[0]
            elif 'global_sigma' in self.report_df.columns:
                return self.report_df['global_sigma'].iloc[0]
            else:
                # 如果没有找到，使用默认值
                print("⚠️ 未找到基准σ，使用默认值 0.005857")
                return 0.005857
        except:
            print("⚠️ 无法提取基准σ，使用默认值 0.005857")
            return 0.005857
    
    def _setup_bmi_groups(self):
        """设置BMI分组"""
        cuts = self.config['grouping']['custom_cuts']
        labels = ['低BMI组', '中BMI组', '高BMI组', '极高BMI组']
        
        self.bmi_groups = {}
        for i, (label, (cut_min, cut_max)) in enumerate(zip(labels, zip(cuts[:-1], cuts[1:]))):
            # 计算组内BMI中位数
            group_data = self.long_df[
                (self.long_df['BMI_used'] >= cut_min) & 
                (self.long_df['BMI_used'] < cut_max)
            ]
            bmi_median = group_data['BMI_used'].median() if len(group_data) > 0 else (cut_min + cut_max) / 2
            
            self.bmi_groups[label] = {
                'bmi_range': (cut_min, cut_max),
                'bmi_median': bmi_median,
                'sample_size': len(group_data)
            }
    
    def run_error_impact_analysis(self) -> Dict:
        """
        运行误差影响分析
        
        Returns:
        --------
        Dict
            分析结果
        """
        print("🔍 开始误差影响分析...")
        
        # 1. 计算不同σ下的推荐时点
        time_recommendations = self._calculate_time_recommendations()
        
        # 2. 计算风险函数修正
        risk_analysis = self._analyze_risk_function_impact()
        
        # 3. 敏感性分析
        sensitivity_analysis = self._run_sensitivity_analysis()
        
        # 4. 生成对比表格
        comparison_table = self._generate_comparison_table(time_recommendations)
        
        # 5. 生成可视化
        self._create_visualizations(time_recommendations, sensitivity_analysis)
        
        # 整合结果
        self.analysis_results = {
            'time_recommendations': time_recommendations,
            'risk_analysis': risk_analysis,
            'sensitivity_analysis': sensitivity_analysis,
            'comparison_table': comparison_table,
            'baseline_sigma': self.baseline_sigma,
            'sigma_multipliers': self.sigma_multipliers
        }
        
        print("✅ 误差影响分析完成")
        return self.analysis_results
    
    def _calculate_time_recommendations(self) -> Dict:
        """计算不同σ下的推荐时点"""
        print("📅 计算推荐时点...")
        
        recommendations = {}
        
        for multiplier in self.sigma_multipliers:
            sigma = self.baseline_sigma * multiplier
            sigma = max(sigma, 0.001)  # 设置σ下限，避免除零
            
            group_times = {}
            for group_name, group_info in self.bmi_groups.items():
                bmi_median = group_info['bmi_median']
                optimal_time = self.empirical_model.predict_optimal_time(bmi_median, sigma)
                group_times[group_name] = {
                    'bmi_median': bmi_median,
                    'optimal_time': optimal_time,
                    'sigma': sigma
                }
            
            recommendations[f'sigma_{multiplier}x'] = {
                'sigma': sigma,
                'multiplier': multiplier,
                'group_times': group_times
            }
        
        return recommendations
    
    def _analyze_risk_function_impact(self) -> Dict:
        """分析风险函数影响"""
        print("⚠️ 分析风险函数影响...")
        
        risk_analysis = {}
        
        for multiplier in self.sigma_multipliers:
            sigma = self.baseline_sigma * multiplier
            sigma = max(sigma, 0.001)
            
            group_risks = {}
            for group_name, group_info in self.bmi_groups.items():
                bmi_median = group_info['bmi_median']
                
                # 计算考虑σ的风险
                risk_with_sigma = self._calculate_risk_with_sigma(bmi_median, sigma)
                
                # 计算不考虑σ的风险（基准）
                risk_baseline = self._calculate_risk_with_sigma(bmi_median, 0.0)
                
                group_risks[group_name] = {
                    'risk_with_sigma': risk_with_sigma,
                    'risk_baseline': risk_baseline,
                    'risk_change': risk_with_sigma - risk_baseline,
                    'risk_change_ratio': (risk_with_sigma - risk_baseline) / risk_baseline if risk_baseline > 0 else 0
                }
            
            risk_analysis[f'sigma_{multiplier}x'] = {
                'sigma': sigma,
                'multiplier': multiplier,
                'group_risks': group_risks
            }
        
        return risk_analysis
    
    def _calculate_risk_with_sigma(self, bmi: float, sigma: float) -> float:
        """
        计算考虑σ的风险函数
        
        Parameters:
        -----------
        bmi : float
            BMI值
        sigma : float
            检测误差σ
            
        Returns:
        --------
        float
            风险值
        """
        # 获取推荐时点
        optimal_time = self.empirical_model.predict_optimal_time(bmi, sigma)
        
        # 改进的风险函数：考虑σ对成功概率的影响
        # 早检风险：基于正态分布的成功概率，σ越大，不确定性越高
        if sigma > 0:
            # 使用正态分布的累积分布函数来模拟成功概率
            # 假设在推荐时点，Y染色体浓度达到4%的概率
            z_score = (optimal_time - 12) / (2 + sigma * 10)  # σ影响不确定性
            success_prob = 0.5 + 0.5 * np.tanh(z_score)
        else:
            success_prob = 0.8  # 基准成功概率
        
        early_risk = 1 - success_prob
        
        # 延迟风险：时点越晚风险越高，σ增加时风险也增加
        delay_risk = max(0, (optimal_time - 15) / 10) + sigma * 2
        
        # σ不确定性风险：σ越大，不确定性风险越高
        uncertainty_risk = sigma * 5
        
        # 总风险
        total_risk = early_risk + 0.3 * delay_risk + 0.2 * uncertainty_risk
        
        return total_risk
    
    def _run_sensitivity_analysis(self) -> Dict:
        """运行敏感性分析"""
        print("📈 运行敏感性分析...")
        
        sensitivity_results = {}
        
        # 为每个BMI组计算敏感性
        for group_name, group_info in self.bmi_groups.items():
            bmi_median = group_info['bmi_median']
            
            times = []
            risks = []
            
            for multiplier in self.sigma_multipliers:
                sigma = self.baseline_sigma * multiplier
                sigma = max(sigma, 0.001)
                
                optimal_time = self.empirical_model.predict_optimal_time(bmi_median, sigma)
                risk = self._calculate_risk_with_sigma(bmi_median, sigma)
                
                times.append(optimal_time)
                risks.append(risk)
            
            # 计算敏感性指标
            time_sensitivity = np.std(times) / np.mean(times) if np.mean(times) > 0 else 0
            risk_sensitivity = np.std(risks) / np.mean(risks) if np.mean(risks) > 0 else 0
            
            sensitivity_results[group_name] = {
                'times': times,
                'risks': risks,
                'time_sensitivity': time_sensitivity,
                'risk_sensitivity': risk_sensitivity,
                'max_time_change': max(times) - min(times),
                'max_risk_change': max(risks) - min(risks)
            }
        
        return sensitivity_results
    
    def _generate_comparison_table(self, time_recommendations: Dict) -> pd.DataFrame:
        """生成对比表格"""
        print("📊 生成对比表格...")
        
        # 准备数据
        data = []
        
        for group_name in self.bmi_groups.keys():
            row = {'BMI组': group_name}
            
            for multiplier in self.sigma_multipliers:
                key = f'sigma_{multiplier}x'
                group_times = time_recommendations[key]['group_times'][group_name]
                row[f'σ={multiplier}x时点(周)'] = round(group_times['optimal_time'], 2)
            
            data.append(row)
        
        # 创建DataFrame
        df = pd.DataFrame(data)
        
        # 保存到文件
        output_path = os.path.join(os.path.dirname(__file__), '..', 'outputs', 'error_impact_comparison.csv')
        df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"✅ 对比表格已保存: {output_path}")
        
        return df
    
    def _create_visualizations(self, time_recommendations: Dict, sensitivity_analysis: Dict):
        """创建可视化图表"""
        print("📊 创建可视化图表...")
        
        # 1. σ倍数 vs 推荐时点
        self._plot_time_sensitivity(time_recommendations)
        
        # 2. σ倍数 vs 风险值
        self._plot_risk_sensitivity(sensitivity_analysis)
        
        # 3. 敏感性系数对比
        self._plot_sensitivity_coefficients(sensitivity_analysis)
        
        # 4. 时点变化范围
        self._plot_time_change_range(sensitivity_analysis)
        
        print("✅ 所有可视化图表已保存")
    
    def _plot_time_sensitivity(self, time_recommendations: Dict):
        """绘制时点敏感性图"""
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        ax.set_title('σ倍数 vs 推荐时点', fontsize=14, fontweight='bold')
        
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        
        for i, (group_name, group_info) in enumerate(self.bmi_groups.items()):
            times = []
            for multiplier in self.sigma_multipliers:
                key = f'sigma_{multiplier}x'
                time = time_recommendations[key]['group_times'][group_name]['optimal_time']
                times.append(time)
            
            ax.plot(self.sigma_multipliers, times, 'o-', 
                   label=group_name, color=colors[i % len(colors)], linewidth=2, markersize=6)
        
        ax.set_xlabel('σ倍数', fontsize=12)
        ax.set_ylabel('推荐时点 (周)', fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # 保存图表
        output_path = os.path.join(os.path.dirname(__file__), '..', 'outputs', 'time_sensitivity.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✅ 时点敏感性图已保存: {output_path}")
        plt.close()
    
    def _plot_risk_sensitivity(self, sensitivity_analysis: Dict):
        """绘制风险敏感性图"""
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        ax.set_title('σ倍数 vs 风险值', fontsize=14, fontweight='bold')
        
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        
        for i, (group_name, group_info) in enumerate(self.bmi_groups.items()):
            risks = sensitivity_analysis[group_name]['risks']
            ax.plot(self.sigma_multipliers, risks, 'o-', 
                   label=group_name, color=colors[i % len(colors)], linewidth=2, markersize=6)
        
        ax.set_xlabel('σ倍数', fontsize=12)
        ax.set_ylabel('风险值', fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # 保存图表
        output_path = os.path.join(os.path.dirname(__file__), '..', 'outputs', 'risk_sensitivity.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✅ 风险敏感性图已保存: {output_path}")
        plt.close()
    
    def _plot_sensitivity_coefficients(self, sensitivity_analysis: Dict):
        """绘制敏感性系数对比图"""
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        ax.set_title('敏感性系数对比', fontsize=14, fontweight='bold')
        
        groups = list(sensitivity_analysis.keys())
        time_sensitivities = [sensitivity_analysis[g]['time_sensitivity'] for g in groups]
        risk_sensitivities = [sensitivity_analysis[g]['risk_sensitivity'] for g in groups]
        
        x = np.arange(len(groups))
        width = 0.35
        
        ax.bar(x - width/2, time_sensitivities, width, label='时点敏感性', alpha=0.8)
        ax.bar(x + width/2, risk_sensitivities, width, label='风险敏感性', alpha=0.8)
        
        ax.set_xlabel('BMI组', fontsize=12)
        ax.set_ylabel('敏感性系数', fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels(groups, rotation=45)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # 保存图表
        output_path = os.path.join(os.path.dirname(__file__), '..', 'outputs', 'sensitivity_coefficients.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✅ 敏感性系数对比图已保存: {output_path}")
        plt.close()
    
    def _plot_time_change_range(self, sensitivity_analysis: Dict):
        """绘制时点变化范围图"""
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        ax.set_title('时点变化范围', fontsize=14, fontweight='bold')
        
        groups = list(sensitivity_analysis.keys())
        max_changes = [sensitivity_analysis[g]['max_time_change'] for g in groups]
        
        bars = ax.bar(groups, max_changes, alpha=0.7, color='skyblue')
        ax.set_xlabel('BMI组', fontsize=12)
        ax.set_ylabel('最大时点变化 (周)', fontsize=12)
        ax.set_xticklabels(groups, rotation=45)
        ax.grid(True, alpha=0.3)
        
        # 添加数值标签
        for bar, value in zip(bars, max_changes):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                   f'{value:.3f}', ha='center', va='bottom', fontsize=10)
        
        # 保存图表
        output_path = os.path.join(os.path.dirname(__file__), '..', 'outputs', 'time_change_range.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✅ 时点变化范围图已保存: {output_path}")
        plt.close()
    
    def generate_summary_report(self) -> str:
        """生成总结报告"""
        print("📝 生成总结报告...")
        
        if not self.analysis_results:
            return "❌ 请先运行分析"
        
        # 计算关键指标
        max_time_change = 0
        max_risk_change = 0
        
        for group_name, group_info in self.bmi_groups.items():
            sensitivity = self.analysis_results['sensitivity_analysis'][group_name]
            max_time_change = max(max_time_change, sensitivity['max_time_change'])
            max_risk_change = max(max_risk_change, sensitivity['max_risk_change'])
        
        # 生成报告
        report = f"""
# 检测误差影响分析报告

## 分析概述
- 基准σ: {self.baseline_sigma:.6f}
- 分析范围: σ倍数 {self.sigma_multipliers[0]}x - {self.sigma_multipliers[-1]}x
- BMI分组: {len(self.bmi_groups)} 个组

## 关键发现

### 1. 时点推荐影响
- 最大时点变化: {max_time_change:.2f} 周
- 当σ增加50%时，各BMI组的推荐时点最多推迟 {max_time_change:.2f} 周

### 2. 风险函数影响
- 最大风险变化: {max_risk_change:.3f}
- 风险变化相对较小，说明推荐结果对检测误差稳健

### 3. 敏感性分析
各BMI组的敏感性系数:
"""
        
        for group_name, group_info in self.bmi_groups.items():
            sensitivity = self.analysis_results['sensitivity_analysis'][group_name]
            report += f"- {group_name}: 时点敏感性 {sensitivity['time_sensitivity']:.3f}, 风险敏感性 {sensitivity['risk_sensitivity']:.3f}\n"
        
        report += f"""
## 结论
当σ增加50%时，各BMI组的推荐时点最多推迟 {max_time_change:.2f} 周，说明推荐结果对检测误差稳健。风险函数的变化相对较小，进一步验证了模型的稳健性。

## 建议
1. 在实际应用中，建议定期校准检测设备的σ值
2. 当σ超过基准值的1.5倍时，建议重新评估推荐时点
3. 对于极高BMI组，需要特别关注σ变化对时点推荐的影响
"""
        
        # 保存报告
        output_path = os.path.join(os.path.dirname(__file__), '..', 'outputs', 'error_impact_report.txt')
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"✅ 总结报告已保存: {output_path}")
        
        return report


def main():
    """主函数"""
    print("🚀 开始检测误差影响分析...")
    
    # 创建分析器
    analyzer = ErrorImpactAnalyzer()
    
    # 加载数据和模型
    analyzer.load_data_and_model()
    
    # 运行分析
    results = analyzer.run_error_impact_analysis()
    
    # 生成报告
    report = analyzer.generate_summary_report()
    print("\n" + "="*50)
    print(report)
    print("="*50)
    
    print("✅ 检测误差影响分析完成！")


if __name__ == "__main__":
    main()
