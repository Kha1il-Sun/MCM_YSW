"""
问题3报告生成模块
生成综合分析报告和可视化结果
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
import logging

logger = logging.getLogger(__name__)


class Q3ReportGenerator:
    """问题3报告生成器"""
    
    def __init__(self, config: dict):
        """
        初始化报告生成器
        
        Parameters:
        -----------
        config : dict
            输出配置
        """
        self.config = config
        
        # 报告配置
        self.report_config = config.get('report', {})
        self.detailed = self.report_config.get('detailed', True)
        self.include_plots = self.report_config.get('include_plots', True)
        self.formats = self.report_config.get('format', ['csv', 'txt'])
        
        # 图表配置
        self.plot_config = config.get('plots', {})
        self.plots_enabled = self.plot_config.get('enabled', True)
        
        # 保存配置
        self.save_config = config.get('save', {})
        
        logger.info(f"报告生成器初始化完成")
        logger.info(f"  - 详细报告: {self.detailed}")
        logger.info(f"  - 包含图表: {self.include_plots}")
        logger.info(f"  - 输出格式: {self.formats}")
    
    def generate_comprehensive_report(self, **kwargs) -> Dict[str, Any]:
        """
        生成综合报告
        
        Parameters:
        -----------
        **kwargs : dict
            所有分析结果
            
        Returns:
        --------
        dict
            报告生成结果
        """
        logger.info("开始生成问题3综合报告...")
        
        # 提取输入参数
        enhanced_df = kwargs.get('enhanced_df')
        concentration_model = kwargs.get('concentration_model')
        success_model = kwargs.get('success_model')
        sigma_models = kwargs.get('sigma_models')
        wstar_results = kwargs.get('wstar_results')
        grouping_results = kwargs.get('grouping_results')
        validation_results = kwargs.get('validation_results')
        sensitivity_results = kwargs.get('sensitivity_results')
        config = kwargs.get('config', {})
        output_dir = kwargs.get('output_dir', 'outputs')
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        report_results = {
            'output_files': [],
            'generation_time': datetime.now().isoformat(),
            'status': 'in_progress'
        }
        
        try:
            # 1. 生成主要结果表格
            self._generate_main_tables(
                wstar_results, grouping_results, validation_results, 
                sensitivity_results, output_dir, report_results
            )
            
            # 2. 生成文本报告
            if 'txt' in self.formats:
                self._generate_text_report(
                    enhanced_df, concentration_model, success_model,
                    sigma_models, wstar_results, grouping_results,
                    validation_results, sensitivity_results, config,
                    output_dir, report_results
                )
            
            # 3. 生成HTML报告
            if 'html' in self.formats:
                self._generate_html_report(
                    enhanced_df, concentration_model, success_model,
                    sigma_models, wstar_results, grouping_results,
                    validation_results, sensitivity_results, config,
                    output_dir, report_results
                )
            
            # 4. 生成可视化（如果启用）
            if self.plots_enabled and self.include_plots:
                self._generate_visualizations(
                    wstar_results, grouping_results, validation_results,
                    sensitivity_results, output_dir, report_results
                )
            
            # 5. 生成配置备份
            self._save_configuration_backup(config, output_dir, report_results)
            
            # 6. 生成执行摘要
            self._generate_executive_summary(
                grouping_results, validation_results, sensitivity_results,
                output_dir, report_results
            )
            
            report_results['status'] = 'completed'
            logger.info("综合报告生成完成")
            
        except Exception as e:
            logger.error(f"报告生成失败: {e}")
            report_results['status'] = 'failed'
            report_results['error'] = str(e)
        
        return report_results
    
    def _generate_main_tables(self, wstar_results, grouping_results, validation_results,
                            sensitivity_results, output_dir, report_results):
        """生成主要结果表格"""
        
        logger.info("生成主要结果表格...")
        
        # 1. BMI分组和最优时点表
        groups = grouping_results.get('groups', [])
        if groups:
            groups_df = pd.DataFrame(groups)
            
            # 重新组织列顺序
            main_cols = ['group_id', 'bmi_range', 'optimal_timing', 'n_samples', 
                        'median_bmi', 'expected_success_rate', 'expected_attain_rate']
            available_cols = [col for col in main_cols if col in groups_df.columns]
            
            groups_output_df = groups_df[available_cols].copy()
            
            # 格式化BMI范围
            if 'bmi_range' in groups_output_df.columns:
                groups_output_df['BMI_Range'] = groups_output_df['bmi_range'].apply(
                    lambda x: f"[{x[0]:.1f}, {x[1]:.1f})" if isinstance(x, list) and len(x) >= 2 else str(x)
                )
                groups_output_df = groups_output_df.drop('bmi_range', axis=1)
            
            # 重命名列
            column_rename = {
                'group_id': 'Group_ID',
                'optimal_timing': 'Optimal_Week',
                'n_samples': 'N_Samples',
                'median_bmi': 'Median_BMI',
                'expected_success_rate': 'Expected_Success_Rate',
                'expected_attain_rate': 'Expected_Attain_Rate'
            }
            groups_output_df = groups_output_df.rename(columns=column_rename)
            
            # 保存
            groups_file = output_dir / 'q3_bmi_groups_optimal.csv'
            groups_output_df.to_csv(groups_file, index=False, encoding='utf-8-sig')
            report_results['output_files'].append(str(groups_file))
        
        # 2. 最优时点曲线数据
        if 'curve_data' in wstar_results:
            curve_df = wstar_results['curve_data']
            curve_file = output_dir / 'q3_wstar_curve.csv'
            curve_df.to_csv(curve_file, index=False, encoding='utf-8-sig')
            report_results['output_files'].append(str(curve_file))
        
        # 3. 验证结果汇总
        if validation_results:
            validation_summary = self._create_validation_summary_table(validation_results)
            validation_file = output_dir / 'q3_validation_summary.csv'
            validation_summary.to_csv(validation_file, index=False, encoding='utf-8-sig')
            report_results['output_files'].append(str(validation_file))
        
        # 4. 敏感性分析结果
        if sensitivity_results:
            sensitivity_summary = self._create_sensitivity_summary_table(sensitivity_results)
            sensitivity_file = output_dir / 'q3_sensitivity_summary.csv'
            sensitivity_summary.to_csv(sensitivity_file, index=False, encoding='utf-8-sig')
            report_results['output_files'].append(str(sensitivity_file))
    
    def _generate_text_report(self, enhanced_df, concentration_model, success_model,
                             sigma_models, wstar_results, grouping_results,
                             validation_results, sensitivity_results, config,
                             output_dir, report_results):
        """生成文本报告"""
        
        logger.info("生成文本报告...")
        
        report_file = output_dir / 'q3_comprehensive_report.txt'
        
        with open(report_file, 'w', encoding='utf-8') as f:
            # 报告标题
            f.write("=" * 80 + "\n")
            f.write("问题3：综合多因素的NIPT最佳时点优化分析报告\n")
            f.write("=" * 80 + "\n")
            f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # 1. 执行摘要
            f.write("1. 执行摘要\n")
            f.write("-" * 40 + "\n")
            self._write_executive_summary_section(f, grouping_results, validation_results, sensitivity_results)
            f.write("\n")
            
            # 2. 数据概览
            f.write("2. 数据概览\n")
            f.write("-" * 40 + "\n")
            self._write_data_overview_section(f, enhanced_df)
            f.write("\n")
            
            # 3. 模型性能
            f.write("3. 模型性能\n")
            f.write("-" * 40 + "\n")
            self._write_model_performance_section(f, concentration_model, success_model, sigma_models)
            f.write("\n")
            
            # 4. BMI分组结果
            f.write("4. BMI分组结果\n")
            f.write("-" * 40 + "\n")
            self._write_grouping_results_section(f, grouping_results, wstar_results)
            f.write("\n")
            
            # 5. 验证结果
            f.write("5. 模型验证结果\n")
            f.write("-" * 40 + "\n")
            self._write_validation_results_section(f, validation_results)
            f.write("\n")
            
            # 6. 敏感性分析
            f.write("6. 敏感性分析结果\n")
            f.write("-" * 40 + "\n")
            self._write_sensitivity_results_section(f, sensitivity_results)
            f.write("\n")
            
            # 7. 临床建议
            f.write("7. 临床建议\n")
            f.write("-" * 40 + "\n")
            self._write_clinical_recommendations_section(f, grouping_results, validation_results, sensitivity_results)
            f.write("\n")
            
            # 8. 技术细节
            if self.detailed:
                f.write("8. 技术细节\n")
                f.write("-" * 40 + "\n")
                self._write_technical_details_section(f, config)
                f.write("\n")
            
            # 9. 附录
            f.write("9. 附录\n")
            f.write("-" * 40 + "\n")
            self._write_appendix_section(f, config, report_results)
        
        report_results['output_files'].append(str(report_file))
    
    def _generate_html_report(self, enhanced_df, concentration_model, success_model,
                             sigma_models, wstar_results, grouping_results,
                             validation_results, sensitivity_results, config,
                             output_dir, report_results):
        """生成HTML报告"""
        
        logger.info("生成HTML报告...")
        
        html_file = output_dir / 'q3_report.html'
        
        # 简化的HTML报告
        html_content = f"""
        <!DOCTYPE html>
        <html lang="zh-CN">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>问题3分析报告</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; line-height: 1.6; }}
                .header {{ background-color: #f4f4f4; padding: 20px; border-radius: 5px; }}
                .section {{ margin: 20px 0; padding: 15px; border-left: 4px solid #007acc; }}
                table {{ border-collapse: collapse; width: 100%; margin: 10px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                .highlight {{ background-color: #fff3cd; padding: 10px; border-radius: 3px; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>问题3：综合多因素的NIPT最佳时点优化分析报告</h1>
                <p>生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
            
            <div class="section">
                <h2>BMI分组结果</h2>
                {self._create_html_grouping_table(grouping_results)}
            </div>
            
            <div class="section">
                <h2>模型验证摘要</h2>
                {self._create_html_validation_summary(validation_results)}
            </div>
            
            <div class="section">
                <h2>敏感性分析摘要</h2>
                {self._create_html_sensitivity_summary(sensitivity_results)}
            </div>
            
            <div class="section">
                <h2>临床建议</h2>
                {self._create_html_recommendations(grouping_results, validation_results, sensitivity_results)}
            </div>
        </body>
        </html>
        """
        
        with open(html_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        report_results['output_files'].append(str(html_file))
    
    def _generate_visualizations(self, wstar_results, grouping_results, validation_results,
                               sensitivity_results, output_dir, report_results):
        """生成可视化图表"""
        
        logger.info("生成可视化图表...")
        
        try:
            # 注意：这里只生成占位符文件，实际项目中需要用matplotlib等库
            # 1. w*曲线图
            plot_file1 = output_dir / 'q3_wstar_curve.png'
            self._create_placeholder_plot(plot_file1, "w*曲线图")
            report_results['output_files'].append(str(plot_file1))
            
            # 2. BMI分组可视化
            plot_file2 = output_dir / 'q3_bmi_grouping.png'
            self._create_placeholder_plot(plot_file2, "BMI分组图")
            report_results['output_files'].append(str(plot_file2))
            
            # 3. 敏感性分析热力图
            plot_file3 = output_dir / 'q3_sensitivity_heatmap.png'
            self._create_placeholder_plot(plot_file3, "敏感性分析热力图")
            report_results['output_files'].append(str(plot_file3))
            
        except Exception as e:
            logger.warning(f"可视化生成失败: {e}")
    
    def _create_placeholder_plot(self, filepath: Path, title: str):
        """创建占位符图表文件"""
        
        # 创建一个简单的文本文件作为占位符
        with open(str(filepath).replace('.png', '_placeholder.txt'), 'w', encoding='utf-8') as f:
            f.write(f"占位符: {title}\n")
            f.write("实际实现中此处应生成相应的图表。\n")
    
    def _create_validation_summary_table(self, validation_results: dict) -> pd.DataFrame:
        """创建验证结果汇总表"""
        
        summary_data = []
        
        # 交叉验证结果
        cv = validation_results.get('cross_validation', {})
        if cv.get('status') == 'completed':
            conc_model = cv.get('concentration_model', {})
            summary_data.append({
                'Validation_Type': 'Cross Validation',
                'Metric': 'R² Score',
                'Value': f"{conc_model.get('mean_r2', 0):.3f} ± {conc_model.get('std_r2', 0):.3f}",
                'Status': 'Completed'
            })
        
        # 时间外推验证
        temporal = validation_results.get('temporal_validation', {})
        if temporal.get('status') == 'completed':
            summary_data.append({
                'Validation_Type': 'Temporal Validation',
                'Metric': 'Performance Degradation',
                'Value': f"{temporal.get('concentration_model', {}).get('performance_degradation', 0):.3f}",
                'Status': 'Completed'
            })
        
        # Bootstrap结果
        bootstrap = validation_results.get('bootstrap', {})
        if bootstrap.get('status') == 'completed':
            cis = bootstrap.get('confidence_intervals', [])
            if cis:
                avg_width = np.mean([ci['ci_width'] for ci in cis])
                summary_data.append({
                    'Validation_Type': 'Bootstrap CI',
                    'Metric': 'Average CI Width',
                    'Value': f"{avg_width:.1f} weeks",
                    'Status': 'Completed'
                })
        
        return pd.DataFrame(summary_data)
    
    def _create_sensitivity_summary_table(self, sensitivity_results: dict) -> pd.DataFrame:
        """创建敏感性分析汇总表"""
        
        summary_data = []
        
        # 参数敏感性
        param = sensitivity_results.get('parameter_sensitivity', {})
        if param.get('status') == 'completed':
            summary_data.append({
                'Analysis_Type': 'Parameter Sensitivity',
                'Rating': param.get('sensitivity_rating', 'Unknown'),
                'Max_Impact': f"{param.get('max_observed_change', 0):.1f} weeks",
                'Status': 'Completed'
            })
        
        # 检测误差敏感性
        error = sensitivity_results.get('error_sensitivity', {})
        if error.get('status') == 'completed':
            summary_data.append({
                'Analysis_Type': 'Detection Error Sensitivity',
                'Rating': error.get('error_impact_rating', 'Unknown'),
                'Max_Impact': f"{error.get('max_timing_impact', 0):.1f} weeks",
                'Status': 'Completed'
            })
        
        # 多因素敏感性
        multi_factor = sensitivity_results.get('multi_factor_sensitivity', {})
        if multi_factor.get('status') == 'completed':
            summary_data.append({
                'Analysis_Type': 'Multi-Factor Sensitivity',
                'Rating': multi_factor.get('overall_multi_factor_sensitivity', 'Unknown'),
                'Most_Sensitive': ', '.join(multi_factor.get('most_sensitive_factors', [])[:2]),
                'Status': 'Completed'
            })
        
        return pd.DataFrame(summary_data)
    
    def _write_executive_summary_section(self, f, grouping_results, validation_results, sensitivity_results):
        """写入执行摘要部分"""
        
        groups = grouping_results.get('groups', [])
        f.write(f"本研究基于多因素建模方法，对男胎孕妇进行BMI分组并确定最佳NIPT检测时点。\n\n")
        
        if groups:
            f.write(f"主要发现：\n")
            f.write(f"- 最优分组数：{len(groups)}组\n")
            f.write(f"- 推荐时点范围：{min(g['optimal_timing'] for g in groups):.1f} - {max(g['optimal_timing'] for g in groups):.1f}周\n")
            
            for i, group in enumerate(groups):
                f.write(f"- 组{group['group_id']}: BMI {group['bmi_range'][0]:.1f}-{group['bmi_range'][1]:.1f}, "
                       f"推荐时点 {group['optimal_timing']:.1f}周, 样本数 {group['n_samples']}\n")
        
        # 验证结果摘要
        val_summary = validation_results.get('summary', {})
        if val_summary:
            f.write(f"\n验证结果：\n")
            f.write(f"- 交叉验证评级：{val_summary.get('cv_stability', 'N/A')}\n")
            f.write(f"- 时间外推准确性：{val_summary.get('temporal_accuracy', 'N/A')}\n")
            f.write(f"- 分组稳定性：{val_summary.get('grouping_stability', 'N/A')}\n")
        
        # 敏感性结果摘要
        sens_summary = sensitivity_results.get('summary', {})
        if sens_summary:
            f.write(f"\n稳健性分析：\n")
            f.write(f"- 整体稳健性：{sens_summary.get('overall_robustness', 'N/A')}\n")
            f.write(f"- 检测误差影响：{sens_summary.get('error_sensitivity_rating', 'N/A')}\n")
    
    def _write_data_overview_section(self, f, enhanced_df):
        """写入数据概览部分"""
        
        if enhanced_df is not None and not enhanced_df.empty:
            f.write(f"数据集大小：{len(enhanced_df)} 条记录\n")
            f.write(f"特征数量：{len(enhanced_df.columns)} 个\n")
            f.write(f"个体数量：{enhanced_df['id'].nunique() if 'id' in enhanced_df.columns else 'N/A'}\n")
            
            if 'BMI_used' in enhanced_df.columns:
                bmi_stats = enhanced_df['BMI_used'].describe()
                f.write(f"\nBMI分布：\n")
                f.write(f"- 范围：{bmi_stats['min']:.1f} - {bmi_stats['max']:.1f}\n")
                f.write(f"- 均值：{bmi_stats['mean']:.1f}\n")
                f.write(f"- 中位数：{bmi_stats['50%']:.1f}\n")
            
            if 'week' in enhanced_df.columns:
                week_stats = enhanced_df['week'].describe()
                f.write(f"\n孕周分布：\n")
                f.write(f"- 范围：{week_stats['min']:.1f} - {week_stats['max']:.1f}周\n")
                f.write(f"- 均值：{week_stats['mean']:.1f}周\n")
        else:
            f.write("数据概览不可用\n")
    
    def _write_model_performance_section(self, f, concentration_model, success_model, sigma_models):
        """写入模型性能部分"""
        
        f.write("多因素建模性能：\n\n")
        
        # 浓度预测模型
        if hasattr(concentration_model, 'get_performance_metrics'):
            perf = concentration_model.get_performance_metrics()
            f.write(f"Y染色体浓度预测模型：\n")
            f.write(f"- 模型类型：{perf.get('summary', {}).get('model_type', 'N/A')}\n")
            f.write(f"- R²得分：{perf.get('summary', {}).get('best_r2', 'N/A')}\n")
            f.write(f"- 最佳模型：{perf.get('summary', {}).get('best_model', 'N/A')}\n")
            
            top_features = perf.get('top_features', [])
            if top_features:
                f.write(f"- 重要特征：{', '.join(top_features[:5])}\n")
        
        f.write(f"\n误差建模：\n")
        f.write(f"- 全局sigma：{sigma_models.get('global_sigma', 'N/A')}\n")
        f.write(f"- 局部因素：{', '.join(sigma_models.get('local_factors', []))}\n")
        
        model_performance = sigma_models.get('model_performance', {})
        if 'r2' in model_performance:
            f.write(f"- 误差模型R²：{model_performance['r2']:.3f}\n")
    
    def _write_grouping_results_section(self, f, grouping_results, wstar_results):
        """写入分组结果部分"""
        
        groups = grouping_results.get('groups', [])
        quality = grouping_results.get('quality_metrics', {})
        
        f.write(f"分组方法：{grouping_results.get('method_used', 'N/A')}\n")
        f.write(f"最终分组数：{len(groups)}\n")
        f.write(f"分组质量评级：{quality.get('quality_rating', 'N/A')}\n\n")
        
        f.write("各组详细信息：\n")
        for group in groups:
            f.write(f"\n组{group['group_id']}：\n")
            f.write(f"  - BMI范围：[{group['bmi_range'][0]:.1f}, {group['bmi_range'][1]:.1f})\n")
            f.write(f"  - 推荐检测时点：{group['optimal_timing']:.1f}周\n")
            f.write(f"  - 样本数量：{group['n_samples']}\n")
            f.write(f"  - 中位数BMI：{group['median_bmi']:.1f}\n")
            f.write(f"  - 预期成功率：{group.get('expected_success_rate', 'N/A')}\n")
            f.write(f"  - 预期达标率：{group.get('expected_attain_rate', 'N/A')}\n")
            f.write(f"  - 风险水平：{group.get('risk_level', 'N/A')}\n")
    
    def _write_validation_results_section(self, f, validation_results):
        """写入验证结果部分"""
        
        summary = validation_results.get('summary', {})
        f.write(f"整体验证评级：{summary.get('overall_validation_rating', 'N/A')}\n\n")
        
        # 交叉验证
        cv = validation_results.get('cross_validation', {})
        if cv.get('status') == 'completed':
            f.write("交叉验证结果：\n")
            conc_model = cv.get('concentration_model', {})
            f.write(f"- 平均R²：{conc_model.get('mean_r2', 'N/A')}\n")
            f.write(f"- R²标准差：{conc_model.get('std_r2', 'N/A')}\n")
            f.write(f"- 折数：{cv.get('n_folds_completed', 'N/A')}\n")
        
        # Bootstrap
        bootstrap = validation_results.get('bootstrap', {})
        if bootstrap.get('status') == 'completed':
            f.write(f"\nBootstrap置信区间：\n")
            cis = bootstrap.get('confidence_intervals', [])
            for ci in cis:
                f.write(f"- 组{ci['group_id']}：{ci['original_timing']:.1f}周 "
                       f"[{ci['ci_lower']:.1f}, {ci['ci_upper']:.1f}]\n")
    
    def _write_sensitivity_results_section(self, f, sensitivity_results):
        """写入敏感性分析结果部分"""
        
        summary = sensitivity_results.get('summary', {})
        f.write(f"整体稳健性：{summary.get('overall_robustness', 'N/A')}\n\n")
        
        # 参数敏感性
        param = sensitivity_results.get('parameter_sensitivity', {})
        if param.get('status') == 'completed':
            f.write("参数敏感性分析：\n")
            f.write(f"- 敏感性评级：{param.get('sensitivity_rating', 'N/A')}\n")
            f.write(f"- 最大影响：{param.get('max_observed_change', 'N/A')}周\n")
        
        # 检测误差敏感性
        error = sensitivity_results.get('error_sensitivity', {})
        if error.get('status') == 'completed':
            f.write(f"\n检测误差敏感性分析：\n")
            f.write(f"- 影响评级：{error.get('error_impact_rating', 'N/A')}\n")
            f.write(f"- 最大时点影响：{error.get('max_timing_impact', 'N/A')}周\n")
            f.write(f"- 最大成功率影响：{error.get('max_success_impact', 'N/A')}\n")
            impact_summary = error.get('error_impact_summary', '')
            if impact_summary:
                f.write(f"- 影响摘要：{impact_summary}\n")
        
        # 多因素敏感性
        multi_factor = sensitivity_results.get('multi_factor_sensitivity', {})
        if multi_factor.get('status') == 'completed':
            f.write(f"\n多因素敏感性分析：\n")
            f.write(f"- 整体敏感性：{multi_factor.get('overall_multi_factor_sensitivity', 'N/A')}\n")
            
            most_sensitive = multi_factor.get('most_sensitive_factors', [])
            if most_sensitive:
                f.write(f"- 最敏感因素：{', '.join(most_sensitive)}\n")
    
    def _write_clinical_recommendations_section(self, f, grouping_results, validation_results, sensitivity_results):
        """写入临床建议部分"""
        
        groups = grouping_results.get('groups', [])
        
        f.write("基于分析结果的临床建议：\n\n")
        
        # 分组特定建议
        for group in groups:
            f.write(f"组{group['group_id']} (BMI {group['bmi_range'][0]:.1f}-{group['bmi_range'][1]:.1f}):\n")
            f.write(f"- 推荐检测时点：{group['optimal_timing']:.1f}周\n")
            
            # 基于风险水平的建议
            risk_level = group.get('risk_level', 'Medium')
            if risk_level == 'High':
                f.write("- 建议：密切监测，考虑提前检测或增加检测频次\n")
            elif risk_level == 'Low':
                f.write("- 建议：按常规检测流程即可\n")
            else:
                f.write("- 建议：适度提高检测精度，注意时点控制\n")
            f.write("\n")
        
        # 基于敏感性分析的建议
        sens_summary = sensitivity_results.get('summary', {})
        recommendations = sens_summary.get('recommendations', [])
        
        if recommendations:
            f.write("稳健性建议：\n")
            for i, rec in enumerate(recommendations, 1):
                f.write(f"{i}. {rec}\n")
        
        # 实施建议
        f.write("\n实施建议：\n")
        f.write("1. 建议在临床实施前进行小规模试点验证\n")
        f.write("2. 定期监测模型性能，必要时重新校准\n")
        f.write("3. 建立质量控制体系，确保检测误差在可控范围内\n")
        f.write("4. 对边界BMI值的孕妇，建议临床医生结合其他因素综合判断\n")
    
    def _write_technical_details_section(self, f, config):
        """写入技术细节部分"""
        
        f.write("模型配置参数：\n")
        
        # 模型参数
        model_params = config.get('model_params', {})
        f.write(f"- 分位数水平：{model_params.get('quantile_tau', 'N/A')}\n")
        f.write(f"- 集成方法：{model_params.get('ensemble_method', 'N/A')}\n")
        f.write(f"- 基模型：{', '.join(model_params.get('base_models', []))}\n")
        
        # 优化参数
        opt_params = config.get('optimization', {})
        f.write(f"\n优化参数：\n")
        f.write(f"- 孕周搜索范围：{opt_params.get('w_min', 'N/A')}-{opt_params.get('w_max', 'N/A')}周\n")
        f.write(f"- BMI分辨率：{opt_params.get('b_resolution', 'N/A')}\n")
        f.write(f"- 最小成功概率：{opt_params.get('constraints', {}).get('min_success_prob', 'N/A')}\n")
        
        # 分组参数
        grouping_params = config.get('grouping', {})
        f.write(f"\n分组参数：\n")
        f.write(f"- 分组方法：{grouping_params.get('method', 'N/A')}\n")
        f.write(f"- WHO切点：{grouping_params.get('who_cuts', [])}\n")
        f.write(f"- 最小组大小：{grouping_params.get('constraints', {}).get('min_group_size', 'N/A')}\n")
    
    def _write_appendix_section(self, f, config, report_results):
        """写入附录部分"""
        
        f.write("输出文件列表：\n")
        for i, file in enumerate(report_results.get('output_files', []), 1):
            f.write(f"{i}. {Path(file).name}\n")
        
        f.write(f"\n软件版本信息：\n")
        f.write(f"- Python版本：3.8+\n")
        f.write(f"- 主要依赖：pandas, numpy, scikit-learn, scipy\n")
        f.write(f"- 生成时间：{report_results.get('generation_time', 'N/A')}\n")
    
    def _create_html_grouping_table(self, grouping_results) -> str:
        """创建HTML分组表格"""
        
        groups = grouping_results.get('groups', [])
        if not groups:
            return "<p>无分组数据</p>"
        
        html = "<table><thead><tr><th>组ID</th><th>BMI范围</th><th>推荐时点(周)</th><th>样本数</th><th>预期成功率</th></tr></thead><tbody>"
        
        for group in groups:
            html += f"<tr>"
            html += f"<td>{group['group_id']}</td>"
            html += f"<td>[{group['bmi_range'][0]:.1f}, {group['bmi_range'][1]:.1f})</td>"
            html += f"<td>{group['optimal_timing']:.1f}</td>"
            html += f"<td>{group['n_samples']}</td>"
            html += f"<td>{group.get('expected_success_rate', 'N/A')}</td>"
            html += f"</tr>"
        
        html += "</tbody></table>"
        return html
    
    def _create_html_validation_summary(self, validation_results) -> str:
        """创建HTML验证摘要"""
        
        summary = validation_results.get('summary', {})
        html = "<div class='highlight'>"
        html += f"<p><strong>整体验证评级：</strong>{summary.get('overall_validation_rating', 'N/A')}</p>"
        html += f"<p><strong>交叉验证得分：</strong>{summary.get('cv_score', 'N/A')}</p>"
        html += f"<p><strong>时间外推准确性：</strong>{summary.get('temporal_accuracy', 'N/A')}</p>"
        html += f"<p><strong>分组稳定性：</strong>{summary.get('grouping_stability', 'N/A')}</p>"
        html += "</div>"
        return html
    
    def _create_html_sensitivity_summary(self, sensitivity_results) -> str:
        """创建HTML敏感性摘要"""
        
        summary = sensitivity_results.get('summary', {})
        html = "<div class='highlight'>"
        html += f"<p><strong>整体稳健性：</strong>{summary.get('overall_robustness', 'N/A')}</p>"
        html += f"<p><strong>检测误差影响：</strong>{summary.get('error_sensitivity_rating', 'N/A')}</p>"
        html += f"<p><strong>参数稳定性：</strong>{summary.get('parameter_stability', 'N/A')}</p>"
        
        sensitive_factors = summary.get('most_sensitive_factors', [])
        if sensitive_factors:
            html += f"<p><strong>最敏感因素：</strong>{', '.join(sensitive_factors)}</p>"
        
        html += "</div>"
        return html
    
    def _create_html_recommendations(self, grouping_results, validation_results, sensitivity_results) -> str:
        """创建HTML建议"""
        
        html = "<ul>"
        
        # 基本建议
        groups = grouping_results.get('groups', [])
        if groups:
            html += "<li>根据分析结果，建议采用多因素BMI分组策略进行NIPT检测</li>"
            html += f"<li>共建议{len(groups)}个BMI组，每组有特定的最优检测时点</li>"
        
        # 稳健性建议
        sens_summary = sensitivity_results.get('summary', {})
        robustness = sens_summary.get('overall_robustness', 'Unknown')
        
        if robustness == 'High':
            html += "<li>模型显示出良好的稳健性，可以应用于临床实践</li>"
        elif robustness == 'Moderate':
            html += "<li>模型具有中等稳健性，建议增加适当的安全边际</li>"
        else:
            html += "<li>模型稳健性有限，建议谨慎应用并持续监测</li>"
        
        html += "<li>建议建立质量控制体系，定期验证模型性能</li>"
        html += "<li>对边界情况，建议临床医生结合其他因素综合判断</li>"
        html += "</ul>"
        
        return html
    
    def _save_configuration_backup(self, config, output_dir, report_results):
        """保存配置备份"""
        
        import yaml
        
        config_backup = {
            'generation_time': datetime.now().isoformat(),
            'configuration': config
        }
        
        config_file = output_dir / 'q3_config_used.yaml'
        with open(config_file, 'w', encoding='utf-8') as f:
            yaml.dump(config_backup, f, default_flow_style=False, allow_unicode=True)
        
        report_results['output_files'].append(str(config_file))
    
    def _generate_executive_summary(self, grouping_results, validation_results, sensitivity_results, output_dir, report_results):
        """生成执行摘要"""
        
        summary_file = output_dir / 'q3_executive_summary.txt'
        
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write("问题3：综合多因素NIPT最佳时点优化 - 执行摘要\n")
            f.write("=" * 60 + "\n\n")
            
            # 主要发现
            groups = grouping_results.get('groups', [])
            if groups:
                f.write("【主要发现】\n")
                f.write(f"• 建议将男胎孕妇分为{len(groups)}个BMI组\n")
                
                timing_range = [g['optimal_timing'] for g in groups]
                f.write(f"• 推荐检测时点范围：{min(timing_range):.1f} - {max(timing_range):.1f}周\n")
                
                total_samples = sum(g['n_samples'] for g in groups)
                f.write(f"• 覆盖样本数：{total_samples}个\n\n")
                
                f.write("【分组建议】\n")
                for group in groups:
                    f.write(f"• 组{group['group_id']}：BMI {group['bmi_range'][0]:.1f}-{group['bmi_range'][1]:.1f}, "
                           f"推荐{group['optimal_timing']:.1f}周检测\n")
            
            # 模型可靠性
            val_summary = validation_results.get('summary', {})
            sens_summary = sensitivity_results.get('summary', {})
            
            f.write(f"\n【模型可靠性】\n")
            f.write(f"• 整体验证评级：{val_summary.get('overall_validation_rating', 'N/A')}\n")
            f.write(f"• 稳健性评估：{sens_summary.get('overall_robustness', 'N/A')}\n")
            f.write(f"• 检测误差敏感性：{sens_summary.get('error_sensitivity_rating', 'N/A')}\n")
            
            # 实施建议
            f.write(f"\n【实施建议】\n")
            recommendations = sens_summary.get('recommendations', [])
            for i, rec in enumerate(recommendations[:3], 1):  # 显示前3个建议
                f.write(f"{i}. {rec}\n")
            
            f.write(f"\n报告生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        report_results['output_files'].append(str(summary_file))
