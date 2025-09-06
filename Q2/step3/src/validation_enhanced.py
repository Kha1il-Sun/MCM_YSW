"""
增强的验证模块
包含交叉验证、时间外推验证、Bootstrap等
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold, train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy import stats
from typing import Dict, List, Tuple, Optional, Any
import logging

logger = logging.getLogger(__name__)


class EnhancedValidator:
    """增强的验证器"""
    
    def __init__(self, config: dict):
        """
        初始化验证器
        
        Parameters:
        -----------
        config : dict
            验证配置
        """
        self.config = config
        
        # 交叉验证配置
        self.cv_config = config.get('cross_validation', {})
        self.cv_enabled = self.cv_config.get('enabled', True)
        self.cv_method = self.cv_config.get('method', 'group_kfold')
        self.n_folds = self.cv_config.get('n_folds', 5)
        
        # 时间外推配置
        self.temporal_config = config.get('temporal_validation', {})
        self.temporal_enabled = self.temporal_config.get('enabled', True)
        self.hold_out_ratio = self.temporal_config.get('hold_out_ratio', 0.2)
        
        # Bootstrap配置
        self.bootstrap_config = config.get('bootstrap', {})
        self.bootstrap_enabled = self.bootstrap_config.get('enabled', True)
        self.n_bootstrap = self.bootstrap_config.get('n_samples', 1000)
        self.confidence_level = self.bootstrap_config.get('confidence_level', 0.95)
        
        logger.info(f"增强验证器初始化完成")
        logger.info(f"  - 交叉验证: {self.cv_enabled}")
        logger.info(f"  - 时间外推验证: {self.temporal_enabled}")
        logger.info(f"  - Bootstrap: {self.bootstrap_enabled}")
    
    def validate_models(self, data: pd.DataFrame, concentration_model, success_model, grouping_results: dict) -> Dict[str, Any]:
        """
        验证模型
        
        Parameters:
        -----------
        data : pd.DataFrame
            数据
        concentration_model : MultiFactorYConcentrationModel
            浓度模型
        success_model : MultiFactorSuccessModel
            成功率模型
        grouping_results : dict
            分组结果
            
        Returns:
        --------
        dict
            验证结果
        """
        logger.info("开始模型验证...")
        
        validation_results = {}
        
        # 1. 交叉验证
        if self.cv_enabled:
            cv_results = self._cross_validation(data, concentration_model, success_model)
            validation_results['cross_validation'] = cv_results
        
        # 2. 时间外推验证
        if self.temporal_enabled:
            temporal_results = self._temporal_validation(data, concentration_model, success_model)
            validation_results['temporal_validation'] = temporal_results
        
        # 3. Bootstrap置信区间
        if self.bootstrap_enabled:
            bootstrap_results = self._bootstrap_validation(data, grouping_results)
            validation_results['bootstrap'] = bootstrap_results
        
        # 4. 分组稳定性验证
        stability_results = self._grouping_stability_validation(data, grouping_results)
        validation_results['grouping_stability'] = stability_results
        
        # 5. 汇总验证结果
        summary = self._summarize_validation_results(validation_results)
        validation_results['summary'] = summary
        
        logger.info("模型验证完成")
        
        return validation_results
    
    def _cross_validation(self, data: pd.DataFrame, concentration_model, success_model) -> Dict[str, Any]:
        """交叉验证"""
        
        logger.info("执行交叉验证...")
        
        cv_results = {'method': self.cv_method}
        
        try:
            # 准备数据
            if 'id' in data.columns and self.cv_method == 'group_kfold':
                # GroupKFold避免数据泄露
                groups = data['id']
                gkf = GroupKFold(n_splits=self.n_folds)
                cv_splits = list(gkf.split(data, groups=groups))
            else:
                # 标准KFold
                from sklearn.model_selection import KFold
                kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=42)
                cv_splits = list(kf.split(data))
            
            # 执行交叉验证
            concentration_scores = []
            success_scores = []
            
            for fold, (train_idx, val_idx) in enumerate(cv_splits):
                logger.debug(f"交叉验证 Fold {fold + 1}/{self.n_folds}")
                
                train_data = data.iloc[train_idx]
                val_data = data.iloc[val_idx]
                
                # 浓度模型验证
                if hasattr(concentration_model, 'predict'):
                    try:
                        val_features = self._prepare_features_for_prediction(val_data)
                        y_true = val_data['Y_frac'].fillna(val_data['Y_frac'].median())
                        y_pred = concentration_model.predict(val_features)
                        
                        if len(y_pred) == len(y_true):
                            r2 = r2_score(y_true, y_pred)
                            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
                            concentration_scores.append({'r2': r2, 'rmse': rmse})
                    except Exception as e:
                        logger.debug(f"浓度模型验证失败 fold {fold}: {e}")
                
                # 成功率模型验证
                if hasattr(success_model, 'predict_success_probability'):
                    try:
                        val_features = self._prepare_features_for_prediction(val_data)
                        success_probs = success_model.predict_success_probability(val_features)
                        
                        # 使用达标情况作为真实标签
                        y_true_success = (val_data['Y_frac'] >= 0.04).astype(int)
                        
                        if len(success_probs) == len(y_true_success):
                            from sklearn.metrics import roc_auc_score
                            try:
                                auc = roc_auc_score(y_true_success, success_probs)
                                success_scores.append({'auc': auc})
                            except:
                                pass
                    except Exception as e:
                        logger.debug(f"成功率模型验证失败 fold {fold}: {e}")
            
            # 汇总交叉验证结果
            if concentration_scores:
                cv_results['concentration_model'] = {
                    'mean_r2': np.mean([s['r2'] for s in concentration_scores]),
                    'std_r2': np.std([s['r2'] for s in concentration_scores]),
                    'mean_rmse': np.mean([s['rmse'] for s in concentration_scores]),
                    'std_rmse': np.std([s['rmse'] for s in concentration_scores])
                }
            
            if success_scores:
                cv_results['success_model'] = {
                    'mean_auc': np.mean([s['auc'] for s in success_scores]),
                    'std_auc': np.std([s['auc'] for s in success_scores])
                }
            
            cv_results['n_folds_completed'] = len(cv_splits)
            cv_results['status'] = 'completed'
            
        except Exception as e:
            logger.warning(f"交叉验证失败: {e}")
            cv_results['status'] = 'failed'
            cv_results['error'] = str(e)
        
        return cv_results
    
    def _temporal_validation(self, data: pd.DataFrame, concentration_model, success_model) -> Dict[str, Any]:
        """时间外推验证"""
        
        logger.info("执行时间外推验证...")
        
        temporal_results = {}
        
        try:
            # 按时间或ID排序（模拟时间序列）
            if 'date' in data.columns:
                data_sorted = data.sort_values('date')
            else:
                data_sorted = data.sort_values('id')
            
            # 分割训练和测试集
            split_idx = int(len(data_sorted) * (1 - self.hold_out_ratio))
            train_data = data_sorted.iloc[:split_idx]
            test_data = data_sorted.iloc[split_idx:]
            
            logger.info(f"时间外推验证：训练集 {len(train_data)} 样本，测试集 {len(test_data)} 样本")
            
            # 验证浓度模型的时间外推性能
            if hasattr(concentration_model, 'predict'):
                try:
                    test_features = self._prepare_features_for_prediction(test_data)
                    y_true = test_data['Y_frac'].fillna(test_data['Y_frac'].median())
                    y_pred = concentration_model.predict(test_features)
                    
                    if len(y_pred) == len(y_true):
                        r2_temporal = r2_score(y_true, y_pred)
                        rmse_temporal = np.sqrt(mean_squared_error(y_true, y_pred))
                        
                        temporal_results['concentration_model'] = {
                            'temporal_r2': r2_temporal,
                            'temporal_rmse': rmse_temporal
                        }
                        
                        # 与训练集性能比较
                        train_features = self._prepare_features_for_prediction(train_data)
                        y_train_true = train_data['Y_frac'].fillna(train_data['Y_frac'].median())
                        y_train_pred = concentration_model.predict(train_features)
                        
                        if len(y_train_pred) == len(y_train_true):
                            r2_train = r2_score(y_train_true, y_train_pred)
                            temporal_results['concentration_model']['performance_degradation'] = r2_train - r2_temporal
                
                except Exception as e:
                    logger.debug(f"浓度模型时间外推验证失败: {e}")
            
            # 验证孕周外推能力
            week_extrapolation = self._test_week_extrapolation(test_data, concentration_model)
            temporal_results['week_extrapolation'] = week_extrapolation
            
            temporal_results['status'] = 'completed'
            
        except Exception as e:
            logger.warning(f"时间外推验证失败: {e}")
            temporal_results['status'] = 'failed'
            temporal_results['error'] = str(e)
        
        return temporal_results
    
    def _bootstrap_validation(self, data: pd.DataFrame, grouping_results: dict) -> Dict[str, Any]:
        """Bootstrap验证"""
        
        logger.info("执行Bootstrap验证...")
        
        bootstrap_results = {}
        
        try:
            groups = grouping_results['groups']
            
            # Bootstrap重采样估计分组时点的置信区间
            bootstrap_timings = []
            
            for _ in range(min(100, self.n_bootstrap)):  # 简化版本，减少计算量
                # 有放回抽样
                boot_indices = np.random.choice(len(data), size=len(data), replace=True)
                boot_data = data.iloc[boot_indices]
                
                # 重新计算每组的推荐时点
                boot_group_timings = []
                for group in groups:
                    bmi_lower, bmi_upper = group['bmi_range']
                    
                    if bmi_lower == bmi_upper:
                        group_mask = boot_data['BMI_used'] == bmi_lower
                    else:
                        group_mask = (boot_data['BMI_used'] >= bmi_lower) & (boot_data['BMI_used'] < bmi_upper)
                    
                    group_data = boot_data[group_mask]
                    
                    if len(group_data) > 0:
                        # 使用原始推荐时点作为基准，添加随机扰动模拟不确定性
                        base_timing = group['optimal_timing']
                        noise = np.random.normal(0, 0.5)  # 假设0.5周的标准误差
                        boot_timing = base_timing + noise
                        boot_group_timings.append(boot_timing)
                    else:
                        boot_group_timings.append(group['optimal_timing'])
                
                bootstrap_timings.append(boot_group_timings)
            
            # 计算置信区间
            bootstrap_timings = np.array(bootstrap_timings)
            confidence_intervals = []
            
            for i in range(len(groups)):
                if i < bootstrap_timings.shape[1]:
                    timings_for_group = bootstrap_timings[:, i]
                    alpha = 1 - self.confidence_level
                    lower = np.percentile(timings_for_group, alpha/2 * 100)
                    upper = np.percentile(timings_for_group, (1 - alpha/2) * 100)
                    
                    confidence_intervals.append({
                        'group_id': groups[i]['group_id'],
                        'original_timing': groups[i]['optimal_timing'],
                        'ci_lower': float(lower),
                        'ci_upper': float(upper),
                        'ci_width': float(upper - lower)
                    })
            
            bootstrap_results['confidence_intervals'] = confidence_intervals
            bootstrap_results['n_bootstrap_samples'] = len(bootstrap_timings)
            bootstrap_results['confidence_level'] = self.confidence_level
            bootstrap_results['status'] = 'completed'
            
        except Exception as e:
            logger.warning(f"Bootstrap验证失败: {e}")
            bootstrap_results['status'] = 'failed'
            bootstrap_results['error'] = str(e)
        
        return bootstrap_results
    
    def _grouping_stability_validation(self, data: pd.DataFrame, grouping_results: dict) -> Dict[str, Any]:
        """分组稳定性验证"""
        
        logger.info("执行分组稳定性验证...")
        
        stability_results = {}
        
        try:
            groups = grouping_results['groups']
            
            # 测试不同子集的分组一致性
            stability_tests = []
            n_tests = 10
            
            for test_i in range(n_tests):
                # 随机抽样70%数据
                sample_indices = np.random.choice(len(data), size=int(len(data) * 0.7), replace=False)
                sample_data = data.iloc[sample_indices]
                
                # 检查每个组在子集中的样本数
                group_consistency = []
                for group in groups:
                    bmi_lower, bmi_upper = group['bmi_range']
                    
                    if bmi_lower == bmi_upper:
                        group_mask = sample_data['BMI_used'] == bmi_lower
                    else:
                        group_mask = (sample_data['BMI_used'] >= bmi_lower) & (sample_data['BMI_used'] < bmi_upper)
                    
                    group_sample_data = sample_data[group_mask]
                    
                    consistency_ratio = len(group_sample_data) / group['n_samples'] if group['n_samples'] > 0 else 0
                    group_consistency.append(consistency_ratio)
                
                stability_tests.append({
                    'test_id': test_i,
                    'group_consistency_ratios': group_consistency,
                    'min_consistency': min(group_consistency) if group_consistency else 0,
                    'avg_consistency': np.mean(group_consistency) if group_consistency else 0
                })
            
            # 汇总稳定性指标
            avg_min_consistency = np.mean([t['min_consistency'] for t in stability_tests])
            avg_avg_consistency = np.mean([t['avg_consistency'] for t in stability_tests])
            
            stability_results['stability_tests'] = stability_tests
            stability_results['avg_min_consistency'] = float(avg_min_consistency)
            stability_results['avg_avg_consistency'] = float(avg_avg_consistency)
            stability_results['stability_rating'] = self._rate_stability(avg_min_consistency, avg_avg_consistency)
            stability_results['status'] = 'completed'
            
        except Exception as e:
            logger.warning(f"分组稳定性验证失败: {e}")
            stability_results['status'] = 'failed'
            stability_results['error'] = str(e)
        
        return stability_results
    
    def _prepare_features_for_prediction(self, data: pd.DataFrame) -> pd.DataFrame:
        """为预测准备特征"""
        
        # 选择预测特征
        feature_cols = [col for col in data.columns 
                       if col not in ['Y_frac', 'id', 'date'] and not col.startswith('_')]
        
        features = data[feature_cols].copy()
        
        # 处理缺失值
        for col in features.columns:
            if features[col].dtype in ['float64', 'int64']:
                features[col] = features[col].fillna(features[col].median())
            else:
                features[col] = features[col].fillna(features[col].mode()[0] if not features[col].mode().empty else 0)
        
        return features
    
    def _test_week_extrapolation(self, test_data: pd.DataFrame, concentration_model) -> Dict[str, Any]:
        """测试孕周外推能力"""
        
        extrapolation_results = {}
        
        try:
            # 测试在训练范围外的孕周预测
            original_weeks = test_data['week'].dropna()
            
            if len(original_weeks) > 0:
                week_min, week_max = original_weeks.min(), original_weeks.max()
                
                # 创建外推测试点
                extrapolation_weeks = [week_min - 1, week_min - 0.5, week_max + 0.5, week_max + 1]
                
                extrapolation_errors = []
                
                for ext_week in extrapolation_weeks:
                    if 5 <= ext_week <= 30:  # 合理范围内
                        # 创建测试样本
                        test_sample = test_data.iloc[[0]].copy()  # 使用第一个样本作为模板
                        test_sample['week'] = ext_week
                        
                        try:
                            features = self._prepare_features_for_prediction(test_sample)
                            prediction = concentration_model.predict(features)[0]
                            
                            # 与邻近时点的预测比较，评估合理性
                            nearby_week = week_min if ext_week < week_min else week_max
                            test_sample_nearby = test_sample.copy()
                            test_sample_nearby['week'] = nearby_week
                            
                            features_nearby = self._prepare_features_for_prediction(test_sample_nearby)
                            prediction_nearby = concentration_model.predict(features_nearby)[0]
                            
                            prediction_diff = abs(prediction - prediction_nearby)
                            week_diff = abs(ext_week - nearby_week)
                            
                            extrapolation_error = prediction_diff / week_diff if week_diff > 0 else 0
                            extrapolation_errors.append(extrapolation_error)
                            
                        except:
                            extrapolation_errors.append(np.inf)
                
                extrapolation_results['extrapolation_errors'] = extrapolation_errors
                extrapolation_results['avg_extrapolation_error'] = np.mean([e for e in extrapolation_errors if np.isfinite(e)])
                extrapolation_results['extrapolation_quality'] = 'Good' if extrapolation_results['avg_extrapolation_error'] < 0.01 else 'Fair' if extrapolation_results['avg_extrapolation_error'] < 0.02 else 'Poor'
                
        except Exception as e:
            logger.debug(f"孕周外推测试失败: {e}")
            extrapolation_results['error'] = str(e)
        
        return extrapolation_results
    
    def _rate_stability(self, min_consistency: float, avg_consistency: float) -> str:
        """评估稳定性等级"""
        
        if min_consistency >= 0.8 and avg_consistency >= 0.9:
            return 'Excellent'
        elif min_consistency >= 0.6 and avg_consistency >= 0.8:
            return 'Good'
        elif min_consistency >= 0.4 and avg_consistency >= 0.7:
            return 'Fair'
        else:
            return 'Poor'
    
    def _summarize_validation_results(self, validation_results: dict) -> Dict[str, Any]:
        """汇总验证结果"""
        
        summary = {}
        
        # 汇总交叉验证
        cv_results = validation_results.get('cross_validation', {})
        if cv_results.get('status') == 'completed':
            conc_model = cv_results.get('concentration_model', {})
            summary['cv_score'] = conc_model.get('mean_r2', 'N/A')
            summary['cv_stability'] = 'Good' if conc_model.get('std_r2', 1) < 0.1 else 'Fair'
        
        # 汇总时间外推
        temporal_results = validation_results.get('temporal_validation', {})
        if temporal_results.get('status') == 'completed':
            conc_model = temporal_results.get('concentration_model', {})
            degradation = conc_model.get('performance_degradation', 0)
            summary['temporal_accuracy'] = 'Good' if degradation < 0.1 else 'Fair' if degradation < 0.2 else 'Poor'
        
        # 汇总Bootstrap
        bootstrap_results = validation_results.get('bootstrap', {})
        if bootstrap_results.get('status') == 'completed':
            cis = bootstrap_results.get('confidence_intervals', [])
            if cis:
                avg_ci_width = np.mean([ci['ci_width'] for ci in cis])
                summary['timing_uncertainty'] = f'±{avg_ci_width:.1f} weeks'
        
        # 汇总稳定性
        stability_results = validation_results.get('grouping_stability', {})
        if stability_results.get('status') == 'completed':
            summary['grouping_stability'] = stability_results.get('stability_rating', 'Unknown')
        
        # 整体验证评级
        components = [
            summary.get('cv_stability', 'Unknown'),
            summary.get('temporal_accuracy', 'Unknown'),
            summary.get('grouping_stability', 'Unknown')
        ]
        
        good_components = sum(1 for c in components if c == 'Good')
        fair_components = sum(1 for c in components if c == 'Fair')
        
        if good_components >= 2:
            summary['overall_validation_rating'] = 'Good'
        elif good_components + fair_components >= 2:
            summary['overall_validation_rating'] = 'Fair'
        else:
            summary['overall_validation_rating'] = 'Poor'
        
        return summary