"""
多因素特征工程模块
用于构建、选择和变换多因素特征
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, RobustScaler
from sklearn.feature_selection import (
    SelectKBest, f_regression, mutual_info_regression,
    RFE, SelectFromModel
)
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LassoCV
from sklearn.model_selection import cross_val_score
from scipy import stats
from typing import List, Dict, Tuple, Optional, Any
import logging
import warnings

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


class MultiFactorFeatureEngineer:
    """多因素特征工程器"""
    
    def __init__(self, config: dict):
        """
        初始化特征工程器
        
        Parameters:
        -----------
        config : dict
            特征工程配置
        """
        self.config = config
        self.base_features = config.get('base_features', ['week', 'BMI_used'])
        self.additional_features = config.get('additional_features', [])
        self.interaction_terms = config.get('interaction_terms', True)
        self.polynomial_degree = config.get('polynomial_degree', 2)
        
        # 组件初始化
        self.scaler = RobustScaler()
        self.poly_features = None
        self.feature_selector = None
        self.selected_features_ = None
        self.feature_importance_ = None
        self.is_fitted = False
        
        logger.info(f"特征工程器初始化完成")
        logger.info(f"  - 基础特征: {self.base_features}")
        logger.info(f"  - 额外特征: {self.additional_features}")
        logger.info(f"  - 交互项: {self.interaction_terms}")
        logger.info(f"  - 多项式次数: {self.polynomial_degree}")
    
    def fit_transform(self, df: pd.DataFrame, target_col: str = 'Y_frac') -> pd.DataFrame:
        """
        拟合并变换数据
        
        Parameters:
        -----------
        df : pd.DataFrame
            输入数据
        target_col : str
            目标列名
            
        Returns:
        --------
        pd.DataFrame
            变换后的数据
        """
        logger.info("开始特征工程...")
        
        # 1. 构建基础特征集
        feature_df = self._build_base_features(df)
        
        # 2. 添加多因素特征
        feature_df = self._add_multi_factor_features(feature_df, df)
        
        # 3. 生成交互特征
        if self.interaction_terms:
            feature_df = self._generate_interaction_features(feature_df)
        
        # 4. 生成多项式特征
        feature_df = self._generate_polynomial_features(feature_df)
        
        # 5. 特征缩放
        feature_df = self._scale_features(feature_df)
        
        # 6. 特征选择（如果启用）
        if self.config.get('feature_selection', {}).get('enabled', True):
            feature_df = self._select_features(feature_df, df[target_col])
        
        # 7. 添加元数据
        feature_df = self._add_metadata(feature_df, df)
        
        self.is_fitted = True
        logger.info(f"特征工程完成，最终特征数: {len(feature_df.columns)}")
        
        return feature_df
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        变换新数据（必须先调用fit_transform）
        
        Parameters:
        -----------
        df : pd.DataFrame
            输入数据
            
        Returns:
        --------
        pd.DataFrame
            变换后的数据
        """
        if not self.is_fitted:
            raise ValueError("必须先调用fit_transform")
        
        # 应用相同的变换流程
        feature_df = self._build_base_features(df)
        feature_df = self._add_multi_factor_features(feature_df, df)
        
        if self.interaction_terms:
            feature_df = self._generate_interaction_features(feature_df)
        
        feature_df = self._generate_polynomial_features(feature_df)
        feature_df = self._scale_features(feature_df)
        
        # 应用相同的特征选择
        if self.selected_features_ is not None:
            feature_df = feature_df[self.selected_features_]
        
        feature_df = self._add_metadata(feature_df, df)
        
        return feature_df
    
    def _build_base_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """构建基础特征"""
        feature_df = df[self.base_features].copy()
        
        # 添加基础变换
        if 'week' in feature_df.columns:
            feature_df['week_squared'] = feature_df['week'] ** 2
            feature_df['week_log'] = np.log(feature_df['week'] + 1)
        
        if 'BMI_used' in feature_df.columns:
            feature_df['BMI_squared'] = feature_df['BMI_used'] ** 2
            feature_df['BMI_log'] = np.log(feature_df['BMI_used'])
            
            # BMI分类特征
            feature_df['BMI_underweight'] = (feature_df['BMI_used'] < 18.5).astype(int)
            feature_df['BMI_normal'] = ((feature_df['BMI_used'] >= 18.5) & 
                                     (feature_df['BMI_used'] < 25)).astype(int)
            feature_df['BMI_overweight'] = ((feature_df['BMI_used'] >= 25) & 
                                          (feature_df['BMI_used'] < 30)).astype(int)
            feature_df['BMI_obese'] = (feature_df['BMI_used'] >= 30).astype(int)
        
        return feature_df
    
    def _add_multi_factor_features(self, feature_df: pd.DataFrame, 
                                 original_df: pd.DataFrame) -> pd.DataFrame:
        """添加多因素特征"""
        
        for feature in self.additional_features:
            if feature in original_df.columns:
                values = original_df[feature].fillna(original_df[feature].median())
                feature_df[feature] = values
                
                # 添加变换
                if feature in ['age', 'height', 'weight']:
                    feature_df[f'{feature}_squared'] = values ** 2
                    feature_df[f'{feature}_log'] = np.log(values + 1)
                    
                    # 分位数特征
                    quartiles = values.quantile([0.25, 0.5, 0.75])
                    feature_df[f'{feature}_q1'] = (values <= quartiles[0.25]).astype(int)
                    feature_df[f'{feature}_q2'] = ((values > quartiles[0.25]) & 
                                                 (values <= quartiles[0.5])).astype(int)
                    feature_df[f'{feature}_q3'] = ((values > quartiles[0.5]) & 
                                                 (values <= quartiles[0.75])).astype(int)
                    feature_df[f'{feature}_q4'] = (values > quartiles[0.75]).astype(int)
                
                elif feature in ['gc_percent', 'readcount']:
                    feature_df[f'{feature}_log'] = np.log(values + 1)
                    
                    # 异常值指示器
                    q1, q3 = values.quantile([0.25, 0.75])
                    iqr = q3 - q1
                    lower_bound = q1 - 1.5 * iqr
                    upper_bound = q3 + 1.5 * iqr
                    
                    feature_df[f'{feature}_outlier'] = (
                        (values < lower_bound) | (values > upper_bound)
                    ).astype(int)
        
        # 添加复合特征
        feature_df = self._add_composite_features(feature_df)
        
        return feature_df
    
    def _add_composite_features(self, feature_df: pd.DataFrame) -> pd.DataFrame:
        """添加复合特征"""
        
        # BMI与年龄的复合特征
        if 'BMI_used' in feature_df.columns and 'age' in feature_df.columns:
            feature_df['BMI_age_ratio'] = feature_df['BMI_used'] / (feature_df['age'] + 1)
            feature_df['BMI_age_product'] = feature_df['BMI_used'] * feature_df['age']
        
        # 身高体重比
        if 'height' in feature_df.columns and 'weight' in feature_df.columns:
            feature_df['height_weight_ratio'] = feature_df['height'] / (feature_df['weight'] + 1)
            feature_df['height_weight_product'] = feature_df['height'] * feature_df['weight']
        
        # 孕周与多因素的复合特征
        if 'week' in feature_df.columns:
            for factor in ['age', 'height', 'weight']:
                if factor in feature_df.columns:
                    feature_df[f'week_{factor}_ratio'] = feature_df['week'] / (feature_df[factor] + 1)
                    feature_df[f'week_{factor}_product'] = feature_df['week'] * feature_df[factor]
        
        # 技术因素复合特征
        if 'gc_percent' in feature_df.columns and 'readcount' in feature_df.columns:
            feature_df['gc_readcount_ratio'] = (feature_df['gc_percent'] / 
                                              (feature_df['readcount'] / 1e6 + 1))
            feature_df['gc_readcount_product'] = (feature_df['gc_percent'] * 
                                                feature_df['readcount'] / 1e6)
        
        return feature_df
    
    def _generate_interaction_features(self, feature_df: pd.DataFrame) -> pd.DataFrame:
        """生成交互特征"""
        
        # 选择核心特征进行交互
        core_features = []
        for col in feature_df.columns:
            if any(base in col for base in self.base_features + self.additional_features[:3]):
                if not any(suffix in col for suffix in ['_q1', '_q2', '_q3', '_q4', '_outlier']):
                    core_features.append(col)
        
        core_features = core_features[:8]  # 限制交互特征数量
        
        if len(core_features) >= 2:
            logger.info(f"生成交互特征，核心特征: {core_features[:5]}...")
            
            interaction_df = pd.DataFrame(index=feature_df.index)
            
            # 两两交互
            for i, feat1 in enumerate(core_features):
                for feat2 in core_features[i+1:]:
                    interaction_name = f'{feat1}_x_{feat2}'
                    interaction_df[interaction_name] = feature_df[feat1] * feature_df[feat2]
            
            # 合并交互特征
            feature_df = pd.concat([feature_df, interaction_df], axis=1)
        
        return feature_df
    
    def _generate_polynomial_features(self, feature_df: pd.DataFrame) -> pd.DataFrame:
        """生成多项式特征"""
        
        if self.polynomial_degree <= 1:
            return feature_df
        
        # 选择数值特征进行多项式扩展
        numeric_features = feature_df.select_dtypes(include=[np.number]).columns.tolist()
        
        # 限制特征数量避免维度爆炸
        if len(numeric_features) > 10:
            # 使用方差选择top特征
            variances = feature_df[numeric_features].var()
            numeric_features = variances.nlargest(10).index.tolist()
        
        if self.poly_features is None:
            self.poly_features = PolynomialFeatures(
                degree=self.polynomial_degree,
                interaction_only=False,
                include_bias=False
            )
        
        logger.info(f"生成{self.polynomial_degree}次多项式特征，输入特征: {len(numeric_features)}")
        
        # 应用多项式变换
        poly_array = self.poly_features.fit_transform(feature_df[numeric_features].fillna(0))
        poly_feature_names = self.poly_features.get_feature_names_out(numeric_features)
        
        # 创建多项式特征DataFrame
        poly_df = pd.DataFrame(
            poly_array, 
            index=feature_df.index, 
            columns=poly_feature_names
        )
        
        # 移除原始特征避免重复
        poly_df = poly_df.drop(columns=numeric_features, errors='ignore')
        
        # 合并特征
        result_df = pd.concat([feature_df, poly_df], axis=1)
        
        logger.info(f"多项式特征生成完成，新增特征: {len(poly_df.columns)}")
        
        return result_df
    
    def _scale_features(self, feature_df: pd.DataFrame) -> pd.DataFrame:
        """特征缩放"""
        
        numeric_columns = feature_df.select_dtypes(include=[np.number]).columns
        non_numeric_columns = feature_df.select_dtypes(exclude=[np.number]).columns
        
        if len(numeric_columns) > 0:
            # 填补缺失值
            numeric_data = feature_df[numeric_columns].fillna(feature_df[numeric_columns].median())
            
            # 缩放数值特征
            scaled_data = self.scaler.fit_transform(numeric_data)
            scaled_df = pd.DataFrame(
                scaled_data, 
                index=feature_df.index, 
                columns=numeric_columns
            )
            
            # 合并非数值特征
            if len(non_numeric_columns) > 0:
                result_df = pd.concat([scaled_df, feature_df[non_numeric_columns]], axis=1)
            else:
                result_df = scaled_df
        else:
            result_df = feature_df
        
        return result_df
    
    def _select_features(self, feature_df: pd.DataFrame, target: pd.Series) -> pd.DataFrame:
        """特征选择"""
        
        feature_selection_config = self.config.get('feature_selection', {})
        method = feature_selection_config.get('method', 'mutual_info')
        n_features = feature_selection_config.get('n_features', 15)
        
        logger.info(f"特征选择: 方法={method}, 目标特征数={n_features}")
        
        # 准备数据
        X = feature_df.fillna(0)
        y = target.fillna(target.median())
        
        # 确保数据对齐
        common_idx = X.index.intersection(y.index)
        X = X.loc[common_idx]
        y = y.loc[common_idx]
        
        if method == 'mutual_info':
            selector = SelectKBest(
                score_func=mutual_info_regression, 
                k=min(n_features, len(X.columns))
            )
        elif method == 'f_regression':
            selector = SelectKBest(
                score_func=f_regression, 
                k=min(n_features, len(X.columns))
            )
        elif method == 'recursive':
            base_estimator = RandomForestRegressor(n_estimators=50, random_state=42)
            selector = RFE(
                estimator=base_estimator, 
                n_features_to_select=min(n_features, len(X.columns))
            )
        elif method == 'lasso':
            lasso = LassoCV(cv=3, random_state=42)
            selector = SelectFromModel(lasso, max_features=n_features)
        else:
            logger.warning(f"未知的特征选择方法: {method}，跳过特征选择")
            return feature_df
        
        try:
            # 执行特征选择
            X_selected = selector.fit_transform(X, y)
            
            # 获取选择的特征名
            if hasattr(selector, 'get_support'):
                selected_mask = selector.get_support()
                selected_features = X.columns[selected_mask].tolist()
            else:
                selected_features = X.columns[:n_features].tolist()
            
            self.selected_features_ = selected_features
            self.feature_selector = selector
            
            # 计算特征重要性
            if hasattr(selector, 'scores_'):
                feature_scores = selector.scores_
                self.feature_importance_ = dict(zip(X.columns, feature_scores))
            elif hasattr(selector, 'estimator_') and hasattr(selector.estimator_, 'feature_importances_'):
                feature_importance = selector.estimator_.feature_importances_
                self.feature_importance_ = dict(zip(selected_features, feature_importance))
            
            logger.info(f"特征选择完成，保留特征: {len(selected_features)}")
            logger.info(f"选择的特征: {selected_features[:5]}...")
            
            return feature_df[selected_features]
            
        except Exception as e:
            logger.error(f"特征选择失败: {e}")
            return feature_df.iloc[:, :n_features]  # 回退策略
    
    def _add_metadata(self, feature_df: pd.DataFrame, original_df: pd.DataFrame) -> pd.DataFrame:
        """添加元数据"""
        
        # 添加ID列用于追踪
        if 'id' in original_df.columns:
            feature_df['id'] = original_df['id']
        
        # 添加目标变量
        if 'Y_frac' in original_df.columns:
            feature_df['Y_frac'] = original_df['Y_frac']
        
        # 添加其他重要元数据
        metadata_cols = ['week', 'BMI_used', 'date']
        for col in metadata_cols:
            if col in original_df.columns and col not in feature_df.columns:
                feature_df[col] = original_df[col]
        
        return feature_df
    
    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """获取特征重要性"""
        return self.feature_importance_
    
    def get_selected_features(self) -> Optional[List[str]]:
        """获取选择的特征"""
        return self.selected_features_
    
    def get_feature_stats(self) -> Dict[str, Any]:
        """获取特征统计信息"""
        stats = {
            'n_selected_features': len(self.selected_features_) if self.selected_features_ else 0,
            'selected_features': self.selected_features_,
            'feature_importance': self.feature_importance_,
            'is_fitted': self.is_fitted
        }
        return stats


class AdvancedFeatureEngineer(MultiFactorFeatureEngineer):
    """高级特征工程器，包含更多复杂特征"""
    
    def __init__(self, config: dict):
        super().__init__(config)
        self.time_features_enabled = config.get('time_features', True)
        self.medical_features_enabled = config.get('medical_features', True)
    
    def _add_time_based_features(self, feature_df: pd.DataFrame, 
                                original_df: pd.DataFrame) -> pd.DataFrame:
        """添加基于时间的特征"""
        
        if not self.time_features_enabled or 'week' not in feature_df.columns:
            return feature_df
        
        week = feature_df['week']
        
        # 妊娠期划分
        feature_df['trimester_1'] = (week <= 12).astype(int)
        feature_df['trimester_2'] = ((week > 12) & (week <= 27)).astype(int)
        feature_df['trimester_3'] = (week > 27).astype(int)
        
        # 关键时间点
        feature_df['early_pregnancy'] = (week <= 10).astype(int)
        feature_df['mid_pregnancy'] = ((week > 10) & (week <= 20)).astype(int)
        feature_df['late_pregnancy'] = (week > 20).astype(int)
        
        # 时间趋势特征
        feature_df['week_normalized'] = (week - week.min()) / (week.max() - week.min())
        feature_df['week_centered'] = week - week.mean()
        
        return feature_df
    
    def _add_medical_features(self, feature_df: pd.DataFrame, 
                            original_df: pd.DataFrame) -> pd.DataFrame:
        """添加医学相关特征"""
        
        if not self.medical_features_enabled:
            return feature_df
        
        # BMI相关医学特征
        if 'BMI_used' in feature_df.columns:
            bmi = feature_df['BMI_used']
            
            # WHO BMI分类
            feature_df['bmi_who_underweight'] = (bmi < 18.5).astype(int)
            feature_df['bmi_who_normal'] = ((bmi >= 18.5) & (bmi < 25)).astype(int)
            feature_df['bmi_who_overweight'] = ((bmi >= 25) & (bmi < 30)).astype(int)
            feature_df['bmi_who_obese_1'] = ((bmi >= 30) & (bmi < 35)).astype(int)
            feature_df['bmi_who_obese_2'] = ((bmi >= 35) & (bmi < 40)).astype(int)
            feature_df['bmi_who_obese_3'] = (bmi >= 40).astype(int)
            
            # 妊娠期BMI风险
            feature_df['bmi_pregnancy_risk'] = np.where(
                bmi < 18.5, 1,  # 低体重风险
                np.where(bmi > 30, 2, 0)  # 肥胖风险
            )
        
        # 年龄相关医学特征
        if 'age' in feature_df.columns:
            age = feature_df['age']
            
            # 生育年龄分组
            feature_df['age_optimal'] = ((age >= 20) & (age <= 35)).astype(int)
            feature_df['age_advanced'] = (age > 35).astype(int)
            feature_df['age_very_young'] = (age < 20).astype(int)
            
            # 年龄风险评分
            feature_df['age_risk_score'] = np.where(
                age < 20, 1,
                np.where(age > 35, (age - 35) * 0.1 + 1, 0)
            )
        
        return feature_df


# 工具函数
def calculate_feature_correlation(df: pd.DataFrame, target_col: str = 'Y_frac') -> pd.Series:
    """计算特征与目标变量的相关性"""
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    correlations = {}
    
    for col in numeric_cols:
        if col != target_col:
            try:
                corr = df[col].corr(df[target_col])
                if not np.isnan(corr):
                    correlations[col] = abs(corr)
            except:
                continue
    
    return pd.Series(correlations).sort_values(ascending=False)


def identify_redundant_features(df: pd.DataFrame, threshold: float = 0.95) -> List[str]:
    """识别冗余特征"""
    
    numeric_df = df.select_dtypes(include=[np.number])
    corr_matrix = numeric_df.corr().abs()
    
    # 找到高相关的特征对
    redundant_features = set()
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            if corr_matrix.iloc[i, j] > threshold:
                # 保留方差更大的特征
                var_i = numeric_df.iloc[:, i].var()
                var_j = numeric_df.iloc[:, j].var()
                
                if var_i > var_j:
                    redundant_features.add(corr_matrix.columns[j])
                else:
                    redundant_features.add(corr_matrix.columns[i])
    
    return list(redundant_features)


def generate_feature_report(feature_engineer: MultiFactorFeatureEngineer, 
                          df: pd.DataFrame) -> Dict[str, Any]:
    """生成特征工程报告"""
    
    report = {
        'feature_engineering_summary': {
            'total_features': len(df.columns),
            'selected_features': len(feature_engineer.get_selected_features() or []),
            'feature_reduction_ratio': (len(feature_engineer.get_selected_features() or []) / 
                                      len(df.columns)) if len(df.columns) > 0 else 0
        },
        'feature_importance': feature_engineer.get_feature_importance(),
        'selected_features': feature_engineer.get_selected_features(),
        'data_quality': {
            'missing_values': df.isnull().sum().to_dict(),
            'data_types': df.dtypes.astype(str).to_dict(),
            'numerical_features': len(df.select_dtypes(include=[np.number]).columns),
            'categorical_features': len(df.select_dtypes(exclude=[np.number]).columns)
        }
    }
    
    return report