"""
达标概率模型
使用GAM/GLMM拟合连续Y%并计算达标概率
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
import logging

logger = logging.getLogger(__name__)


class AttainModel(BaseEstimator, RegressorMixin):
    """达标概率模型类"""
    
    def __init__(self, model_type='logistic', **kwargs):
        self.model_type = model_type
        self.kwargs = kwargs
        self.model = None
        self.feature_columns = None
        self.is_fitted = False
        self.feature_importance_ = None
        
    def _create_features(self, X):
        """创建特征"""
        features = X.copy()
        
        # 添加交互项
        if '孕周' in features.columns and 'BMI' in features.columns:
            features['孕周_BMI'] = features['孕周'] * features['BMI']
        
        if '孕周' in features.columns and 'GC含量' in features.columns:
            features['孕周_GC'] = features['孕周'] * features['GC含量']
        
        # 添加多项式特征
        if '孕周' in features.columns:
            features['孕周_squared'] = features['孕周'] ** 2
            features['孕周_cubed'] = features['孕周'] ** 3
        
        if 'BMI' in features.columns:
            features['BMI_squared'] = features['BMI'] ** 2
        
        return features
    
    def fit(self, X, y):
        """训练模型"""
        logger.info(f"训练达标概率模型，模型类型: {self.model_type}")
        
        self.feature_columns = X.columns.tolist()
        features = self._create_features(X)
        
        if self.model_type == 'logistic':
            # 使用逻辑回归
            self.model = LogisticRegression(
                random_state=42,
                max_iter=1000,
                **self.kwargs
            )
            self.model.fit(features, y)
            
        elif self.model_type == 'random_forest':
            # 使用随机森林
            self.model = RandomForestRegressor(
                random_state=42,
                n_estimators=100,
                **self.kwargs
            )
            self.model.fit(features, y)
            
        else:
            raise ValueError(f"不支持的模型类型: {self.model_type}")
        
        self.is_fitted = True
        
        # 计算特征重要性
        if hasattr(self.model, 'feature_importances_'):
            self.feature_importance_ = dict(zip(features.columns, self.model.feature_importances_))
        elif hasattr(self.model, 'coef_'):
            self.feature_importance_ = dict(zip(features.columns, np.abs(self.model.coef_[0])))
        
        logger.info("达标概率模型训练完成")
        return self
    
    def predict_proba(self, X):
        """预测达标概率"""
        if not self.is_fitted:
            raise ValueError("模型尚未训练")
        
        features = self._create_features(X)
        
        if self.model_type == 'logistic':
            return self.model.predict_proba(features)[:, 1]
        elif self.model_type == 'random_forest':
            # 将回归结果转换为概率
            predictions = self.model.predict(features)
            return np.clip(predictions, 0, 1)
    
    def predict(self, X):
        """预测达标概率"""
        return self.predict_proba(X)
    
    def score(self, X, y):
        """计算模型得分"""
        if not self.is_fitted:
            raise ValueError("模型尚未训练")
        
        predictions = self.predict(X)
        
        if self.model_type == 'logistic':
            from sklearn.metrics import roc_auc_score
            return roc_auc_score(y, predictions)
        else:
            return r2_score(y, predictions)


class YPercentModel:
    """Y染色体浓度连续值模型"""
    
    def __init__(self, model_type='random_forest', **kwargs):
        self.model_type = model_type
        self.kwargs = kwargs
        self.model = None
        self.feature_columns = None
        self.is_fitted = False
        self.residual_std = None
        
    def _create_features(self, X):
        """创建特征"""
        features = X.copy()
        
        # 添加交互项
        if '孕周' in features.columns and 'BMI' in features.columns:
            features['孕周_BMI'] = features['孕周'] * features['BMI']
        
        if '孕周' in features.columns and 'GC含量' in features.columns:
            features['孕周_GC'] = features['孕周'] * features['GC含量']
        
        # 添加多项式特征
        if '孕周' in features.columns:
            features['孕周_squared'] = features['孕周'] ** 2
            features['孕周_cubed'] = features['孕周'] ** 3
        
        if 'BMI' in features.columns:
            features['BMI_squared'] = features['BMI'] ** 2
        
        return features
    
    def fit(self, X, y):
        """训练模型"""
        logger.info(f"训练Y染色体浓度模型，模型类型: {self.model_type}")
        
        self.feature_columns = X.columns.tolist()
        features = self._create_features(X)
        
        if self.model_type == 'random_forest':
            self.model = RandomForestRegressor(
                random_state=42,
                n_estimators=100,
                **self.kwargs
            )
        else:
            raise ValueError(f"不支持的模型类型: {self.model_type}")
        
        self.model.fit(features, y)
        
        # 计算残差标准差
        predictions = self.model.predict(features)
        residuals = y - predictions
        self.residual_std = np.std(residuals)
        
        self.is_fitted = True
        logger.info(f"Y染色体浓度模型训练完成，残差标准差: {self.residual_std:.4f}")
        return self
    
    def predict(self, X):
        """预测Y染色体浓度"""
        if not self.is_fitted:
            raise ValueError("模型尚未训练")
        
        features = self._create_features(X)
        return self.model.predict(features)
    
    def predict_attain_prob(self, X, threshold=0.04):
        """预测达标概率"""
        if not self.is_fitted:
            raise ValueError("模型尚未训练")
        
        # 预测均值
        mean_pred = self.predict(X)
        
        # 计算达标概率
        from scipy.stats import norm
        z_scores = (threshold - mean_pred) / self.residual_std
        attain_probs = 1 - norm.cdf(z_scores)
        
        return np.clip(attain_probs, 0, 1)
    
    def score(self, X, y):
        """计算模型得分"""
        if not self.is_fitted:
            raise ValueError("模型尚未训练")
        
        predictions = self.predict(X)
        return r2_score(y, predictions)
