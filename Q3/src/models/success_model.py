"""
测序成功概率模型
使用Logistic回归或随机森林预测测序成功率
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report
import logging

logger = logging.getLogger(__name__)


class SuccessModel:
    """测序成功概率模型类"""
    
    def __init__(self, model_type='logistic', **kwargs):
        self.model_type = model_type
        self.kwargs = kwargs
        self.model = None
        self.feature_columns = None
        self.is_fitted = False
        
    def _create_features(self, X):
        """创建特征"""
        features = X.copy()
        
        # 添加交互项
        if '孕周' in features.columns and 'BMI' in features.columns:
            features['孕周_BMI'] = features['孕周'].fillna(0) * features['BMI'].fillna(0)
        
        if '孕周' in features.columns and 'GC含量' in features.columns:
            features['孕周_GC'] = features['孕周'].fillna(0) * features['GC含量'].fillna(0)
        
        if 'BMI' in features.columns and 'GC含量' in features.columns:
            features['BMI_GC'] = features['BMI'].fillna(0) * features['GC含量'].fillna(0)
        
        # 添加多项式特征
        if '孕周' in features.columns:
            features['孕周_squared'] = features['孕周'].fillna(0) ** 2
        
        if 'BMI' in features.columns:
            features['BMI_squared'] = features['BMI'].fillna(0) ** 2
        
        # 添加技术指标特征
        if 'read_count' in features.columns:
            features['read_count_log'] = np.log1p(features['read_count'].fillna(0))
        
        if 'GC含量' in features.columns:
            features['GC_deviation'] = np.abs(features['GC含量'].fillna(0.5) - 0.5)
        
        return features
    
    def fit(self, X, y):
        """训练模型"""
        logger.info(f"训练测序成功概率模型，模型类型: {self.model_type}")
        
        self.feature_columns = X.columns.tolist()
        features = self._create_features(X)
        
        if self.model_type == 'logistic':
            self.model = LogisticRegression(
                random_state=42,
                max_iter=1000,
                **self.kwargs
            )
        elif self.model_type == 'random_forest':
            self.model = RandomForestClassifier(
                random_state=42,
                n_estimators=100,
                **self.kwargs
            )
        else:
            raise ValueError(f"不支持的模型类型: {self.model_type}")
        
        self.model.fit(features, y)
        self.is_fitted = True
        
        # 计算训练集性能
        train_score = self.score(features, y)
        logger.info(f"测序成功概率模型训练完成，训练集AUC: {train_score:.4f}")
        
        return self
    
    def predict_proba(self, X):
        """预测成功概率"""
        if not self.is_fitted:
            raise ValueError("模型尚未训练")
        
        features = self._create_features(X)
        return self.model.predict_proba(features)[:, 1]
    
    def predict(self, X):
        """预测成功概率"""
        return self.predict_proba(X)
    
    def score(self, X, y):
        """计算模型得分（AUC）"""
        if not self.is_fitted:
            raise ValueError("模型尚未训练")
        
        predictions = self.predict(X)
        return roc_auc_score(y, predictions)
    
    def get_feature_importance(self):
        """获取特征重要性"""
        if not self.is_fitted:
            raise ValueError("模型尚未训练")
        
        if hasattr(self.model, 'feature_importances_'):
            return dict(zip(self.feature_columns, self.model.feature_importances_))
        elif hasattr(self.model, 'coef_'):
            return dict(zip(self.feature_columns, np.abs(self.model.coef_[0])))
        else:
            return None
    
    def evaluate(self, X, y):
        """详细评估模型性能"""
        if not self.is_fitted:
            raise ValueError("模型尚未训练")
        
        predictions = self.predict(X)
        pred_classes = (predictions > 0.5).astype(int)
        
        auc = roc_auc_score(y, predictions)
        accuracy = accuracy_score(y, pred_classes)
        
        logger.info(f"模型评估结果:")
        logger.info(f"  AUC: {auc:.4f}")
        logger.info(f"  准确率: {accuracy:.4f}")
        
        # 打印分类报告
        report = classification_report(y, pred_classes, target_names=['失败', '成功'])
        logger.info(f"分类报告:\n{report}")
        
        return {
            'auc': auc,
            'accuracy': accuracy,
            'classification_report': report
        }
