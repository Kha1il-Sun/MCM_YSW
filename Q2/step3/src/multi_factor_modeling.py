"""
多因素建模模块
实现综合考虑多种因素的Y染色体浓度预测和成功率预测模型
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import (
    RandomForestRegressor, GradientBoostingRegressor, 
    VotingRegressor, StackingRegressor,
    RandomForestClassifier, GradientBoostingClassifier,
    VotingClassifier, StackingClassifier
)
from sklearn.linear_model import LinearRegression, LogisticRegression, LassoCV, RidgeCV
from sklearn.svm import SVR, SVC
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.model_selection import (
    cross_val_score, GridSearchCV, RandomizedSearchCV,
    train_test_split, GroupKFold
)
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score
)
from sklearn.preprocessing import StandardScaler
from sklearn.isotonic import IsotonicRegression
from scipy import stats
from scipy.optimize import minimize_scalar
import xgboost as xgb
import lightgbm as lgb
from typing import Dict, List, Tuple, Optional, Any, Callable
import logging
import warnings
import joblib

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


class MultiFactorYConcentrationModel:
    """多因素Y染色体浓度预测模型"""
    
    def __init__(self, config: dict):
        """
        初始化模型
        
        Parameters:
        -----------
        config : dict
            模型配置
        """
        self.config = config
        self.quantile_tau = config.get('quantile_tau', 0.9)
        self.ensemble_method = config.get('ensemble_method', 'stacking')
        self.base_models = config.get('base_models', ['xgboost', 'gam', 'rf'])
        
        # 模型组件
        self.models = {}
        self.ensemble_model = None
        self.scaler = StandardScaler()
        self.is_fitted = False
        self.feature_importance_ = None
        self.performance_metrics_ = {}
        
        logger.info(f"Y染色体浓度模型初始化完成")
        logger.info(f"  - 分位数水平: {self.quantile_tau}")
        logger.info(f"  - 集成方法: {self.ensemble_method}")
        logger.info(f"  - 基模型: {self.base_models}")
    
    def fit(self, df: pd.DataFrame, target_col: str = 'Y_frac'):
        """
        训练模型
        
        Parameters:
        -----------
        df : pd.DataFrame
            训练数据
        target_col : str
            目标列名
        """
        logger.info("开始训练Y染色体浓度预测模型...")
        
        # 准备数据
        X, y = self._prepare_data(df, target_col)
        
        # 训练基模型
        self._fit_base_models(X, y)
        
        # 构建集成模型
        self._build_ensemble_model(X, y)
        
        # 计算性能指标
        self._evaluate_model(X, y)
        
        # 计算特征重要性
        self._compute_feature_importance(X.columns)
        
        self.is_fitted = True
        logger.info("Y染色体浓度模型训练完成")
        
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        预测Y染色体浓度
        
        Parameters:
        -----------
        X : pd.DataFrame
            输入特征
            
        Returns:
        --------
        np.ndarray
            预测的Y染色体浓度
        """
        if not self.is_fitted:
            raise ValueError("模型尚未训练")
        
        X_processed = self._prepare_features(X)
        
        if self.ensemble_model is not None:
            return self.ensemble_model.predict(X_processed)
        else:
            # 如果集成模型不可用，使用最佳单模型
            best_model = self._get_best_model()
            return best_model.predict(X_processed)
    
    def predict_quantile(self, X: pd.DataFrame, quantile: float = None) -> np.ndarray:
        """
        预测指定分位数的Y染色体浓度
        
        Parameters:
        -----------
        X : pd.DataFrame
            输入特征
        quantile : float, optional
            分位数水平，默认使用初始化时的tau
            
        Returns:
        --------
        np.ndarray
            分位数预测
        """
        if quantile is None:
            quantile = self.quantile_tau
        
        # 使用分位数回归模型
        if 'quantile_regression' in self.models:
            return self.models['quantile_regression'].predict(X)
        else:
            # 回退到标准预测
            mean_pred = self.predict(X)
            std_pred = self._predict_std(X)
            
            # 使用正态分布假设
            quantile_pred = mean_pred + stats.norm.ppf(quantile) * std_pred
            return quantile_pred
    
    def predict_with_uncertainty(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        预测Y染色体浓度和不确定性
        
        Parameters:
        -----------
        X : pd.DataFrame
            输入特征
            
        Returns:
        --------
        tuple
            (预测值, 不确定性)
        """
        mean_pred = self.predict(X)
        std_pred = self._predict_std(X)
        
        return mean_pred, std_pred
    
    def _prepare_data(self, df: pd.DataFrame, target_col: str) -> Tuple[pd.DataFrame, pd.Series]:
        """准备训练数据"""
        
        # 选择特征列
        feature_cols = [col for col in df.columns 
                       if col not in [target_col, 'id', 'date'] and 
                       not col.startswith('_')]
        
        X = df[feature_cols].copy()
        y = df[target_col].copy()
        
        # 处理缺失值
        X = X.fillna(X.median())
        y = y.fillna(y.median())
        
        # 确保数据对齐
        common_idx = X.index.intersection(y.index)
        X = X.loc[common_idx]
        y = y.loc[common_idx]
        
        logger.info(f"数据准备完成: {len(X)} 样本, {len(X.columns)} 特征")
        
        return X, y
    
    def _prepare_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """准备预测特征"""
        
        X_processed = X.copy()
        
        # 处理缺失值
        for col in X_processed.columns:
            if X_processed[col].dtype in ['float64', 'int64']:
                X_processed[col] = X_processed[col].fillna(X_processed[col].median())
            else:
                X_processed[col] = X_processed[col].fillna(X_processed[col].mode()[0] 
                                                          if not X_processed[col].mode().empty 
                                                          else 0)
        
        return X_processed
    
    def _fit_base_models(self, X: pd.DataFrame, y: pd.Series):
        """训练基模型"""
        
        logger.info("训练基模型...")
        
        # XGBoost模型
        if 'xgboost' in self.base_models:
            self.models['xgboost'] = self._fit_xgboost(X, y)
        
        # LightGBM模型
        if 'lightgbm' in self.base_models:
            self.models['lightgbm'] = self._fit_lightgbm(X, y)
        
        # 随机森林模型
        if 'rf' in self.base_models:
            self.models['rf'] = self._fit_random_forest(X, y)
        
        # 梯度提升模型
        if 'gbr' in self.base_models:
            self.models['gbr'] = self._fit_gradient_boosting(X, y)
        
        # 神经网络模型
        if 'mlp' in self.base_models:
            self.models['mlp'] = self._fit_neural_network(X, y)
        
        # SVR模型
        if 'svr' in self.base_models:
            self.models['svr'] = self._fit_svr(X, y)
        
        # 分位数回归模型
        if 'quantile_regression' in self.base_models:
            self.models['quantile_regression'] = self._fit_quantile_regression(X, y)
        
        logger.info(f"基模型训练完成，共{len(self.models)}个模型")
    
    def _fit_xgboost(self, X: pd.DataFrame, y: pd.Series):
        """训练XGBoost模型"""
        
        params = {
            'n_estimators': 200,
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'n_jobs': -1
        }
        
        model = xgb.XGBRegressor(**params)
        
        # 超参数调优
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [4, 6, 8],
            'learning_rate': [0.05, 0.1, 0.15]
        }
        
        grid_search = GridSearchCV(
            model, param_grid, cv=5, scoring='neg_mean_squared_error',
            n_jobs=-1, verbose=0
        )
        
        grid_search.fit(X, y)
        
        logger.info(f"XGBoost最佳参数: {grid_search.best_params_}")
        return grid_search.best_estimator_
    
    def _fit_lightgbm(self, X: pd.DataFrame, y: pd.Series):
        """训练LightGBM模型"""
        
        params = {
            'n_estimators': 200,
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'n_jobs': -1,
            'verbose': -1
        }
        
        model = lgb.LGBMRegressor(**params)
        model.fit(X, y)
        
        return model
    
    def _fit_random_forest(self, X: pd.DataFrame, y: pd.Series):
        """训练随机森林模型"""
        
        params = {
            'n_estimators': 200,
            'max_depth': 10,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'random_state': 42,
            'n_jobs': -1
        }
        
        model = RandomForestRegressor(**params)
        model.fit(X, y)
        
        return model
    
    def _fit_gradient_boosting(self, X: pd.DataFrame, y: pd.Series):
        """训练梯度提升模型"""
        
        params = {
            'n_estimators': 200,
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'random_state': 42
        }
        
        model = GradientBoostingRegressor(**params)
        model.fit(X, y)
        
        return model
    
    def _fit_neural_network(self, X: pd.DataFrame, y: pd.Series):
        """训练神经网络模型"""
        
        # 标准化数据
        X_scaled = self.scaler.fit_transform(X)
        
        params = {
            'hidden_layer_sizes': (100, 50),
            'activation': 'relu',
            'solver': 'adam',
            'alpha': 0.01,
            'learning_rate': 'adaptive',
            'max_iter': 500,
            'random_state': 42
        }
        
        model = MLPRegressor(**params)
        model.fit(X_scaled, y)
        
        return model
    
    def _fit_svr(self, X: pd.DataFrame, y: pd.Series):
        """训练SVR模型"""
        
        # 标准化数据
        X_scaled = self.scaler.fit_transform(X)
        
        params = {
            'kernel': 'rbf',
            'C': 1.0,
            'gamma': 'scale',
            'epsilon': 0.1
        }
        
        model = SVR(**params)
        model.fit(X_scaled, y)
        
        return model
    
    def _fit_quantile_regression(self, X: pd.DataFrame, y: pd.Series):
        """训练分位数回归模型"""
        
        # 使用XGBoost实现分位数回归
        params = {
            'n_estimators': 200,
            'max_depth': 6,
            'learning_rate': 0.1,
            'objective': f'reg:quantileerror',
            'quantile_alpha': self.quantile_tau,
            'random_state': 42
        }
        
        model = xgb.XGBRegressor(**params)
        model.fit(X, y)
        
        return model
    
    def _build_ensemble_model(self, X: pd.DataFrame, y: pd.Series):
        """构建集成模型"""
        
        if self.ensemble_method == 'voting':
            self._build_voting_ensemble(X, y)
        elif self.ensemble_method == 'stacking':
            self._build_stacking_ensemble(X, y)
        elif self.ensemble_method == 'bagging':
            self._build_bagging_ensemble(X, y)
        else:
            logger.warning(f"未知的集成方法: {self.ensemble_method}")
    
    def _build_voting_ensemble(self, X: pd.DataFrame, y: pd.Series):
        """构建投票集成模型"""
        
        estimators = [(name, model) for name, model in self.models.items() 
                     if hasattr(model, 'predict')]
        
        if len(estimators) >= 2:
            self.ensemble_model = VotingRegressor(estimators=estimators)
            self.ensemble_model.fit(X, y)
            logger.info(f"投票集成模型构建完成，包含{len(estimators)}个基模型")
    
    def _build_stacking_ensemble(self, X: pd.DataFrame, y: pd.Series):
        """构建堆叠集成模型"""
        
        estimators = [(name, model) for name, model in self.models.items() 
                     if hasattr(model, 'predict')]
        
        if len(estimators) >= 2:
            # 使用线性回归作为元学习器
            meta_learner = LinearRegression()
            
            self.ensemble_model = StackingRegressor(
                estimators=estimators,
                final_estimator=meta_learner,
                cv=5,
                n_jobs=-1
            )
            
            self.ensemble_model.fit(X, y)
            logger.info(f"堆叠集成模型构建完成，包含{len(estimators)}个基模型")
    
    def _build_bagging_ensemble(self, X: pd.DataFrame, y: pd.Series):
        """构建Bagging集成模型"""
        
        # 使用最佳单模型进行Bagging
        best_model = self._get_best_model()
        
        from sklearn.ensemble import BaggingRegressor
        
        self.ensemble_model = BaggingRegressor(
            base_estimator=best_model,
            n_estimators=10,
            random_state=42,
            n_jobs=-1
        )
        
        self.ensemble_model.fit(X, y)
        logger.info("Bagging集成模型构建完成")
    
    def _evaluate_model(self, X: pd.DataFrame, y: pd.Series):
        """评估模型性能"""
        
        logger.info("评估模型性能...")
        
        # 单模型评估
        for name, model in self.models.items():
            try:
                if hasattr(model, 'predict'):
                    if name in ['mlp', 'svr']:
                        # 需要标准化的模型
                        X_scaled = self.scaler.transform(X)
                        y_pred = model.predict(X_scaled)
                    else:
                        y_pred = model.predict(X)
                    
                    mse = mean_squared_error(y, y_pred)
                    mae = mean_absolute_error(y, y_pred)
                    r2 = r2_score(y, y_pred)
                    
                    self.performance_metrics_[name] = {
                        'mse': mse,
                        'mae': mae,
                        'r2': r2,
                        'rmse': np.sqrt(mse)
                    }
                    
                    logger.info(f"{name}: R²={r2:.4f}, RMSE={np.sqrt(mse):.4f}")
                    
            except Exception as e:
                logger.warning(f"无法评估模型 {name}: {e}")
        
        # 集成模型评估
        if self.ensemble_model is not None:
            try:
                y_pred = self.ensemble_model.predict(X)
                
                mse = mean_squared_error(y, y_pred)
                mae = mean_absolute_error(y, y_pred)
                r2 = r2_score(y, y_pred)
                
                self.performance_metrics_['ensemble'] = {
                    'mse': mse,
                    'mae': mae,
                    'r2': r2,
                    'rmse': np.sqrt(mse)
                }
                
                logger.info(f"Ensemble: R²={r2:.4f}, RMSE={np.sqrt(mse):.4f}")
                
            except Exception as e:
                logger.warning(f"无法评估集成模型: {e}")
    
    def _compute_feature_importance(self, feature_names: pd.Index):
        """计算特征重要性"""
        
        importance_dict = {}
        
        # 收集各模型的特征重要性
        for name, model in self.models.items():
            if hasattr(model, 'feature_importances_'):
                importance_dict[name] = dict(zip(feature_names, model.feature_importances_))
            elif hasattr(model, 'coef_'):
                importance_dict[name] = dict(zip(feature_names, np.abs(model.coef_)))
        
        # 计算平均重要性
        if importance_dict:
            avg_importance = {}
            for feature in feature_names:
                importances = [model_imp.get(feature, 0) for model_imp in importance_dict.values()]
                avg_importance[feature] = np.mean(importances)
            
            # 排序
            self.feature_importance_ = dict(sorted(avg_importance.items(), 
                                                  key=lambda x: x[1], reverse=True))
    
    def _get_best_model(self):
        """获取最佳单模型"""
        
        if not self.performance_metrics_:
            return list(self.models.values())[0]
        
        best_model_name = max(self.performance_metrics_.keys(), 
                            key=lambda x: self.performance_metrics_[x].get('r2', -np.inf))
        
        return self.models.get(best_model_name, list(self.models.values())[0])
    
    def _predict_std(self, X: pd.DataFrame) -> np.ndarray:
        """预测不确定性（标准差）"""
        
        # 使用多个模型的预测方差
        predictions = []
        
        for name, model in self.models.items():
            try:
                if name in ['mlp', 'svr']:
                    X_scaled = self.scaler.transform(X)
                    pred = model.predict(X_scaled)
                else:
                    pred = model.predict(X)
                predictions.append(pred)
            except:
                continue
        
        if predictions:
            predictions = np.array(predictions)
            return np.std(predictions, axis=0)
        else:
            return np.ones(len(X)) * 0.01  # 默认不确定性
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """获取性能指标"""
        
        metrics = self.performance_metrics_.copy()
        
        # 添加最佳模型信息
        if metrics:
            best_model_name = max(metrics.keys(), 
                                key=lambda x: metrics[x].get('r2', -np.inf))
            best_r2 = metrics[best_model_name]['r2']
            
            metrics['summary'] = {
                'best_model': best_model_name,
                'best_r2': best_r2,
                'n_models': len(self.models),
                'model_type': 'MultiFactorYConcentration'
            }
        
        # 添加特征重要性
        if self.feature_importance_:
            top_features = list(self.feature_importance_.keys())[:10]
            metrics['top_features'] = top_features
        
        return metrics


class MultiFactorSuccessModel:
    """多因素成功率预测模型"""
    
    def __init__(self, config: dict, data: pd.DataFrame = None):
        """
        初始化成功率模型
        
        Parameters:
        -----------
        config : dict
            模型配置
        data : pd.DataFrame, optional
            数据用于分析失败模式
        """
        self.config = config
        self.method = config.get('method', 'logistic')
        self.failure_factors = config.get('failure_factors', {})
        self.scenario_params = config.get('scenario_params', {})
        
        # 模型组件
        self.success_model = None
        self.failure_model = None
        self.scenario_model = None
        self.is_fitted = False
        
        # 分析数据确定建模策略
        if data is not None:
            self._analyze_failure_patterns(data)
        
        logger.info(f"成功率模型初始化完成")
        logger.info(f"  - 建模方法: {self.method}")
        logger.info(f"  - 失败因素: {list(self.failure_factors.keys())}")
    
    def fit(self, df: pd.DataFrame, success_col: str = None):
        """
        训练成功率模型
        
        Parameters:
        -----------
        df : pd.DataFrame
            训练数据
        success_col : str, optional
            成功标签列名
        """
        logger.info("开始训练成功率模型...")
        
        if success_col is not None and success_col in df.columns:
            # 有标签的监督学习
            self._fit_supervised_model(df, success_col)
        else:
            # 无标签的情景建模
            self._fit_scenario_model(df)
        
        self.is_fitted = True
        logger.info("成功率模型训练完成")
    
    def predict_success_probability(self, X: pd.DataFrame) -> np.ndarray:
        """
        预测成功概率
        
        Parameters:
        -----------
        X : pd.DataFrame
            输入特征
            
        Returns:
        --------
        np.ndarray
            成功概率
        """
        if not self.is_fitted:
            raise ValueError("模型尚未训练")
        
        if self.success_model is not None:
            # 使用监督学习模型
            return self.success_model.predict_proba(X)[:, 1]
        elif self.scenario_model is not None:
            # 使用情景模型
            return self._predict_scenario_success(X)
        else:
            # 回退到简单情景函数
            return self._simple_success_function(X)
    
    def predict_failure_probability(self, X: pd.DataFrame) -> np.ndarray:
        """
        预测失败概率
        
        Parameters:
        -----------
        X : pd.DataFrame
            输入特征
            
        Returns:
        --------
        np.ndarray
            失败概率
        """
        return 1.0 - self.predict_success_probability(X)
    
    def _analyze_failure_patterns(self, data: pd.DataFrame):
        """分析失败模式"""
        
        logger.info("分析数据中的失败模式...")
        
        # 检查是否有失败标签
        failure_indicators = ['failure', 'failed', 'no_call', 'inconclusive']
        failure_col = None
        
        for indicator in failure_indicators:
            if indicator in data.columns:
                failure_col = indicator
                break
        
        if failure_col is not None:
            failure_rate = data[failure_col].mean()
            logger.info(f"发现失败标签列: {failure_col}, 失败率: {failure_rate:.3f}")
            
            # 分析失败与特征的关系
            self._analyze_failure_correlations(data, failure_col)
        else:
            logger.info("未发现明确的失败标签，将使用情景建模")
    
    def _analyze_failure_correlations(self, data: pd.DataFrame, failure_col: str):
        """分析失败与特征的相关性"""
        
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        correlations = {}
        
        for col in numeric_cols:
            if col != failure_col:
                try:
                    corr = data[col].corr(data[failure_col])
                    if not np.isnan(corr):
                        correlations[col] = abs(corr)
                except:
                    continue
        
        # 记录高相关特征
        high_corr_features = {k: v for k, v in correlations.items() if v > 0.1}
        logger.info(f"与失败高相关的特征: {high_corr_features}")
    
    def _fit_supervised_model(self, df: pd.DataFrame, success_col: str):
        """训练监督学习模型"""
        
        # 准备特征
        feature_cols = [col for col in df.columns 
                       if col not in [success_col, 'id', 'date', 'Y_frac'] and 
                       not col.startswith('_')]
        
        X = df[feature_cols].fillna(0)
        y = df[success_col].fillna(0)
        
        if self.method == 'logistic':
            self.success_model = LogisticRegression(random_state=42)
        elif self.method == 'rf':
            self.success_model = RandomForestClassifier(random_state=42)
        elif self.method == 'xgboost':
            self.success_model = xgb.XGBClassifier(random_state=42)
        else:
            self.success_model = LogisticRegression(random_state=42)
        
        self.success_model.fit(X, y)
        
        # 评估模型
        y_pred = self.success_model.predict(X)
        accuracy = accuracy_score(y, y_pred)
        logger.info(f"监督学习成功率模型训练完成，准确率: {accuracy:.3f}")
    
    def _fit_scenario_model(self, df: pd.DataFrame):
        """训练情景模型"""
        
        logger.info("构建情景成功率模型...")
        
        # 基于先验知识构建情景函数
        self.scenario_model = {
            'base_success_rate': 0.90,
            'early_penalty': self.scenario_params.get('early_penalty_factor', 2.0),
            'bmi_penalty': self.scenario_params.get('bmi_penalty_factor', 1.5),
            'age_threshold': self.scenario_params.get('age_effect_threshold', 35),
            'parameters_calibrated': False
        }
        
        # 如果有足够数据，校准参数
        if len(df) > 50:
            self._calibrate_scenario_parameters(df)
    
    def _calibrate_scenario_parameters(self, df: pd.DataFrame):
        """校准情景参数"""
        
        # 使用观测数据校准情景参数
        logger.info("校准情景参数...")
        
        # 基于观测的Y染色体达标率来校准成功率
        if 'Y_frac' in df.columns:
            threshold = 0.04
            actual_success_rate = (df['Y_frac'] >= threshold).mean()
            
            # 调整基础成功率
            if actual_success_rate > 0:
                self.scenario_model['base_success_rate'] = min(0.95, actual_success_rate + 0.05)
            
            logger.info(f"基于观测数据校准基础成功率: {self.scenario_model['base_success_rate']:.3f}")
        
        self.scenario_model['parameters_calibrated'] = True
    
    def _predict_scenario_success(self, X: pd.DataFrame) -> np.ndarray:
        """情景成功率预测"""
        
        if self.scenario_model is None:
            return self._simple_success_function(X)
        
        base_rate = self.scenario_model['base_success_rate']
        n_samples = len(X)
        success_probs = np.full(n_samples, base_rate)
        
        # 早孕期惩罚
        if 'week' in X.columns and self.failure_factors.get('early_pregnancy', True):
            early_mask = X['week'] < 12
            early_penalty = (12 - X['week']) * 0.02 * self.scenario_model['early_penalty']
            success_probs[early_mask] -= early_penalty[early_mask]
        
        # BMI惩罚
        if 'BMI_used' in X.columns and self.failure_factors.get('high_bmi', True):
            high_bmi_mask = X['BMI_used'] > 30
            bmi_penalty = (X['BMI_used'] - 30) * 0.01 * self.scenario_model['bmi_penalty']
            success_probs[high_bmi_mask] -= bmi_penalty[high_bmi_mask]
        
        # 年龄效应
        if 'age' in X.columns and self.failure_factors.get('age_effect', True):
            age_threshold = self.scenario_model['age_threshold']
            old_age_mask = X['age'] > age_threshold
            age_penalty = (X['age'] - age_threshold) * 0.005
            success_probs[old_age_mask] -= age_penalty[old_age_mask]
        
        # 技术因素
        if self.failure_factors.get('technical_factors', True):
            # GC%异常
            if 'gc_percent' in X.columns:
                gc_normal_range = (X['gc_percent'] >= 40) & (X['gc_percent'] <= 45)
                gc_penalty = 0.05
                success_probs[~gc_normal_range] -= gc_penalty
            
            # 读段数过低
            if 'readcount' in X.columns:
                low_readcount_mask = X['readcount'] < 5e6  # 500万reads
                readcount_penalty = 0.03
                success_probs[low_readcount_mask] -= readcount_penalty
        
        # 确保概率在[0, 1]范围内
        success_probs = np.clip(success_probs, 0.1, 0.99)
        
        return success_probs
    
    def _simple_success_function(self, X: pd.DataFrame) -> np.ndarray:
        """简单情景函数"""
        
        base_rate = 0.85
        n_samples = len(X)
        success_probs = np.full(n_samples, base_rate)
        
        # 基于周数的简单调整
        if 'week' in X.columns:
            # 早期妊娠成功率较低
            early_adjustment = np.where(
                X['week'] < 12, 
                -0.1 * (12 - X['week']) / 12, 
                0.0
            )
            success_probs += early_adjustment
        
        # 基于BMI的简单调整
        if 'BMI_used' in X.columns:
            bmi_adjustment = np.where(
                X['BMI_used'] > 35,
                -0.05 * (X['BMI_used'] - 35) / 10,
                0.0
            )
            success_probs += bmi_adjustment
        
        return np.clip(success_probs, 0.1, 0.95)


# 工具函数
def create_success_probability_function(concentration_model: MultiFactorYConcentrationModel,
                                      success_model: MultiFactorSuccessModel,
                                      sigma_models: dict) -> Callable:
    """
    创建综合成功概率函数
    
    Parameters:
    -----------
    concentration_model : MultiFactorYConcentrationModel
        浓度预测模型
    success_model : MultiFactorSuccessModel
        成功率模型
    sigma_models : dict
        误差模型
        
    Returns:
    --------
    callable
        成功概率函数 p_success(t, features)
    """
    
    def p_success(t: float, features: dict) -> float:
        """
        计算给定时点和特征的成功概率
        
        Parameters:
        -----------
        t : float
            孕周
        features : dict
            特征字典
            
        Returns:
        --------
        float
            成功概率
        """
        # 构建输入DataFrame
        input_data = pd.DataFrame([features])
        input_data['week'] = t
        
        # 预测Y染色体浓度
        y_pred = concentration_model.predict(input_data)[0]
        
        # 获取预测不确定性
        sigma = sigma_models.get('global_sigma', 0.01)
        if 'local_sigma_func' in sigma_models:
            sigma = sigma_models['local_sigma_func'](t, features.get('BMI_used', 25))
        
        # 动态阈值调整
        threshold = 0.04
        if sigma_models.get('dynamic_threshold', False):
            z_alpha = 1.96  # 95%置信水平
            threshold = 0.04 + z_alpha * sigma
        
        # 达标概率
        p_hit = stats.norm.cdf((y_pred - threshold) / sigma)
        
        # 测序成功概率
        p_tech_success = success_model.predict_success_probability(input_data)[0]
        
        # 综合成功概率
        total_success = p_hit * p_tech_success
        
        return float(np.clip(total_success, 0.01, 0.99))
    
    return p_success
