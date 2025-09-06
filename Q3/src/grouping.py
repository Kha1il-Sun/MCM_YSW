"""
BMI分组模块
使用CART算法基于BMI进行合理分组
"""

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import logging

logger = logging.getLogger(__name__)


class BMIGrouper:
    """BMI分组器"""
    
    def __init__(self, config):
        self.config = config
        self.cart_params = config['models']['cart']
        self.tree_model = None
        self.group_boundaries = None
        self.group_stats = None
        
    def fit_cart_model(self, X, y):
        """训练CART模型进行BMI分组"""
        logger.info("训练CART模型进行BMI分组...")
        
        # 提取BMI特征
        if 'BMI' not in X.columns:
            raise ValueError("数据中缺少BMI特征")
        
        bmi_data = X[['BMI']].copy()
        
        # 训练CART模型
        self.tree_model = DecisionTreeRegressor(
            min_samples_split=self.cart_params['min_samples_split'],
            min_samples_leaf=self.cart_params['min_samples_leaf'],
            max_depth=self.cart_params['max_depth'],
            random_state=42
        )
        
        self.tree_model.fit(bmi_data, y)
        
        # 提取分组边界
        self._extract_group_boundaries()
        
        logger.info(f"CART模型训练完成，分组数: {len(self.group_boundaries) + 1}")
        return self
    
    def _extract_group_boundaries(self):
        """从决策树中提取分组边界"""
        tree = self.tree_model.tree_
        boundaries = []
        
        def extract_boundaries(node, depth=0):
            if tree.children_left[node] != tree.children_right[node]:  # 非叶子节点
                feature = tree.feature[node]
                threshold = tree.threshold[node]
                
                if feature == 0:  # BMI特征
                    boundaries.append(threshold)
                
                # 递归处理子节点
                extract_boundaries(tree.children_left[node], depth + 1)
                extract_boundaries(tree.children_right[node], depth + 1)
        
        extract_boundaries(0)
        self.group_boundaries = sorted(boundaries)
        
        logger.info(f"分组边界: {self.group_boundaries}")
    
    def predict_groups(self, X):
        """预测BMI分组"""
        if self.tree_model is None:
            raise ValueError("模型尚未训练")
        
        bmi_data = X[['BMI']].copy()
        groups = self.tree_model.predict(bmi_data)
        return groups
    
    def get_group_labels(self, X):
        """获取分组标签"""
        if self.group_boundaries is None:
            raise ValueError("分组边界尚未计算")
        
        bmi_values = X['BMI'].values
        labels = []
        
        for bmi in bmi_values:
            group_idx = 0
            for boundary in self.group_boundaries:
                if bmi >= boundary:
                    group_idx += 1
                else:
                    break
            labels.append(f"组{group_idx + 1}")
        
        return labels
    
    def analyze_groups(self, X, y, optimal_weeks):
        """分析各组的统计信息"""
        logger.info("分析BMI分组统计信息...")
        
        # 获取分组标签
        group_labels = self.get_group_labels(X)
        
        # 创建结果DataFrame
        results_df = pd.DataFrame({
            'BMI': X['BMI'].values,
            'optimal_week': optimal_weeks,
            'group': group_labels
        })
        
        # 计算各组统计信息
        group_stats = []
        
        for group in sorted(set(group_labels)):
            group_data = results_df[results_df['group'] == group]
            
            if len(group_data) == 0:
                continue
            
            # BMI范围
            bmi_min = group_data['BMI'].min()
            bmi_max = group_data['BMI'].max()
            bmi_mean = group_data['BMI'].mean()
            bmi_std = group_data['BMI'].std()
            
            # 最优时点统计
            week_mean = group_data['optimal_week'].mean()
            week_std = group_data['optimal_week'].std()
            week_median = group_data['optimal_week'].median()
            week_min = group_data['optimal_week'].min()
            week_max = group_data['optimal_week'].max()
            
            # 样本数
            n_samples = len(group_data)
            
            group_stats.append({
                'group': group,
                'n_samples': n_samples,
                'bmi_min': bmi_min,
                'bmi_max': bmi_max,
                'bmi_mean': bmi_mean,
                'bmi_std': bmi_std,
                'bmi_range': f"{bmi_min:.1f}-{bmi_max:.1f}",
                'optimal_week_mean': week_mean,
                'optimal_week_std': week_std,
                'optimal_week_median': week_median,
                'optimal_week_min': week_min,
                'optimal_week_max': week_max,
                'recommended_week': week_median  # 推荐时点使用中位数
            })
        
        self.group_stats = pd.DataFrame(group_stats)
        
        logger.info("BMI分组分析完成:")
        for _, row in self.group_stats.iterrows():
            logger.info(f"  {row['group']}: BMI {row['bmi_range']}, "
                       f"推荐时点 {row['recommended_week']:.1f}周, "
                       f"样本数 {row['n_samples']}")
        
        return self.group_stats
    
    def get_recommendations(self):
        """获取分组推荐结果"""
        if self.group_stats is None:
            raise ValueError("尚未进行分组分析")
        
        recommendations = []
        
        for _, row in self.group_stats.iterrows():
            recommendations.append({
                'BMI组': row['group'],
                'BMI范围': row['bmi_range'],
                '推荐检测时点': f"{row['recommended_week']:.1f}周",
                '样本数': row['n_samples'],
                'BMI均值': f"{row['bmi_mean']:.1f}",
                '时点范围': f"{row['optimal_week_min']:.1f}-{row['optimal_week_max']:.1f}周"
            })
        
        return pd.DataFrame(recommendations)
    
    def plot_group_analysis(self, X, y, optimal_weeks, save_path=None):
        """绘制分组分析图"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. BMI分布直方图
        axes[0, 0].hist(X['BMI'], bins=30, alpha=0.7, edgecolor='black')
        axes[0, 0].axvline(x=self.group_boundaries[0] if self.group_boundaries else 0, 
                          color='red', linestyle='--', label='分组边界')
        for boundary in self.group_boundaries[1:]:
            axes[0, 0].axvline(x=boundary, color='red', linestyle='--')
        axes[0, 0].set_xlabel('BMI')
        axes[0, 0].set_ylabel('频数')
        axes[0, 0].set_title('BMI分布与分组边界')
        axes[0, 0].legend()
        
        # 2. BMI vs 最优时点散点图
        group_labels = self.get_group_labels(X)
        colors = plt.cm.Set1(np.linspace(0, 1, len(set(group_labels))))
        
        for i, group in enumerate(sorted(set(group_labels))):
            mask = np.array(group_labels) == group
            axes[0, 1].scatter(X['BMI'][mask], np.array(optimal_weeks)[mask], 
                              c=[colors[i]], label=group, alpha=0.7)
        
        axes[0, 1].set_xlabel('BMI')
        axes[0, 1].set_ylabel('最优检测时点（周）')
        axes[0, 1].set_title('BMI与最优检测时点关系')
        axes[0, 1].legend()
        
        # 3. 各组最优时点箱线图
        group_data = []
        group_names = []
        for group in sorted(set(group_labels)):
            mask = np.array(group_labels) == group
            group_data.append(np.array(optimal_weeks)[mask])
            group_names.append(group)
        
        axes[1, 0].boxplot(group_data, labels=group_names)
        axes[1, 0].set_ylabel('最优检测时点（周）')
        axes[1, 0].set_title('各组最优检测时点分布')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # 4. 决策树可视化（简化版）
        if self.tree_model is not None:
            from sklearn.tree import plot_tree
            plot_tree(self.tree_model, ax=axes[1, 1], feature_names=['BMI'], 
                     class_names=['时点'], filled=True, rounded=True)
            axes[1, 1].set_title('BMI分组决策树')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"分组分析图已保存: {save_path}")
        
        plt.show()
        
        return fig
