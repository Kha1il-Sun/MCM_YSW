import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class YChromosomePredictor:
    """Y染色体浓度预测模型类"""
    
    def __init__(self, input_dim=None):
        """
        初始化模型
        
        Args:
            input_dim (int): 输入特征维度
        """
        self.model = None
        self.input_dim = input_dim
        self.history = None
        self.scaler = None
        self.feature_columns = None
        
    def build_model(self, input_dim, hidden_layers=[64, 32], dropout_rate=0.2):
        """
        构建深度学习模型
        
        Args:
            input_dim (int): 输入特征维度
            hidden_layers (list): 隐藏层神经元数量列表
            dropout_rate (float): Dropout比率
        """
        self.input_dim = input_dim
        
        # 构建深度学习模型
        self.model = Sequential()
        
        # 输入层和第一个隐藏层
        self.model.add(Dense(hidden_layers[0], input_dim=input_dim, activation='relu'))
        self.model.add(Dropout(dropout_rate))
        
        # 其他隐藏层
        for units in hidden_layers[1:]:
            self.model.add(Dense(units, activation='relu'))
            self.model.add(Dropout(dropout_rate))
        
        # 输出层，预测Y染色体浓度
        self.model.add(Dense(1, activation='linear'))
        
        # 编译模型
        self.model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        print("模型构建完成！")
        print(f"模型结构：")
        self.model.summary()
        
    def train_model(self, X_train, y_train, X_val, y_val, 
                   epochs=200, batch_size=32, verbose=1):
        """
        训练模型
        
        Args:
            X_train, y_train: 训练数据
            X_val, y_val: 验证数据
            epochs (int): 训练轮数
            batch_size (int): 批次大小
            verbose (int): 输出详细程度
        """
        if self.model is None:
            raise ValueError("请先构建模型！")
        
        # 设置回调函数
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=20,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=10,
                min_lr=1e-6,
                verbose=1
            )
        ]
        
        print("开始训练模型...")
        
        # 训练模型
        self.history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            verbose=verbose
        )
        
        print("模型训练完成！")
        
    def evaluate_model(self, X_test, y_test):
        """
        评估模型性能
        
        Args:
            X_test, y_test: 测试数据
        
        Returns:
            dict: 评估指标字典
        """
        if self.model is None:
            raise ValueError("请先训练模型！")
        
        # 预测
        y_pred = self.model.predict(X_test, verbose=0)
        y_pred = y_pred.flatten()
        
        # 计算评估指标
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        
        metrics = {
            'MSE': mse,
            'MAE': mae,
            'RMSE': rmse,
            'R²': r2
        }
        
        print("模型评估结果：")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.6f}")
        
        return metrics, y_pred
    
    def plot_training_history(self, save_path=None):
        """绘制训练历史"""
        if self.history is None:
            print("没有训练历史可显示！")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # 损失函数
        ax1.plot(self.history.history['loss'], label='训练损失')
        ax1.plot(self.history.history['val_loss'], label='验证损失')
        ax1.set_title('模型损失')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # MAE
        ax2.plot(self.history.history['mae'], label='训练MAE')
        ax2.plot(self.history.history['val_mae'], label='验证MAE')
        ax2.set_title('平均绝对误差')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('MAE')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"训练历史图已保存到：{save_path}")
        
        plt.show()
    
    def plot_predictions(self, y_true, y_pred, save_path=None):
        """绘制预测结果对比图"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # 散点图：真实值 vs 预测值
        ax1.scatter(y_true, y_pred, alpha=0.6)
        ax1.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
        ax1.set_xlabel('真实值')
        ax1.set_ylabel('预测值')
        ax1.set_title('真实值 vs 预测值')
        ax1.grid(True)
        
        # 残差图
        residuals = y_true - y_pred
        ax2.scatter(y_pred, residuals, alpha=0.6)
        ax2.axhline(y=0, color='r', linestyle='--')
        ax2.set_xlabel('预测值')
        ax2.set_ylabel('残差')
        ax2.set_title('残差图')
        ax2.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"预测结果图已保存到：{save_path}")
        
        plt.show()
    
    def save_model(self, model_path, scaler_path=None, feature_columns_path=None):
        """保存模型"""
        if self.model is None:
            raise ValueError("没有模型可保存！")
        
        # 保存模型
        self.model.save(model_path)
        print(f"模型已保存到：{model_path}")
        
        # 保存标准化器
        if self.scaler is not None and scaler_path:
            joblib.dump(self.scaler, scaler_path)
            print(f"标准化器已保存到：{scaler_path}")
        
        # 保存特征列名
        if self.feature_columns is not None and feature_columns_path:
            with open(feature_columns_path, 'w', encoding='utf-8') as f:
                for col in self.feature_columns:
                    f.write(f"{col}\n")
            print(f"特征列名已保存到：{feature_columns_path}")
    
    def load_model(self, model_path, scaler_path=None, feature_columns_path=None):
        """加载模型"""
        from tensorflow.keras.models import load_model
        
        # 加载模型
        self.model = load_model(model_path)
        print(f"模型已从 {model_path} 加载")
        
        # 加载标准化器
        if scaler_path:
            self.scaler = joblib.load(scaler_path)
            print(f"标准化器已从 {scaler_path} 加载")
        
        # 加载特征列名
        if feature_columns_path:
            with open(feature_columns_path, 'r', encoding='utf-8') as f:
                self.feature_columns = [line.strip() for line in f.readlines()]
            print(f"特征列名已从 {feature_columns_path} 加载")
    
    def predict(self, X):
        """预测Y染色体浓度"""
        if self.model is None:
            raise ValueError("请先加载或训练模型！")
        
        return self.model.predict(X, verbose=0).flatten()

def load_preprocessed_data(data_dir='.'):
    """加载预处理后的数据"""
    try:
        X_train = np.load(f'{data_dir}/X_train.npy')
        X_test = np.load(f'{data_dir}/X_test.npy')
        y_train = np.load(f'{data_dir}/y_train.npy')
        y_test = np.load(f'{data_dir}/y_test.npy')
        
        with open(f'{data_dir}/feature_columns.txt', 'r', encoding='utf-8') as f:
            feature_columns = [line.strip() for line in f.readlines()]
        
        scaler = joblib.load(f'{data_dir}/scaler.pkl')
        
        print("预处理数据加载成功！")
        return X_train, X_test, y_train, y_test, scaler, feature_columns
    
    except FileNotFoundError as e:
        print(f"错误：找不到预处理数据文件 - {e}")
        return None, None, None, None, None, None

if __name__ == "__main__":
    # 加载预处理后的数据
    X_train, X_test, y_train, y_test, scaler, feature_columns = load_preprocessed_data()
    
    if X_train is not None:
        # 创建模型实例
        predictor = YChromosomePredictor()
        
        # 构建模型
        predictor.build_model(
            input_dim=X_train.shape[1],
            hidden_layers=[64, 32, 16],
            dropout_rate=0.2
        )
        
        # 训练模型
        predictor.train_model(
            X_train, y_train, X_test, y_test,
            epochs=200,
            batch_size=32
        )
        
        # 评估模型
        metrics, y_pred = predictor.evaluate_model(X_test, y_test)
        
        # 绘制训练历史
        predictor.plot_training_history('training_history.png')
        
        # 绘制预测结果
        predictor.plot_predictions(y_test, y_pred, 'predictions.png')
        
        # 保存模型
        predictor.scaler = scaler
        predictor.feature_columns = feature_columns
        predictor.save_model(
            'y_chromosome_model.h5',
            'scaler.pkl',
            'feature_columns.txt'
        )
        
        print("模型训练和评估完成！")
    else:
        print("请先运行 data_preprocessing.py 进行数据预处理！")
