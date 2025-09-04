import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

def load_and_preprocess_data(file_path='appendix.xlsx'):
    """
    数据加载、清洗与预处理
    
    Args:
        file_path (str): 数据文件路径，默认为'appendix.xlsx'
    
    Returns:
        tuple: (X_train, X_test, y_train, y_test, scaler)
    """
    print("正在加载数据...")
    
    # 读取数据
    try:
        data = pd.read_excel(file_path)
        print(f"数据加载成功，共{len(data)}行数据")
    except FileNotFoundError:
        print(f"错误：找不到文件 {file_path}")
        return None, None, None, None, None, None
    except Exception as e:
        print(f"读取文件时出错：{e}")
        return None, None, None, None, None, None
    
    # 显示数据基本信息
    print("\n数据基本信息：")
    print(f"数据形状：{data.shape}")
    print(f"列名：{list(data.columns)}")
    
    # 数据预处理：选择需要的列
    # 根据实际数据调整列名
    required_columns = ['Y染色体浓度', '检测孕周', '孕妇BMI', '年龄', '身高', '体重']
    
    # 检查哪些列存在
    available_columns = [col for col in required_columns if col in data.columns]
    missing_columns = [col for col in required_columns if col not in data.columns]
    
    if missing_columns:
        print(f"警告：以下列不存在于数据中：{missing_columns}")
        print(f"可用的列：{available_columns}")
    
    # 使用可用的列
    data_cleaned = data[available_columns].copy()
    
    # 处理孕妇的孕周数据（从"周+天"格式转换为以周为单位的数值）
    def week_to_numeric(week_str):
        """将孕周从'周+天'格式转换为数值"""
        try:
            if pd.isna(week_str):
                return np.nan
            week_str = str(week_str)
            if 'w+' in week_str:
                weeks, days = week_str.split('w+')
                return int(weeks) + int(days) / 7
            else:
                # 如果格式不匹配，尝试直接转换为数字
                return float(week_str)
        except:
            return np.nan
    
    if '检测孕周' in data_cleaned.columns:
        print("正在处理孕周数据...")
        data_cleaned['检测孕周'] = data_cleaned['检测孕周'].apply(week_to_numeric)
    
    # 清洗数据，去掉缺失值
    print(f"清洗前数据行数：{len(data_cleaned)}")
    data_cleaned.dropna(inplace=True)
    print(f"清洗后数据行数：{len(data_cleaned)}")
    
    if len(data_cleaned) == 0:
        print("错误：清洗后没有剩余数据")
        return None, None, None, None, None, None
    
    # 特征与目标变量
    feature_columns = [col for col in available_columns if col != 'Y染色体浓度']
    X = data_cleaned[feature_columns]
    y = data_cleaned['Y染色体浓度']
    
    print(f"\n特征列：{feature_columns}")
    print(f"目标变量：Y染色体浓度")
    print(f"特征数据形状：{X.shape}")
    print(f"目标数据形状：{y.shape}")
    
    # 数据标准化
    print("正在进行数据标准化...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 数据拆分：80%训练数据，20%测试数据
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )
    
    print(f"训练集大小：{X_train.shape}")
    print(f"测试集大小：{X_test.shape}")
    
    # 显示数据统计信息
    print("\n数据统计信息：")
    print("特征统计：")
    feature_df = pd.DataFrame(X_scaled, columns=feature_columns)
    print(feature_df.describe())
    
    print("\n目标变量统计：")
    print(y.describe())
    
    return X_train, X_test, y_train, y_test, scaler, feature_columns

def save_preprocessed_data(X_train, X_test, y_train, y_test, feature_columns, 
                          scaler, output_dir='.'):
    """
    保存预处理后的数据
    
    Args:
        X_train, X_test, y_train, y_test: 训练和测试数据
        feature_columns: 特征列名
        scaler: 标准化器
        output_dir: 输出目录
    """
    import joblib
    
    # 保存数据
    np.save(f'{output_dir}/X_train.npy', X_train)
    np.save(f'{output_dir}/X_test.npy', X_test)
    np.save(f'{output_dir}/y_train.npy', y_train)
    np.save(f'{output_dir}/y_test.npy', y_test)
    
    # 保存特征列名和标准化器
    with open(f'{output_dir}/feature_columns.txt', 'w', encoding='utf-8') as f:
        for col in feature_columns:
            f.write(f"{col}\n")
    
    joblib.dump(scaler, f'{output_dir}/scaler.pkl')
    
    print(f"预处理数据已保存到 {output_dir}")

if __name__ == "__main__":
    # 运行数据预处理
    X_train, X_test, y_train, y_test, scaler, feature_columns = load_and_preprocess_data()
    
    if X_train is not None:
        # 保存预处理后的数据
        save_preprocessed_data(X_train, X_test, y_train, y_test, feature_columns, scaler)
        print("数据预处理完成！")
    else:
        print("数据预处理失败！")
