import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# ---------------------- 步骤1：真实数据输入与预处理 ----------------------
try:
    df = pd.read_csv('appendix.csv')  # 读取原始数据（需确保列：G(孕周)、B(BMI)、C_Y(Y浓度)）
    print(f"✅ 成功读取原始数据 | 数据形状: {df.shape}")
    print(f"数据列名: {list(df.columns)}")
except FileNotFoundError:
    raise FileNotFoundError("请确保数据文件 'appendix.csv' 存在！")

# 重命名列以便后续处理
df = df.rename(columns={
    '检测孕周': 'G',
    '孕妇BMI': 'B', 
    'Y染色体浓度': 'C_Y'
})

# 处理孕周数据：将字符串格式（如"11w+6"）转换为数值
def convert_gestational_week(week_str):
    """将孕周字符串转换为数值（周）"""
    if pd.isna(week_str):
        return np.nan
    try:
        # 处理格式如 "11w+6" 或 "11w"
        if 'w' in str(week_str):
            parts = str(week_str).replace('w', '').split('+')
            weeks = float(parts[0])
            if len(parts) > 1 and parts[1]:
                days = float(parts[1])
                weeks += days / 7  # 将天数转换为周
            return weeks
        else:
            return float(week_str)
    except:
        return np.nan

df['G'] = df['G'].apply(convert_gestational_week)

# 处理缺失值：删除孕周/ BMI/ Y浓度缺失的行
df = df.dropna(subset=['G', 'B', 'C_Y'])
print(f"✅ 缺失值处理后 | 数据形状: {df.shape}")
print(f"孕周范围: {df['G'].min():.2f} - {df['G'].max():.2f} 周")
print(f"BMI范围: {df['B'].min():.2f} - {df['B'].max():.2f}")
print(f"Y染色体浓度范围: {df['C_Y'].min():.6f} - {df['C_Y'].max():.6f}")

# 特征工程：构造非线性特征（二次项、交互项）
df['G_sq'] = df['G'] ** 2       # 孕周二次项
df['B_sq'] = df['B'] ** 2       # BMI二次项
df['G_B_inter'] = df['G'] * df['B']  # 孕周与BMI的交互项

# 划分特征（X）和目标变量（y）
X = df[['G', 'B', 'G_sq', 'B_sq', 'G_B_inter']]  # 多特征输入
y = df['C_Y']

# 划分训练集（80%）和测试集（20%）
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42  # 固定随机种子保证可复现
)


# ---------------------- 步骤2：多模型训练与评估 ----------------------
### 模型1：线性回归（多项式特征）
poly = PolynomialFeatures(degree=2, include_bias=False)  # 构造二次多项式特征
X_train_poly = poly.fit_transform(X_train[['G', 'B']])   # 仅用孕周、BMI原始特征
X_test_poly = poly.transform(X_test[['G', 'B']])

lr = LinearRegression()
lr.fit(X_train_poly, y_train)
y_pred_lr = lr.predict(X_test_poly)  # 测试集预测


### 模型2：随机森林回归（非线性模型）
rf = RandomForestRegressor(
    n_estimators=100,   # 决策树数量（可调整，越大越准但越慢）
    max_depth=8,        # 树的最大深度（防止过拟合）
    random_state=42
)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)  # 测试集预测


### 模型3：Statsmodels公式化建模（统计显著性分析）
formula = 'C_Y ~ G + G_sq + B + B_sq + G_B_inter'  # 含所有构造特征
sm_model = smf.ols(formula, data=df).fit()


### 交叉验证（10折，评估泛化能力）
# 线性回归（多项式）的交叉验证
cv_lr_scores = cross_val_score(
    lr, poly.transform(X[['G', 'B']]), y, 
    cv=10, scoring='r2'
)
# 随机森林的交叉验证
cv_rf_scores = cross_val_score(
    rf, X, y, 
    cv=10, scoring='r2'
)


# ---------------------- 步骤3：结果输出 ----------------------
print("\n=== 问题一：Y染色体浓度与孕周、BMI的关系（线性+非线性模型对比）===")

### 3.1 线性回归（多项式）测试集效果
print("\n### 1. 线性回归（二次多项式）测试集效果 ###")
print(f"测试集R²: {r2_score(y_test, y_pred_lr):.4f}")
print(f"测试集MSE: {mean_squared_error(y_test, y_pred_lr):.6f}")

### 3.2 随机森林测试集效果
print("\n### 2. 随机森林测试集效果 ###")
print(f"测试集R²: {r2_score(y_test, y_pred_rf):.4f}")
print(f"测试集MSE: {mean_squared_error(y_test, y_pred_rf):.6f}")

### 3.3 交叉验证结果
print("\n### 3. 10折交叉验证结果 ###")
print(f"线性回归（多项式）平均R²: {cv_lr_scores.mean():.4f}, 标准差: {cv_lr_scores.std():.4f}")
print(f"随机森林平均R²: {cv_rf_scores.mean():.4f}, 标准差: {cv_rf_scores.std():.4f}")

### 3.4 Statsmodels统计摘要
print("\n### 4. Statsmodels 统计模型摘要 ###")
print(sm_model.summary())


# ---------------------- 步骤4：可视化分析 ----------------------
### 图1：测试集“真实值 vs 模型预测值”对比
plt.figure(figsize=(12, 6))
# 线性回归预测点
plt.scatter(y_test, y_pred_lr, alpha=0.5, 
            label=f'线性回归（R²={r2_score(y_test, y_pred_lr):.4f}）')
# 随机森林预测点
plt.scatter(y_test, y_pred_rf, alpha=0.5, 
            label=f'随机森林（R²={r2_score(y_test, y_pred_rf):.4f}）')
# 参考线（y=x，代表预测完全准确）
plt.plot([y_test.min(), y_test.max()], 
         [y_test.min(), y_test.max()], 'r--')
plt.xlabel('真实Y染色体浓度（%）')
plt.ylabel('预测Y染色体浓度（%）')
plt.title('线性回归 vs 随机森林：测试集预测效果对比')
plt.legend()
plt.grid(True)
plt.savefig('model_comparison_testset.png', dpi=300)
plt.show()


### 图2：随机森林特征重要性（分析哪些特征最关键）
importances = rf.feature_importances_
feature_names = X.columns
# 按重要性排序
indices = np.argsort(importances)[::-1]  

plt.figure(figsize=(10, 6))
plt.bar(
    range(len(importances)), 
    importances[indices], 
    tick_label=feature_names[indices]
)
plt.xlabel('特征')
plt.ylabel('重要性权重')
plt.title('随机森林：特征重要性排序')
plt.xticks(rotation=45)  # 旋转特征名防止重叠
plt.grid(axis='y', alpha=0.3)
plt.savefig('rf_feature_importance.png', dpi=300)
plt.show()


### 图3：固定BMI，展示“孕周 vs Y浓度”的模型拟合曲线（线性 vs 随机森林）
bmi_mean = df['B'].mean()  # BMI取数据集均值，隔离BMI影响
G_range = np.linspace(df['G'].min(), df['G'].max(), 100)  # 生成连续孕周序列

#### 线性回归的拟合曲线
X_lr_plot = np.column_stack((G_range, np.full_like(G_range, bmi_mean)))  # 固定BMI
X_lr_plot_poly = poly.transform(X_lr_plot)  # 转换为二次多项式特征
y_lr_plot = lr.predict(X_lr_plot_poly)     # 预测Y浓度

#### 随机森林的拟合曲线
# 构造随机森林的输入（所有特征需补全，其他特征固定为均值）
G_sq_range = G_range ** 2
B_sq_value = bmi_mean ** 2
G_B_inter_range = G_range * bmi_mean
X_rf_plot = pd.DataFrame({
    'G': G_range,
    'B': np.full_like(G_range, bmi_mean),
    'G_sq': G_sq_range,
    'B_sq': np.full_like(G_range, B_sq_value),
    'G_B_inter': G_B_inter_range
})
y_rf_plot = rf.predict(X_rf_plot)  # 预测Y浓度

#### 绘图
plt.figure(figsize=(12, 6))
plt.scatter(df['G'], df['C_Y'], alpha=0.3, label='原始数据')  # 原始数据散点
plt.plot(G_range, y_lr_plot, 'r-', linewidth=2, label='线性回归（二次多项式）')
plt.plot(G_range, y_rf_plot, 'g-', linewidth=2, label='随机森林')
plt.xlabel('孕周（周）')
plt.ylabel('Y染色体浓度（%）')
plt.title(f'孕周与Y染色体浓度的关系（固定BMI={bmi_mean:.2f}）')
plt.legend()
plt.grid(True)
plt.savefig('linear_vs_rf_fitting.png', dpi=300)
plt.show()