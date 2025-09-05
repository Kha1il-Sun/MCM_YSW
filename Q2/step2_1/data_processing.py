import pandas as pd
import numpy as np
import re
from datetime import datetime
from scipy import stats

FILE = "../appendix.xlsx"
SHEET = "男胎检测数据"

# 1) 读取数据
df_raw = pd.read_excel(FILE, sheet_name=SHEET)

# 2) 统一列名（便于后续处理：去空格）
df = df_raw.copy()
df.columns = [str(c).strip() for c in df.columns]

# 3) 关键列存在性检查
cols_needed = ["孕妇代码", "检测孕周", "身高", "体重", "Y染色体浓度", "检测日期", "末次月经"]
missing = [c for c in cols_needed if c not in df.columns]

# 4) 解析孕周（如 "11w+6" → 11 + 6/7）
def parse_week_str(s):
    if pd.isna(s):
        return np.nan
    s = str(s).strip().lower()
    m = re.match(r"^\s*(\d+)\s*w(?:\s*\+\s*(\d+))?\s*$", s)
    if m:
        w = int(m.group(1))
        d = int(m.group(2)) if m.group(2) is not None else 0
        return w + d/7.0
    try:
        return float(s)
    except:
        return np.nan

df["week_from_text"] = df.get("检测孕周", np.nan).apply(parse_week_str)

# 5) 解析日期（检测日期/末次月经）并计算按日期推得的孕周（天/7）
def parse_date_generic(x):
    if pd.isna(x):
        return pd.NaT
    s = str(x).strip()
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%Y%m%d", "%Y.%m.%d"):
        try:
            return datetime.strptime(s, fmt).date()
        except:
            pass
    return pd.NaT

detect_date_col = "检测日期"
lmp_col = "末次月经"
df["检测日期_parsed"] = df.get(detect_date_col, pd.Series([np.nan]*len(df))).apply(parse_date_generic)
df["末次月经_parsed"] = df.get(lmp_col, pd.Series([np.nan]*len(df))).apply(parse_date_generic)

def ga_from_dates(row):
    d = row["检测日期_parsed"]
    l = row["末次月经_parsed"]
    if pd.isna(d) or pd.isna(l):
        return np.nan
    try:
        delta_days = (d - l).days
        return delta_days/7.0
    except:
        return np.nan

df["week_from_dates"] = df.apply(ga_from_dates, axis=1)

# 6) 统一采用的孕周：优先文本列，其次日期推算（两者都在则做差值报告）
df["week_used"] = df["week_from_text"]
mask_fill = df["week_used"].isna() & df["week_from_dates"].notna()
df.loc[mask_fill, "week_used"] = df.loc[mask_fill, "week_from_dates"]

df["week_diff_text_vs_date"] = df["week_from_text"] - df["week_from_dates"]

# 统计孕周差异分布并过滤异常值
week_diff_valid = df["week_diff_text_vs_date"].dropna()
if len(week_diff_valid) > 0:
    week_diff_median = np.median(week_diff_valid)
    week_diff_iqr = np.percentile(week_diff_valid, [25, 75])
    week_diff_95ci = np.percentile(week_diff_valid, [2.5, 97.5])
    
    # 过滤明显异常（>3周差异）的个体
    week_diff_extreme_mask = np.abs(df["week_diff_text_vs_date"]) > 3.0
    week_diff_extreme_count = week_diff_extreme_mask.sum()
    df = df[~week_diff_extreme_mask].copy()
    
    print(f"孕周差异统计:")
    print(f"  中位数: {week_diff_median:.2f} 周")
    print(f"  IQR: [{week_diff_iqr[0]:.2f}, {week_diff_iqr[1]:.2f}] 周")
    print(f"  95%区间: [{week_diff_95ci[0]:.2f}, {week_diff_95ci[1]:.2f}] 周")
    print(f"  过滤孕周差异>3周的异常个体: 剔除 {week_diff_extreme_count} 条记录")
else:
    week_diff_median = np.nan
    week_diff_iqr = [np.nan, np.nan]
    week_diff_95ci = [np.nan, np.nan]
    week_diff_extreme_count = 0

# 7) 计算 BMI（如果数据已有 BMI 列就对比，不覆盖）
def to_float(x):
    try:
        return float(x)
    except:
        return np.nan

df["身高_cm"] = df["身高"].apply(to_float)
df["体重_kg"] = df["体重"].apply(to_float)

bmi_calc = df["体重_kg"] / (df["身高_cm"]/100.0)**2
df["BMI_calc"] = bmi_calc

bmi_given_col = None
for c in df.columns:
    if c.upper() == "BMI" or c == "BMI":
        bmi_given_col = c
        break

if bmi_given_col is not None:
    df["BMI_given"] = pd.to_numeric(df[bmi_given_col], errors="coerce")
else:
    df["BMI_given"] = np.nan

def choose_bmi(row, tol=0.5):
    bg = row["BMI_given"]
    bc = row["BMI_calc"]
    if not pd.isna(bg) and not pd.isna(bc):
        if abs(bg - bc) <= tol:
            return bg, "given≈calc"
        else:
            return bc, "calc_over_given"
    elif not pd.isna(bg):
        return bg, "given_only"
    else:
        return bc, "calc_only"

res = df.apply(lambda r: choose_bmi(r), axis=1, result_type="expand")
df["BMI_used"] = res[0]
df["BMI_source"] = res[1]

# 8) 统一Y染色体浓度为比例（0-1）并过滤极端异常值
df["Y_raw"] = pd.to_numeric(df["Y染色体浓度"], errors="coerce")
if (df["Y_raw"] > 1).mean() > 0.2:
    df["Y_frac"] = df["Y_raw"] / 100.0
else:
    df["Y_frac"] = df["Y_raw"]

# 数据驱动的Y极值过滤（使用分位数）
y_q995 = df["Y_frac"].quantile(0.995)
y_extreme_mask = df["Y_frac"] > y_q995
y_extreme_count = y_extreme_mask.sum()
df = df[~y_extreme_mask].copy()
print(f"过滤Y染色体浓度极端异常值（>Q99.5={y_q995:.4f}）: 剔除 {y_extreme_count} 条记录")

# 9) 同日多次检测处理：
thr = 0.04  # 4% as fraction
df["same_day_rank"] = df.groupby(["孕妇代码", "检测日期_parsed"])["week_used"].rank(method="first")

same_day_max = (df.groupby(["孕妇代码", "检测日期_parsed"])["Y_frac"]
                  .agg(["max", "count"])
                  .rename(columns={"max": "Y_same_day_max", "count":"same_day_n"}))
df = df.merge(same_day_max, left_on=["孕妇代码", "检测日期_parsed"],
              right_index=True, how="left")
df["same_day_any_hit"] = (df["Y_same_day_max"] >= thr).astype(int)

# 10) 生成长表与事件-删失表
long_cols = ["孕妇代码","检测日期_parsed","week_used","BMI_used","BMI_source","Y_frac",
             "same_day_n","same_day_any_hit"]
long_df = df[long_cols].copy().rename(columns={"孕妇代码":"id","检测日期_parsed":"date","week_used":"week"})

def construct_interval_censoring(g):
    """构造区间删失结构 - 使用记录级别边界"""
    g = g.sort_values(["week","date"])
    
    # 获取首检BMI
    bmi_base = g["BMI_used"].iloc[0] if len(g) > 0 and not pd.isna(g["BMI_used"].iloc[0]) else np.nan
    
    # 逐条扫描，不做周聚合，避免L=R的问题
    last_below = None
    first_hit = None
    for _, r in g.iterrows():
        if r["Y_frac"] >= thr and first_hit is None:
            first_hit = float(r["week"])
            break
        last_below = float(r["week"])
    
    if first_hit is not None:
        if last_below is not None and last_below < first_hit:
            L, R, censor_type = last_below, first_hit, "interval"
        else:
            L, R, censor_type = np.nan, first_hit, "left"  # 首次检测即达标或没有下界
    else:
        L, R, censor_type = float(g["week"].max()), np.nan, "right"
    
    return pd.Series({
        "L": L, "R": R, "censor_type": censor_type, 
        "BMI_base": bmi_base, "n_records": len(g)
    })

surv_dat = long_df.groupby("id", as_index=False).apply(construct_interval_censoring).reset_index(drop=True)

# 确保id列存在
assert "id" in surv_dat.columns, "surv_dat缺少id列，请检查groupby/apply返回结构"

# 与生存库的左/右删失约定对齐
surv_dat2 = surv_dat.copy()
surv_dat2["L_fit"] = surv_dat2.apply(lambda r: -np.inf if r["censor_type"]=="left" else r["L"], axis=1)
surv_dat2["R_fit"] = surv_dat2.apply(lambda r: np.inf if r["censor_type"]=="right" else r["R"], axis=1)

# 11) 增强数据质量报告
# 检查孕周是否严格递增
def check_week_monotonicity(g):
    weeks = g["week"].dropna().sort_values()
    if len(weeks) <= 1:
        return True
    return (weeks.diff().dropna() >= 0).all()

week_monotonic_check = long_df.groupby("id").apply(check_week_monotonicity)
week_monotonic_failures = (~week_monotonic_check).sum()

# 区间删失类型分布
censor_type_dist = surv_dat["censor_type"].value_counts().to_dict()

# 检测误差建模：全局和局部化σ估计
threshold_nearby_mask = (df["Y_frac"] >= 0.03) & (df["Y_frac"] <= 0.05)
if threshold_nearby_mask.sum() > 10:  # 至少需要10个样本
    y_near_threshold = df.loc[threshold_nearby_mask, "Y_frac"]
    y_residual_var_global = np.var(y_near_threshold)
    y_residual_std_global = np.std(y_near_threshold)
    print(f"阈值附近（3-5%）Y染色体浓度残差统计:")
    print(f"  样本数: {len(y_near_threshold)}")
    print(f"  全局标准差: {y_residual_std_global:.4f}")
    print(f"  全局方差: {y_residual_var_global:.6f}")
    
    # 局部化σ估计：按(t,b)分箱
    df_nearby = df.loc[threshold_nearby_mask].copy()
    if len(df_nearby) > 20:  # 足够样本进行分箱
        # 创建孕周和BMI分箱
        df_nearby["week_bin"] = pd.cut(df_nearby["week_used"], bins=5, labels=False)
        df_nearby["bmi_bin"] = pd.cut(df_nearby["BMI_used"], bins=3, labels=False)
        
        # 计算每个(t,b)分箱的局部方差
        local_vars = df_nearby.groupby(["week_bin", "bmi_bin"])["Y_frac"].var().dropna()
        if len(local_vars) > 0:
            print(f"  局部方差分箱数: {len(local_vars)}")
            print(f"  局部方差范围: [{local_vars.min():.6f}, {local_vars.max():.6f}]")
            print(f"  局部方差中位数: {local_vars.median():.6f}")
        else:
            print("  局部方差分箱失败，使用全局方差")
    else:
        print("  样本数不足进行局部化分箱")
else:
    y_residual_std_global = np.nan
    y_residual_var_global = np.nan
    print("阈值附近（3-5%）样本数不足，无法计算残差统计")

report = {
    "总记录条数": int(len(df)),
    "唯一孕妇人数": int(df["孕妇代码"].nunique() if "孕妇代码" in df.columns else 0),
    "有文本孕周的比例": float(df["week_from_text"].notna().mean()),
    "有日期孕周的比例": float(df["week_from_dates"].notna().mean()),
    "文本vs日期孕周差的中位数(周)": float(week_diff_median) if not pd.isna(week_diff_median) else np.nan,
    "文本vs日期孕周差的IQR(周)": [float(week_diff_iqr[0]), float(week_diff_iqr[1])] if not pd.isna(week_diff_iqr[0]) else [np.nan, np.nan],
    "文本vs日期孕周差的95%区间(周)": [float(week_diff_95ci[0]), float(week_diff_95ci[1])] if not pd.isna(week_diff_95ci[0]) else [np.nan, np.nan],
    "过滤孕周差异>3周的异常个体数": int(week_diff_extreme_count),
    "同日多次检测的样本对数": int((df.groupby(["孕妇代码","检测日期_parsed"]).size()>1).sum()),
    "存在BMI给定列": bool(bmi_given_col is not None),
    "BMI来源分布": {str(k): int(v) for k,v in df["BMI_source"].value_counts(dropna=False).to_dict().items()},
    "Y染色体浓度极值过滤结果": f"剔除>Q99.5={y_q995:.4f}的异常值 {y_extreme_count} 条记录",
    "孕周严格递增检查": f"孕周非严格递增的个体数: {week_monotonic_failures}",
    "区间删失类型分布": {str(k): int(v) for k,v in censor_type_dist.items()},
    "检测误差建模": {
        "阈值附近样本数": int(threshold_nearby_mask.sum()),
        "Y浓度残差标准差": float(y_residual_std_global) if not pd.isna(y_residual_std_global) else np.nan,
        "Y浓度残差方差": float(y_residual_var_global) if not pd.isna(y_residual_var_global) else np.nan
    }
}

# 12) 保存输出
long_path = "step1_long_records.csv"
surv_path = "step1_surv_dat.csv"
surv_fit_path = "step1_surv_dat_fit.csv"  # 对齐版删失表
report_path = "step1_report.csv"
config_path = "step1_config.yaml"

df_report = pd.DataFrame([report])

# 保存配置文件
import yaml
config = {
    "thresholds": {
        "Y_concentration_threshold": 0.04,  # 4%
        "week_diff_extreme_threshold": 3.0,  # 3周
        "Y_outlier_quantile": 0.995,  # Q99.5
        "Y_outlier_threshold": float(y_q995)
    },
    "sigma_estimation": {
        "threshold_nearby_range": [0.03, 0.05],  # 3-5%
        "local_binning": {
            "week_bins": 5,
            "bmi_bins": 3
        }
    },
    "data_processing": {
        "use_first_bmi": True,
        "record_level_boundaries": True,
        "survival_library_alignment": True
    }
}

with open(config_path, 'w', encoding='utf-8') as f:
    yaml.dump(config, f, default_flow_style=False, allow_unicode=True)

long_df.to_csv(long_path, index=False, encoding="utf-8-sig")
surv_dat.to_csv(surv_path, index=False, encoding="utf-8-sig")
surv_dat2.to_csv(surv_fit_path, index=False, encoding="utf-8-sig")  # 对齐版
df_report.to_csv(report_path, index=False, encoding="utf-8-sig")

# 展示
print("\n" + "="*60)
print("数据处理完成！")
print("="*60)

print("\nStep1_长表（清洗后逐次检测记录）")
print(long_df.head(10))
print(f"长表总记录数: {len(long_df)}")

print("\nStep1_区间删失表（每人一行）")
print(surv_dat.head(10))
print(f"删失表总人数: {len(surv_dat)}")

print("\n数据质量报告摘要:")
print(f"  总记录条数: {report['总记录条数']}")
print(f"  唯一孕妇人数: {report['唯一孕妇人数']}")
print(f"  区间删失类型分布: {report['区间删失类型分布']}")
print(f"  Y染色体浓度极值过滤: {report['Y染色体浓度极值过滤结果']}")
print(f"  孕周严格递增检查: {report['孕周严格递增检查']}")

print(f"\n输出文件路径:")
print(f"长表: {long_path}")
print(f"事件-删失表: {surv_path}")
print(f"对齐版删失表: {surv_fit_path}")
print(f"报告: {report_path}")
print(f"配置文件: {config_path}")

(long_path, surv_path, report_path)
