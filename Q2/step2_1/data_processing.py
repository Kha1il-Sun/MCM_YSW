import pandas as pd
import numpy as np
import re
from datetime import datetime

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

# 8) 统一Y染色体浓度为比例（0-1）
df["Y_raw"] = pd.to_numeric(df["Y染色体浓度"], errors="coerce")
if (df["Y_raw"] > 1).mean() > 0.2:
    df["Y_frac"] = df["Y_raw"] / 100.0
else:
    df["Y_frac"] = df["Y_raw"]

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

def first_event_time(g):
    g = g.sort_values(["week","date"])
    week_has_hit = g.groupby("week")["Y_frac"].max().reset_index()
    hit_weeks = week_has_hit.loc[week_has_hit["Y_frac"]>=thr, "week"].values
    if len(hit_weeks)>0:
        t = float(np.min(hit_weeks))
        status = 1
    else:
        t = float(np.max(g["week"])) if g["week"].notna().any() else np.nan
        status = 0
    BMI = float(np.nanmean(g["BMI_used"])) if g["BMI_used"].notna().any() else np.nan
    return pd.Series({"event_time": t, "status": status, "BMI": BMI,
                      "last_obs_week": float(np.max(g["week"])) if g["week"].notna().any() else np.nan,
                      "n_records": len(g)})

surv_dat = long_df.groupby("id", as_index=False).apply(first_event_time).reset_index(drop=True)

# 11) 基本报告
report = {
    "总记录条数": int(len(df)),
    "唯一孕妇人数": int(df["孕妇代码"].nunique() if "孕妇代码" in df.columns else 0),
    "有文本孕周的比例": float(df["week_from_text"].notna().mean()),
    "有日期孕周的比例": float(df["week_from_dates"].notna().mean()),
    "文本vs日期孕周差的中位数(周)": float(np.nanmedian(df["week_diff_text_vs_date"])) if df["week_diff_text_vs_date"].notna().any() else np.nan,
    "同日多次检测的样本对数": int((df.groupby(["孕妇代码","检测日期_parsed"]).size()>1).sum()),
    "存在BMI给定列": bool(bmi_given_col is not None),
    "BMI来源分布": {str(k): int(v) for k,v in df["BMI_source"].value_counts(dropna=False).to_dict().items()}
}

# 12) 保存输出
long_path = "step1_long_records.csv"
surv_path = "step1_surv_dat.csv"
report_path = "step1_report.csv"
df_report = pd.DataFrame([report])

long_df.to_csv(long_path, index=False, encoding="utf-8-sig")
surv_dat.to_csv(surv_path, index=False, encoding="utf-8-sig")
df_report.to_csv(report_path, index=False, encoding="utf-8-sig")

# 展示
print("Step1_长表（清洗后逐次检测记录）")
print(long_df.head(30))
print("\nStep1_事件-删失表（每人一行）")
print(surv_dat.head(30))

print(f"\n输出文件路径:")
print(f"长表: {long_path}")
print(f"事件-删失表: {surv_path}")
print(f"报告: {report_path}")

(long_path, surv_path, report_path)
