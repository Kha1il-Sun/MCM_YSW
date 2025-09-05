下面是一份可直接附在仓库里的 **工程说明** 。我把代码框架、目录结构、数学模型、实验流程、复现/验证、容易踩坑与忽视的细节全部写清楚，并与 Step-1 的产物严格对齐（长表、区间删失表、报告、配置、σ 估计等）。其中关键设计点参考了你当前的试验代码与改进方案，以及 Step-1 的数据处理与 README。

---

# 工程说明（Problem 2：BMI 分组与最佳 NIPT 时点优化）

## 0. 目标与原则

 **目标** ：在保证临床可解释性的前提下，依据 **BMI** 对男胎孕妇合理分组，并为每组给出 **最佳 NIPT 时点** ，使 **潜在风险** （早检假阴性、窗口缩短、失败/重抽成本） **最小化** ；同时量化**检测误差**对结果的影响。实现上采用“两条并行通道”（纵向/分位数与区间删失/生存）交叉验证结论，提升稳健性。

**输入依赖（来自 Step-1）**

* `step1_long_records.csv`：逐次检测长表（id, date, week, BMI_used, Y_frac, …）。
* `step1_surv_dat_fit.csv`：对齐版区间删失表（含 `L_fit=-∞`/`R_fit=+∞`）。
* `step1_report.csv`：质量统计与 **阈值邻域 σ** （全局/局部）摘要。
* `step1_config.yaml`：所有阈值、参数与策略记录（便于复现）。

> 注：Step-1 已将达标时间按记录级别构造成 **左/右/区间删失** ，并生成了与生存库接口对齐的 `L_fit/R_fit`；并在阈值邻域（3–5%）估计了全局/局部  **σ** （测量误差）。这些都是第二问优化的关键基座。

---

## 1. 目录结构（建议）

```
Q2/
├── appendix.xlsx                      # 原始数据（保留）
├── README.md                          # 顶层项目说明（建议汇总Step1/Step2的入口与产物）
├── step2_1/                           # 步骤1：数据预处理（保持不变）
│   ├── data_processing.py             # 数据预处理脚本
│   ├── README.md                      # 这里已有详细数据处理说明与指标
│   ├── step1_config.yaml              # 步骤1说明文档
│   ├── step1_long_records.csv         # 长格式记录数据
│   ├── step1_report.csv               # 处理报告
│   ├── step1_surv_dat_fit.csv         # 生存分析拟合数据
│   └── step1_surv_dat.csv             # 生存分析原始数据
└── step2_2/                           # 步骤2：BMI分组与最佳检测时点分析（本次规范化）
    ├── p2.py                          # CLI 入口：读Step1产物→建模→优化→可视化→报告
    ├── config/
    │   ├── step2_config.yaml          # Step2参数：τ、δ、搜索范围、分布假设、bootstrap等
    ├── data/                          # （只读）指向Step1输出，避免复制
    │   ├── step1_long_records.csv     # ➜ ../step2_1/step1_long_records.csv（软链接/复制二选一）
    │   ├── step1_surv_dat_fit.csv     # ➜ ../step2_1/step1_surv_dat_fit.csv
    │   ├── step1_report.csv           # ➜ ../step2_1/step1_report.csv
    │   └── step1_config.yaml          # ➜ ../step2_1/step1_config.yaml
    ├── src/                           # 代码模块化拆分（建议）
    │   ├── io_utils.py                # 读取/校验Step1产物与可选QC
    │   ├── sigma_lookup.py            # σ(t,b)查表/插值（优先局部网格，回退全局）
    │   ├── models_long.py             # 纵向通道：μ(t,b)/p_hit(t,b)
    │   ├── models_fail.py             # 失败通道：q_fail(t,b)（无标签则情景/先验）
    │   ├── models_surv.py             # 生存通道：AFT/PH区间删失拟合，S(t|b)
    │   ├── objective.py               # 风险函数：S版/误差修正版（EC）
    │   ├── optimize.py                # w*(b)数值求解（粗→细）、保序、组内最小化
    │   ├── grouping.py                # WHO起点 + 决策树/DP微调 + 约束（间距/样本量）
    │   ├── validate.py                # GroupKFold/LOPO、时间外推、bootstrap CI
    │   ├── sensitivity.py             # τ/δ/阈值/σ扰动，情景集
    │   ├── report.py                  # 汇总主表/图，写CSV/PNG/TXT；回填配置
    │   └── viz.py                     # 主图（w*曲线+切点）/分位曲线/敏感性热力图
    └── outputs/                       # Step2全部产物（自动生成，纳入.gitignore）
        ├── p2_wstar_curve.csv
        ├── p2_group_recommendation.csv
        ├── main_analysis.png
        ├── sensitivity_analysis.png
        └── p2_report.txt

```

---

## 2. 数学模型与求解框架

### 2.1 两条并行通道（交叉验证思想）

* **纵向/分位数通道**

  用分位数回归（或 GAM/GBR）学习 YY–tt–bb 的条件分布，给出

  μτ(t,b)\mu_\tau(t,b) 与**命中概率**$p_{\text{hit}}(t,b)=\Pr(Y\ge \text{thr\_adj}(t,b))$。

  其中$\text{thr\_adj}(t,b)=0.04 + z_\alpha\,\sigma(t,b)$，$\sigma $来自 Step-1 的阈值邻域估计（优先局部，回退全局）。
* **区间删失/生存通道**

  用 `step1_surv_dat_fit.csv` 直接拟合“达标时间” $T^\*$（左/右/区间删失），试 Log-logistic/Weibull/Log-normal AFT，按 AIC/校准优选，输出$S(t\mid b) $与 $E[T^\*|b]$。该通道天然融入“由低到高跨阈值”的生成机制，对“首次达标”刻画更贴近流程。

> 两通道的 w*(b) 与分组/时点建议彼此对照，若结论一致性高，可信度显著提升；若存在偏差，走敏感性与误差来源剖析。

### 2.2 成功概率与风险函数

* **失败概率** $q_{\text{fail}}(t,b)$：若无真实失败标签，则采用 **情景函数** （早孕周/高 BMI 失败率↑）或以 QC 代理建模（配置化）。不要把“未达标”当“失败”。
* **成功概率（统一表述）**

  $p_{\text{succ}}(t,b)=(1-q_{\text{fail}}(t,b))\cdot p_{\text{hit}}(t,b).$
* **风险函数（期望风险）**

  $\mathcal R(t,b)=w_1\,[1-p_{\text{succ}}(t,b)]+w_2\,\text{Delay}_{[13,27],[\ge 28]}(t)+w_3\,\text{Redraw}(t,b),$
  其中第二项体现题干“12 周内风险低、13–27 周高、≥28 周极高”的惩罚分段；第三项可按 qfailq_{\text{fail}} 计重抽/延期成本。组内风险$ R_g(t)=\mathbb E_{b\in G}[\mathcal R(t,b)]$。令 $t_g^\*=\arg\min_t \mathcal R_g(t)$。
* **生存版本的“早检风险”替代**

  用 $S(t\mid b)=\Pr(T^\*>t) $近似“尚未达标”的概率，得到

  $ R_S(t,b)=w_1\,S(t|b)+\dots$，与上式并行做敏感性。

### 2.3 w*(b) 求解与 BMI 分组

* **w*(b) 数值解** ：先粗网格（如 0.5 周）定位，再二分/Brent 细化至 0.05 周；取满足 $p_{\text{succ}}\ge \tau$ 的 **最左根** ；失败则回退保守策略（如上界）。
* **保序约束** ：w*(b) 经 **等距 BMI 网格的保序回归（Isotonic）** 平滑，确保生理可解释性（BMI↑→更晚）。
* **分组策略（混合）** ：WHO 切点为起点（18.5/25/30），用回归树/一维 DP 在 ±Δ 窗口微调，加入**最小切点间距**与**组内最小样本量**硬约束，保证稳定与可解释性。

---

## 3. 代码框架（Python 主要接口）

> 只给**主要代码框架**与函数签名，便于你实现/替换；细节请保持你现有习惯。

### 3.1 I/O 与配置

```python
# src/io_utils.py
def load_step1_products(datadir):
    long_df = pd.read_csv(f"{datadir}/step1_long_records.csv")
    surv_df = pd.read_csv(f"{datadir}/step1_surv_dat_fit.csv")  # 含 L_fit/R_fit
    report  = pd.read_csv(f"{datadir}/step1_report.csv")
    cfg1    = yaml.safe_load(open(f"{datadir}/step1_config.yaml"))
    return long_df, surv_df, report, cfg1

# src/sigma_lookup.py
def build_sigma_lookup(report, sigma_grid_path=None):
    # 若存在局部 σ 网格文件则优先；否则用 report 的全局 σ
    # 返回函数 sigma(t, b)
    ...
```

（Step-1 已在 README/代码中说明 σ 的估计与对齐约定，可据此查表/回退。 ）

### 3.2 纵向通道（分位数/GAM/GBR）

```python
# src/models_long.py
def fit_quantile_models(long_df, tau, features="poly+interactions", model="GBR|GAM"):
    # 返回 μ_tau(t,b) 预测器或 p_hit(t,b)=Pr(Y>=thr_adj(t,b))
    ...

def make_p_hit(mu_or_model, sigma_func, alpha):
    # p_hit(t,b) = Phi((mu(t,b) - thr)/sigma(t,b)) 或由分类器直接给出
    ...
```

（你已实现了 GBR/GAM 与特征工程、交叉验证接口，此处沿用并将阈值改为**动态 thr_adj** 接口。）

### 3.3 失败通道（有/无标签）

```python
# src/models_fail.py
def fit_q_fail(long_df, qc_cols=None):
    # 若有失败/no-call 标签就逻辑回归/GAM；否则返回配置化情景函数
    # q_fail(t,b) in [0,1]
    ...

def q_fail_scenario(t, b, params):
    # logistic 族；高 BMI、早孕周→更高失败率
    ...
```

（不要用 “Y<thr_adj” 伪造失败标签；失败与未达标是两回事。）

### 3.4 生存通道（AFT/PH）

```python
# src/models_surv.py
from lifelines import LogLogisticAFTFitter, WeibullAFTFitter, LogNormalAFTFitter

def fit_aft_models(surv_df):
    # 尝试多个分布，按AIC/校准选最优；返回 best_model
    ...

def S_of(best_model):
    # 返回 S(t|b) 的可调用接口
    ...
```

（依赖 Step-1 的 `L_fit/R_fit` 左/右删失约定，直接 fit_interval_censoring。）

### 3.5 风险函数与最优时点

```python
# src/objective.py
def p_succ(t, b, p_hit_func, q_fail_func):
    return (1 - q_fail_func(t,b)) * p_hit_func(t,b)

def risk_S(t, b, S_func, w_delay, w_redraw):
    # 用 S(t|b) 版本的早检风险
    ...

def risk_EC(t, b, p_succ_func, w_delay, w_redraw):
    # 用误差修正版本（推荐）
    ...

# src/optimize.py
def solve_w_star_for_b(b, risk_func, t_range, tol=0.05):
    # 粗网格 → 二分/Brent 最左根；失败时回退策略
    ...

def monotone_smooth_wstar(bs, wstars):
    # BMI 方向保序回归（isotonic）
    ...
```

### 3.6 分组与推荐

```python
# src/grouping.py
def find_bmi_cuts(wstar_curve, who_cuts, method="hybrid", delta=2.0,
                  min_group_n=30, search="tree|dp"):
    # WHO 为起点，决策树/DP 微调，强制最小切点间距与组内样本量
    # 返回 cuts, groups, per-group t*
    ...
```

（你的“混合方法”与最小间距策略在试验代码中已有雏形，保留并加硬约束。）

### 3.7 验证、敏感性与报告

```python
# src/validate.py
def group_kfold_by_id(long_df, k=5):
    # GroupKFold/Leave-One-Patient-Out，防同人泄漏
    ...

def bootstrap_ci(ids, fit_and_solve_fn, B=500):
    # 抽人头 boot；输出 t* 的CI与切点稳定性
    ...

# src/sensitivity.py
def sweep_params(taus, deltas, thresholds, sigmas, scenario_sets):
    # 产出 (τ, δ, thr, σ) → 组推荐时点/风险 的矩阵，便于画热力图/龙卷风图
    ...

# src/report.py
def export_tables_and_plots(pred_curve, groups_tbl, sens, valid, cfg, outdir):
    # 保存 CSV/PNG/TXT，并把 step2_config.yaml 回写到报告头
    ...
```

（这些模块化点与现有说明/代码的功能对应，补足验证与输出落盘。）

---

## 4. 运行流程（p2.py）

1. 读取 Step-1 产物与 step2 配置；构造 `sigma(t,b)` 查表（优先局部，回退全局）。
2. **纵向通道** ：拟合 μ/分位模型 → 构造 `p_hit(t,b)`； **失败通道** ：`q_fail(t,b)`（标签/情景）。
3. **生存通道** ：AFT/PH 拟合，选择最优分布，得 `S(t|b)`。
4. 在两个版本的风险函数下（S/EC），求 `w*(b)`（粗→细），并做 **保序** 。
5. 以 WHO 切点为起点，进行 **混合/DP 微调** ，得到最终 BMI 分组与每组推荐时点（组内风险最小化）。
6. **验证** （GroupKFold/LOPO、时间外推）、 **敏感性** （τ/δ/σ/阈值扰动）、 **bootstrap CI** 。
7. 导出主表、曲线图、敏感性图与综合报告，回填全部配置与版本。

---

## 5. 容易踩的坑（务必避免）

1. **把“未达标”误当“失败/no-call”** ：两者语义不同。失败模型没有标签时只能走情景/先验或 QC 代理；不要用 `Y<thr_adj` 训练失败分类器。
2. **忽略区间删失** ：把“首次观测到达标周”当确切事件会低估不确定性；必须使用 Step-1 给的 `L_fit/R_fit` 做 AFT/PH。
3. **固定阈值** ：不考虑 σ 会高估早检可靠度；应使用 `thr_adj(t,b)=0.04 + z·σ(t,b)`。
4. **训练-测试泄漏** ：按记录切分会让同一 id 同时出现在训练/测试；必须 GroupKFold/LOPO。
5. **w*(b) 非单调** ：直接逐点求解会有噪声抖动；必须做 **保序** 再分组。
6. **决策树切点不稳** ：增加**最小间距**与**最小样本量**硬约束；备选一维 DP 微调。
7. **粗网格终止** ：仅用 0.5 周网格会产生阶梯误差；粗→细二分/Brent 收敛至 0.05 周量级。
8. **σ 局部估计稀疏** ：分箱样本少会不稳定；对局部方差做 **收缩/回填** （向全局 σ 收敛），并记录有效样本数。
9. **生存分布盲选** ：Weibull/Log-logistic/Log-normal 的尾部行为不同；用 AIC 与校准图共同选择。
10. **极端 BMI/孕周外推** ：超出训练覆盖范围时应 clip 并提示“外推”；画图/报告中明确覆盖域。

---

## 6. 容易忽视的细节（建议都做）

* **首检 BMI** ：分层统一以**首检**为基线，不用平均值（Step-1 已输出 `BMI_base` 可直接用）。
* **同日多检** ：已在 Step-1 取 **同日最大 Y** ；但要在方法里写明该近似会略低估“当日内达标时间”。
* **异常过滤** ：Y 极值用**分位数法（Q99.5）**优于固定 0.20，并把阈值写回报告，保证复现。
* **配置回填** ：把所有关键参数（τ、δ、权重、σ 收缩 λ、DP 惩罚、搜索范围）写入 `step2_config.yaml` 并在报告头部回显。
* **多版本结果对照** ：同时输出 S 版与 EC 版的 w*(b)/t*_组；若差异小，说明鲁棒；若差异大，解释来源（σ、失败情景、分布选择）。
* **图表可读性** ：在主图上同时标注 w* 曲线、保序曲线、最终切点与组内推荐时点（带 CI），并在图注说明 τ、δ、thr_adj 的定义。
* **统计功效** ：每组样本量报告（n）、删失类型比例、时间覆盖（min–max），避免过小样本组的过拟合。

---

## 7. 输出与论文落地

* **主表** ：

  `p2_group_recommendation.csv`：每个 BMI 组的区间、推荐时点 tg\*t_g^\*、组内样本量、S/EC 两版本一致性、相对统一时点的 **风险下降幅度** 。

* **曲线与图** ：

  `main_analysis.png`：w*(b) 原始/保序曲线 + 切点；分位曲线 + 阈值线；组间推荐时点柱状图；BMI 分布与切点。

* **敏感性** ：

  `sensitivity_analysis.png`：τ/δ/σ/阈值扰动的热力图/龙卷风图。

* **复现报告** ：

  `p2_report.txt`：写清配置、数据版本、删失分布、σ 统计、模型选择依据、分组与时点建议与 CI、敏感性与稳健性结论。

---

## 8. 最小可运行主流程（p2.py 伪代码）

```python
def main():
    # 1) 读入数据与配置
    long_df, surv_df, rep, cfg1 = load_step1_products("data")
    cfg2 = yaml.safe_load(open("config/step2_config.yaml"))
    sigma = build_sigma_lookup(rep, cfg2.get("sigma_grid_path"))

    # 2) 纵向通道：μ/分位 & p_hit
    q_model = fit_quantile_models(long_df, tau=cfg2["tau"], model=cfg2["quantile_model"])
    p_hit  = make_p_hit(q_model, sigma_func=sigma, alpha=cfg2["alpha"])

    # 3) 失败通道：q_fail（标签/情景）
    q_fail = fit_q_fail(long_df, qc_cols=cfg2.get("qc_cols"))

    # 4) 生存通道：AFT/PH
    aft_best = fit_aft_models(surv_df)
    S_func   = S_of(aft_best)

    # 5) 求 w*(b)（S版与EC版），做保序
    wstar_curve_EC = solve_w_star_curve(p_succ=lambda t,b: p_succ(t,b,p_hit,q_fail), ...)
    wstar_curve_S  = solve_w_star_curve_S(S_func, ...)
    wstar_curve    = monotone_smooth_wstar(...)

    # 6) BMI 分组与每组推荐时点
    cuts, groups_tbl = find_bmi_cuts(wstar_curve, who_cuts=cfg2["who_cuts"], ...)

    # 7) 验证、敏感性、bootstrap CI
    valid = group_kfold_by_id(long_df, ...)
    sens  = sweep_params(...)

    # 8) 导出表格、图与综合报告
    export_tables_and_plots(wstar_curve, groups_tbl, sens, valid, {**cfg1, **cfg2}, "outputs")
```

---

### 结语

以上工程说明把 **p2** 的重写从“代码骨架—数学原理—数据接口—优化与验证—可复现性—落地图表”一条龙打通，并与 Step-1 的区间删失结构、σ 估计与 README 完全对齐。你按此拆分模块推进，就能在**统计一致性、工程稳健性、临床可解释性**三方面同时达标。
