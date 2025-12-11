"""
Replicate Deming (2009) Table 2:
Sibling Differences in Pre-Treatment Covariates, by Preschool Status

Assumptions:
- Input file: deming_cleaned_data.csv
- File already contains all pre-treatment covariates and FE variables created
  according to Deming_2008_0217.do, including:

  MotherID, PreK_FE, HS2_FE90, Pre2_FE90, Age2_Mo90,
  Attrit, Sample90_2,
  Res_0to3, HealthCond_before, logBW, VLow_BW, LogInc_0to3, LogIncAt3,
  FirstBorn, Male, Age2_Yr104, PPVTat3, HOME_Pct_0to3,
  Moth_HrsWorked_BefBirth, Moth_HrsWorked_0to1,
  Father_HH_0to3, GMom_0to3, MomCare, RelCare, NonRelCare,
  Breastfed, Moth_Smoke_BefBirth, Alc_BefBirth,
  Doctor_0to3, Dentist_0to3, Moth_WeightChange,
  Illness_1stYr, Premature, Insurance_0to3, Medicaid_0to3,
  PreTreatIndex.

The script runs sibling fixed-effects regressions with cluster-robust
standard errors (clustered at MotherID), as in Stata's:

  xtreg y HS2_FE90 Pre2_FE90 Male FirstBorn Age2_Mo90, fe vce(cluster MotherID)

except for Attrit, which uses a slightly different sample and without controls,
and PPVTat3, which uses Sample90_2 but also includes Male, FirstBorn, Age2_Mo90.

Output:
  - table2_results.csv  (one row per variable, matching columns 1–4 of Table 2)
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm

# 读取你之前的清洗数据
df = pd.read_csv("deming_cleaned_data.csv")

# ------------------------------------------------------------
# 1. Age2_Yr86 ... Age2_Yr104（月份转成年）
# ------------------------------------------------------------
years = [86, 88, 90, 92, 94, 96, 98, 100, 102, 104]
for x in years:
    mo = f"Age2_Mo{x}"
    yr = f"Age2_Yr{x}"
    if mo in df.columns and yr not in df.columns:
        df[yr] = np.where(df[mo].notna(), np.floor(df[mo] / 12.0), np.nan)

# 同理 94/96/98/100/102 可能后面用到
for x in [94, 96, 98, 100, 102]:
    mo = f"Age2_Mo{x}"
    yr = f"Age2_Yr{x}"
    if mo in df.columns and yr not in df.columns:
        df[yr] = np.where(df[mo].notna(), np.floor(df[mo] / 12.0), np.nan)

# ------------------------------------------------------------
# 2. Res_0to3：0–3 岁是否一直和妈妈同住
# ------------------------------------------------------------
for x in range(79, 91):
    rescol = f"Res{x}"
    if rescol in df.columns:
        df[f"MomHH{x}"] = np.where(
            df[rescol] == 1, 1.0,
            np.where(df[rescol].notna(), 0.0, np.nan)
        )

df["Res_0to3"] = np.nan

def set_rowmin(age, start, end):
    cols = [f"MomHH{y}" for y in range(start, end + 1)]
    mask = df["Age2_Yr104"] == age
    df.loc[mask, "Res_0to3"] = df.loc[mask, cols].min(axis=1)

df.loc[df["Age2_Yr104"] == 14, "Res_0to3"] = df.loc[df["Age2_Yr104"] == 14, "MomHH90"]
set_rowmin(15, 89, 90)
set_rowmin(16, 88, 90)
set_rowmin(17, 87, 90)
set_rowmin(18, 86, 89)
set_rowmin(19, 85, 88)
set_rowmin(20, 84, 87)
set_rowmin(21, 83, 86)
set_rowmin(22, 82, 85)
set_rowmin(23, 81, 84)
set_rowmin(24, 80, 83)
set_rowmin(25, 79, 82)
set_rowmin(26, 79, 81)
set_rowmin(27, 79, 80)
mask28 = df["Age2_Yr104"] == 28
df.loc[mask28, "Res_0to3"] = df.loc[mask28, "MomHH79"]

# ------------------------------------------------------------
# 3. HealthCond_before：5 岁前是否有健康限制
# ------------------------------------------------------------
health_vars = [
    "Brain", "Hyper", "Asthma", "Resp", "Speech", "Deaf",
    "Blind", "Disturb", "Allergy", "Crippled", "Retard", "OtherLim"
]
years_health = [86, 88, 90]

for y in years_health:
    cols = [f"{h}{y}" for h in health_vars if f"{h}{y}" in df.columns]
    tmp = (df[cols].gt(0).any(axis=1)).astype(float)
    tmp[df[cols].isna().all(axis=1)] = np.nan
    df[f"HealthAny{y}"] = tmp

df["HealthCond_before"] = 0.0
for y in years_health:
    mask = (df[f"HealthAny{y}"] == 1) & (df[f"Age2_Mo{y}"] < 60)
    df.loc[mask, "HealthCond_before"] = 1.0

df.loc[df["Res_0to3"].isna(), "HealthCond_before"] = np.nan

# ------------------------------------------------------------
# 4. 低出生体重：Low_BW / VLow_BW / logBW
# ------------------------------------------------------------
df["Low_BW"] = np.where(
    df["BirthWeight"].notna(),
    (df["BirthWeight"] < 88).astype(float),
    np.nan
)
df["VLow_BW"] = np.where(
    df["BirthWeight"].notna(),
    (df["BirthWeight"] < 53).astype(float),
    np.nan
)
df["logBW"] = np.where(
    df["BirthWeight"] > 0,
    np.log(df["BirthWeight"]),
    np.nan
)

# ------------------------------------------------------------
# 5. Attrit：19 岁前 attrition
# ------------------------------------------------------------
df["Attrit"] = np.where(
    df["YA_LastInterview"] == 2004, 0,
    np.where(df["YA_LastInterview"].notna(), 1, np.nan)
)

df.loc[
    (df["Attrit"] == 1) & (df["YA_LastInterview"] == 2002) &
    (df["Age2_Yr102"] >= 19),
    "Attrit"
] = 0
df.loc[
    (df["Attrit"] == 1) & (df["YA_LastInterview"] == 2000) &
    (df["Age2_Yr100"] >= 19),
    "Attrit"
] = 0

for yr in [94, 96, 98]:
    df.loc[
        (df["Attrit"] == 1) &
        (df["YA_LastInterview"] == 1900 + yr) &
        (df[f"Age2_Yr{yr}"] >= 19),
        "Attrit"
    ] = 0

# ------------------------------------------------------------
# 6. Sample90_2：兄弟 FE 分析的样本
# ------------------------------------------------------------
df["Sample90_2"] = np.nan

mask = (
    df["HS2_FE90"].notna() &
    (df["SampleID"] != 12) &
    (df["Attrit"] == 0) &
    (df["Age2_Yr104"] >= 19)
)
df.loc[mask, "Sample90_2"] = 1.0

mask2 = (
    df["HS2_FE90"].notna() &
    (df["Attrit"] == 0) &
    (df["Sample90_2"] != 1) &
    (df["DOB_Yr_Child"] == 1985) &
    (df["DOB_Mo_Child"] < 8)
)
df.loc[mask2, "Sample90_2"] = 1.0

# ------------------------------------------------------------
# 7. LogInc_0to3 / LogIncAt3
# ------------------------------------------------------------
for x in range(78, 90 + 1):
    net = f"NetFamInc{x}"
    if net in df.columns:
        df[f"Income{x}"] = df[net]

df["Income_0to3"] = np.nan

def set_income(age, start, end):
    cols = [f"Income{y}" for y in range(start, end + 1)]
    mask = df["Age2_Yr104"] == age
    df.loc[mask, "Income_0to3"] = df.loc[mask, cols].mean(axis=1, skipna=True)

df.loc[df["Age2_Yr104"] == 14, "Income_0to3"] = df.loc[df["Age2_Yr104"] == 14, "Income90"]

set_income(15, 89, 90)
set_income(16, 88, 90)
set_income(17, 87, 90)
set_income(18, 86, 89)
set_income(19, 85, 88)
set_income(20, 84, 87)
set_income(21, 83, 86)
set_income(22, 82, 85)
set_income(23, 81, 84)
set_income(24, 80, 83)
set_income(25, 79, 82)
set_income(26, 79, 81)
set_income(27, 78, 80)
set_income(28, 78, 79)

mask29 = df["Age2_Yr104"] == 29
df.loc[mask29, "Income_0to3"] = df.loc[mask29, "NetFamInc78"]

df["LogInc_0to3"] = np.where(df["Income_0to3"] > 0,
                             np.log(df["Income_0to3"]), np.nan)

df["IncAt3"] = np.nan
age_year_map = {
    29: 78, 28: 79, 27: 80, 26: 81,
    25: 82, 24: 83, 23: 84, 22: 85,
    21: 86, 20: 87, 19: 88, 18: 89, 17: 90
}
for age, year in age_year_map.items():
    mask = df["Age2_Yr104"] == age
    df.loc[mask, "IncAt3"] = df.loc[mask, f"NetFamInc{year}"]

df["LogIncAt3"] = np.where(df["IncAt3"] > 0, np.log(df["IncAt3"]), np.nan)

# ------------------------------------------------------------
# 8. FirstBorn / Male
# ------------------------------------------------------------
df["FirstBorn"] = np.where(df["BirthOrder"].notna(),
                           (df["BirthOrder"] == 1).astype(float),
                           np.nan)
df["Male"] = np.where(df["Sex_Child"].notna(),
                      (df["Sex_Child"] == 1).astype(float),
                      np.nan)

# ------------------------------------------------------------
# 9. PPVTat3（3 岁 PPVT 原始分）
# ------------------------------------------------------------
df["PPVTat3"] = np.nan

mask = (df["PPVTAge86"].between(36, 46)) & df["PPVT_Raw86"].notna()
df.loc[mask, "PPVTat3"] = df.loc[mask, "PPVT_Raw86"]

mask = (df["PPVTAge88"].between(36, 46)) & df["PPVT_Raw88"].notna() & df["PPVTat3"].isna()
df.loc[mask, "PPVTat3"] = df.loc[mask, "PPVT_Raw88"]

mask = (df["PPVTAge90"].between(36, 46)) & df["PPVT_Raw90"].notna() & df["PPVTat3"].isna()
df.loc[mask, "PPVTat3"] = df.loc[mask, "PPVT_Raw90"]

# ------------------------------------------------------------
# 10. HOME_Pct_0to3
# ------------------------------------------------------------
df["HOME_Pct_0to3"] = np.nan

mask = (df["Age2_Yr104"] <= 19) & (df["Age2_Yr104"] >= 16)
df.loc[mask, "HOME_Pct_0to3"] = df.loc[mask, ["HOME_Pct86", "HOME_Pct88"]].mean(axis=1, skipna=True)

mask = (df["Age2_Yr104"] <= 15) & (df["Age2_Yr104"] >= 14)
temp = df.loc[mask, ["HOME_Pct88", "HOME_Pct90"]].mean(axis=1, skipna=True)
df.loc[mask, "HOME_Pct_0to3"] = temp

mask = (df["Age2_Yr104"] >= 20) & (df["Age2_Yr104"] <= 21)
df.loc[mask, "HOME_Pct_0to3"] = df.loc[mask, "HOME_Pct86"]

# ------------------------------------------------------------
# 11. 母亲工作：Moth_HrsWorked_BefBirth / Moth_HrsWorked_0to1
# ------------------------------------------------------------
before_cols = [
    c for c in df.columns
    if c.startswith("Moth_HrsWorked") and "Before" in c
] + [
    c for c in df.columns
    if c.startswith("Moth_Hrs_Worked") and "Before" in c
]
if before_cols:
    temp = df[before_cols].mean(axis=1, skipna=True)
    df["Moth_HrsWorked_BefBirth"] = temp / 13.0

qtr_cols = [c for c in df.columns if c.startswith("Moth_HrsWorked") and c.endswith("_Qtr")]
if qtr_cols:
    df["Moth_HrsWorked_0to3"] = df[qtr_cols].mean(axis=1, skipna=True)

cols_0to1 = [f"Moth_HrsWorked_{i}_Avg" for i in range(1, 5) if f"Moth_HrsWorked_{i}_Avg" in df.columns]
if cols_0to1:
    df["Moth_HrsWorked_0to1"] = df[cols_0to1].mean(axis=1, skipna=True)

# ------------------------------------------------------------
# 12. Father_HH_0to3
# ------------------------------------------------------------
df["Father_HH_0to3"] = np.nan

def set_father(age, cols):
    mask = df["Age2_Yr104"] == age
    df.loc[mask, "Father_HH_0to3"] = df.loc[mask, cols].mean(axis=1, skipna=True)

set_father(14, ["Father_HH90", "Father_HH92", "Father_HH93"])
set_father(15, ["Father_HH89", "Father_HH90"])
set_father(16, ["Father_HH88", "Father_HH89", "Father_HH90"])
set_father(17, ["Father_HH87", "Father_HH88", "Father_HH89", "Father_HH90"])
set_father(18, ["Father_HH86", "Father_HH87", "Father_HH88", "Father_HH89"])
set_father(19, ["Father_HH85", "Father_HH86", "Father_HH87", "Father_HH88"])
set_father(20, ["Father_HH84", "Father_HH85", "Father_HH86", "Father_HH87"])
set_father(21, ["Father_HH84", "Father_HH85", "Father_HH86"])
set_father(22, ["Father_HH84", "Father_HH85"])

mask = df["Age2_Yr104"] == 23
df.loc[mask, "Father_HH_0to3"] = df.loc[mask, "Father_HH84"]

# ------------------------------------------------------------
# 13. GMom_0to3（外婆在家）
# ------------------------------------------------------------
for x in range(79, 91):
    col = f"Grandmother{x}"
    if col in df.columns:
        df[f"GMom{x}"] = np.where(
            df[col] == 1, 1.0,
            np.where(df[col].notna(), 0.0, np.nan)
        )

df["GMom_0to3"] = np.nan

def set_gmom(age, start, end):
    cols = [f"GMom{y}" for y in range(start, end + 1)]
    mask = df["Age2_Yr104"] == age
    df.loc[mask, "GMom_0to3"] = df.loc[mask, cols].mean(axis=1, skipna=True)

df.loc[df["Age2_Yr104"] == 14, "GMom_0to3"] = df.loc[df["Age2_Yr104"] == 14, "GMom90"]
set_gmom(15, 89, 90)
set_gmom(16, 88, 90)
set_gmom(17, 87, 90)
set_gmom(18, 86, 89)
set_gmom(19, 85, 88)
set_gmom(20, 84, 87)
set_gmom(21, 83, 86)
set_gmom(22, 82, 85)
set_gmom(23, 81, 84)
set_gmom(24, 80, 83)
set_gmom(25, 79, 82)
set_gmom(26, 79, 81)
set_gmom(27, 79, 80)
mask28 = df["Age2_Yr104"] == 28
df.loc[mask28, "GMom_0to3"] = df.loc[mask28, "GMom79"]

# ------------------------------------------------------------
# 14. MomCare / RelCare / NonRelCare, ages 0–3
# ------------------------------------------------------------
# rename: ChildCare_1stYr -> ChildCare_1_Yr, etc.
if "ChildCare_1stYr" in df.columns:
    df["ChildCare_1_Yr"] = df["ChildCare_1stYr"]
if "ChildCare_2ndYr" in df.columns:
    df["ChildCare_2_Yr"] = df["ChildCare_2ndYr"]
if "ChildCare_3rdYr" in df.columns:
    df["ChildCare_3_Yr"] = df["ChildCare_3rdYr"]
if "ChildCare_Type_1stYr" in df.columns:
    df["ChildCare_Type_1_Yr"] = df["ChildCare_Type_1stYr"]
if "ChildCare_Type_2ndYr" in df.columns:
    df["ChildCare_Type_2_Yr"] = df["ChildCare_Type_2ndYr"]
if "ChildCare_Type_3rdYr" in df.columns:
    df["ChildCare_Type_3_Yr"] = df["ChildCare_Type_3rdYr"]

for y in [1, 2, 3]:
    tcol = f"ChildCare_Type_{y}_Yr"
    ccol = f"ChildCare_{y}_Yr"
    if tcol in df.columns and ccol in df.columns:
        df[f"RelCare_{y}_Yr"] = np.where(
            df[tcol].notna() & (df[tcol] <= 10), 1.0,
            np.where(df[ccol].notna(), 0.0, np.nan)
        )
        df[f"NonRelCare_{y}_Yr"] = np.where(
            df[tcol].notna() & (df[tcol] > 10), 1.0,
            np.where(df[ccol].notna(), 0.0, np.nan)
        )
        df[f"MomCare_{y}_Yr"] = np.where(
            (df[f"RelCare_{y}_Yr"] == 0) & (df[f"NonRelCare_{y}_Yr"] == 0), 1.0,
            np.where(
                df[f"RelCare_{y}_Yr"].notna() & df[f"NonRelCare_{y}_Yr"].notna(),
                0.0,
                np.nan
            )
        )

rel_cols = [c for c in df.columns if c.startswith("RelCare_") and c.endswith("_Yr")]
nonrel_cols = [c for c in df.columns if c.startswith("NonRelCare_") and c.endswith("_Yr")]
momcare_cols = [c for c in df.columns if c.startswith("MomCare_") and c.endswith("_Yr")]

df["RelCare"] = df[rel_cols].mean(axis=1, skipna=True) if rel_cols else np.nan
df["NonRelCare"] = df[nonrel_cols].mean(axis=1, skipna=True) if nonrel_cols else np.nan
df["MomCare"] = df[momcare_cols].mean(axis=1, skipna=True) if momcare_cols else np.nan

# ------------------------------------------------------------
# 15. Alc_BefBirth / Doctor_0to3 / Dentist_0to3
# ------------------------------------------------------------
df["Alc_BefBirth"] = np.where(
    df["Freq_Alc_BefBirth"].notna(),
    (df["Freq_Alc_BefBirth"] >= 3).astype(float),
    np.nan
)

for name in ["Doctor", "Dentist"]:
    tmp_col = f"{name}_temp"
    last86 = f"Last_{name}86"
    last88 = f"Last_{name}88"
    last90 = f"Last_{name}90"

    df[tmp_col] = np.nan

    mask = (df["Age2_Yr104"] <= 19) & (df["Age2_Yr104"] >= 16)
    df.loc[mask, tmp_col] = df.loc[mask, [last86, last88]].mean(axis=1, skipna=True)

    mask = (df["Age2_Yr104"] <= 15) & (df["Age2_Yr104"] >= 14)
    temp = df.loc[mask, [last88, last90]].mean(axis=1, skipna=True)
    df.loc[mask, tmp_col] = temp

    mask = (df["Age2_Yr104"] >= 20) & (df["Age2_Yr104"] <= 21)
    df.loc[mask, tmp_col] = df.loc[mask, last86]

df["Doctor_0to3"] = np.where(
    df["Doctor_temp"] <= 2, 1.0,
    np.where(df["Doctor_temp"] > 2, 0.0, np.nan)
)
df["Dentist_0to3"] = np.where(
    df["Dentist_temp"] < 7, 1.0,
    np.where(df["Dentist_temp"] == 7, 0.0, np.nan)
)

df.drop(columns=["Doctor_temp", "Dentist_temp"], inplace=True)

# ------------------------------------------------------------
# 16. Premature / Insurance_0to3 / Medicaid_0to3
# ------------------------------------------------------------
df["Premature"] = np.where(
    df["BornEarlyorLate"] == 1, 1.0,
    np.where(df["BornOnTime"].isna(), np.nan, 0.0)
)

for var in ["Insurance", "Medicaid"]:
    temp_col = f"{var}_0to3"
    df[temp_col] = np.nan

    mask = (df["Age2_Yr104"] <= 19) & (df["Age2_Yr104"] >= 16)
    df.loc[mask, temp_col] = df.loc[mask, [f"{var}86", f"{var}88"]].mean(axis=1, skipna=True)

    mask = (df["Age2_Yr104"] <= 15) & (df["Age2_Yr104"] >= 14)
    temp = df.loc[mask, [f"{var}88", f"{var}90"]].mean(axis=1, skipna=True)
    df.loc[mask, temp_col] = temp

    mask = (df["Age2_Yr104"] >= 20) & (df["Age2_Yr104"] <= 21)
    df.loc[mask, temp_col] = df.loc[mask, f"{var}86"]

# ------------------------------------------------------------
# 17. PreTreatIndex：协变量标准化后取平均再标准化
# ------------------------------------------------------------
covariates = [
    "Res_0to3", "HealthCond_before", "logBW", "LogInc_0to3", "LogIncAt3",
    "FirstBorn", "Male", "Age2_Yr104", "HOME_Pct_0to3", "Moth_HrsWorked_BefBirth",
    "Moth_HrsWorked_0to1", "Father_HH_0to3", "GMom_0to3", "MomCare", "RelCare",
    "NonRelCare", "Moth_Smoke_BefBirth", "Alc_BefBirth", "Breastfed", "Doctor_0to3",
    "Dentist_0to3", "Moth_WeightChange", "Illness_1stYr", "Premature",
    "Insurance_0to3", "Medicaid_0to3"
]

for x in covariates:
    cv = f"{x}_CV"
    df[cv] = np.nan
    mask = df["Sample90_2"] == 1
    vals = df.loc[mask, x]
    mean = vals.mean()
    std = vals.std(ddof=0)
    df.loc[mask, cv] = (vals - mean) / std if std > 0 else np.nan

# 需要反号的协变量
rev = [
    "HealthCond_before_CV", "Male_CV", "Age2_Yr104_CV", "GMom_0to3_CV",
    "MomCare_CV", "RelCare_CV", "Moth_Smoke_BefBirth_CV", "Alc_BefBirth_CV",
    "Illness_1stYr_CV", "Premature_CV", "Medicaid_0to3_CV"
]
for v in rev:
    if v in df.columns:
        df[v] = -df[v]

cv_cols = [c for c in df.columns if c.endswith("_CV")]
df["PreTreatIndex_temp"] = df[cv_cols].mean(axis=1, skipna=True)

vals = df.loc[df["Sample90_2"] == 1, "PreTreatIndex_temp"]
mean = vals.mean()
std = vals.std(ddof=0)
df["PreTreatIndex"] = (df["PreTreatIndex_temp"] - mean) / std
df.drop(columns=["PreTreatIndex_temp"], inplace=True)

# ------------------------------------------------------------
# 保存新的数据：专门给 Table 2 用
# ------------------------------------------------------------
df.to_csv("deming_table2_ready.csv", index=False)
# print("已生成 deming_table2_ready.csv，包含 Table 2 所需全部协变量。")








"""
Replicate Deming (2009) Table 2:
Sibling Differences in Pre-Treatment Covariates, by Preschool Status

Input:
  - deming_table2_ready.csv  （已经在 build_table2_vars.py 里把所有变量造好）

Output:
  - table2_results.csv
"""

# ------------------------------------------------------------
# 1. 读入数据 & 检查变量
# ------------------------------------------------------------
data_path = "deming_table2_ready.csv"
df = pd.read_csv(data_path)

required_cols = [
    "MotherID", "PreK_FE", "HS2_FE90", "Pre2_FE90", "Age2_Mo90",
    "Attrit", "Sample90_2",
    "Res_0to3", "HealthCond_before", "logBW", "VLow_BW",
    "LogInc_0to3", "LogIncAt3", "FirstBorn", "Male", "Age2_Yr104",
    "PPVTat3", "HOME_Pct_0to3",
    "Moth_HrsWorked_BefBirth", "Moth_HrsWorked_0to1",
    "Father_HH_0to3", "GMom_0to3", "MomCare", "RelCare", "NonRelCare",
    "Breastfed", "Moth_Smoke_BefBirth", "Alc_BefBirth",
    "Doctor_0to3", "Dentist_0to3", "Moth_WeightChange",
    "Illness_1stYr", "Premature", "Insurance_0to3", "Medicaid_0to3",
    "PreTreatIndex",
    "DOB_Yr_Child", "DOB_Mo_Child", "SampleID"
]
missing = [c for c in required_cols if c not in df.columns]
if missing:
    raise ValueError(f"这些变量在数据里找不到，请先在 Python/Stata 中生成: {missing}")

# ------------------------------------------------------------
# 2. 通用 fixed-effects 回归: y ~ HS2_FE90 + Pre2_FE90 (+controls) + Mother FE
# ------------------------------------------------------------
def fe_regression(data, y_var, add_controls=True):
    """
    data: 已经筛好样本的 DataFrame
    y_var: 因变量名
    add_controls: 是否加入 Male, FirstBorn, Age2_Mo90

    返回: (coef_HS, se_HS, coef_Pre, se_Pre, N)
    """
    use = data.dropna(subset=[y_var, "HS2_FE90", "Pre2_FE90", "MotherID"])
    if add_controls:
        use = use.dropna(subset=["Male", "FirstBorn", "Age2_Mo90"])

    if use.empty:
        return np.nan, np.nan, np.nan, np.nan, 0

    group = use.groupby("MotherID")

    y_tilde = use[y_var] - group[y_var].transform("mean")
    hs_tilde = use["HS2_FE90"] - group["HS2_FE90"].transform("mean")
    pre_tilde = use["Pre2_FE90"] - group["Pre2_FE90"].transform("mean")

    X_list = [hs_tilde, pre_tilde]

    if add_controls:
        male_tilde = use["Male"] - group["Male"].transform("mean")
        fb_tilde = use["FirstBorn"] - group["FirstBorn"].transform("mean")
        age_tilde = use["Age2_Mo90"] - group["Age2_Mo90"].transform("mean")
        X_list.extend([male_tilde, fb_tilde, age_tilde])

    X = pd.concat(X_list, axis=1)
    X.columns = [
        col for col in ["HS2_FE90", "Pre2_FE90", "Male", "FirstBorn", "Age2_Mo90"]
        if (not add_controls and col in ["HS2_FE90", "Pre2_FE90"]) or add_controls
    ]

    # within estimator, 不要截距
    model = sm.OLS(y_tilde, X)
    res = model.fit(cov_type="cluster", cov_kwds={"groups": use["MotherID"]})

    coef_HS = res.params["HS2_FE90"]
    se_HS = res.bse["HS2_FE90"]
    coef_Pre = res.params["Pre2_FE90"]
    se_Pre = res.bse["Pre2_FE90"]
    n_obs = int(res.nobs)

    return coef_HS, se_HS, coef_Pre, se_Pre, n_obs

# ------------------------------------------------------------
# 3. Attrited（特殊样本 & 无控制变量）
# ------------------------------------------------------------
# 19 岁及以上，或者 1985 年出生且 2004 年 9 月前满 19 岁
cond_age = (df["Age2_Yr104"] >= 19) | (
    (df["DOB_Yr_Child"] == 1985) & (df["DOB_Mo_Child"] < 9)
)

sample_attrit = df[
    (df["HS2_FE90"].notna()) &
    (df["SampleID"] != 12) &
    cond_age
].copy()

attrit_coef_HS, attrit_se_HS, attrit_coef_Pre, attrit_se_Pre, attrit_N = \
    fe_regression(sample_attrit, "Attrit", add_controls=False)

ctrl_series_attrit = sample_attrit.loc[sample_attrit["PreK_FE"] == 3, "Attrit"]
attrit_ctrl_mean = ctrl_series_attrit.mean()
attrit_ctrl_se = ctrl_series_attrit.std(ddof=1) / np.sqrt(ctrl_series_attrit.count())

# ------------------------------------------------------------
# 4. 其它协变量（Sample90_2 == 1）
# ------------------------------------------------------------
covariate_vars = [
    ("Res_0to3", "In mother's HH, 0–3"),
    ("HealthCond_before", "Pre-existing health limitation"),
    ("logBW", "ln(birth weight)"),
    ("VLow_BW", "Very low BW (<3.31 lbs)"),
    ("LogInc_0to3", "ln(income), age 0–3"),
    ("LogIncAt3", "ln(income), age 3"),
    ("FirstBorn", "Firstborn"),
    ("Male", "Male"),
    ("Age2_Yr104", "Age in 2004 (in years)"),
    ("PPVTat3", "PPVT score, age 3"),
    ("HOME_Pct_0to3", "HOME score, age 0–3"),
    ("Moth_HrsWorked_BefBirth", "Mom avg hours worked, year before birth"),
    ("Moth_HrsWorked_0to1", "Mom avg hours worked, age 0–1"),
    ("Father_HH_0to3", "Father in HH, 0–3"),
    ("GMom_0to3", "Grandmother in HH, 0–3"),
    ("MomCare", "Maternal care, age 0–3"),
    ("RelCare", "Relative care, age 0–3"),
    ("NonRelCare", "Nonrelative care, age 0–3"),
    ("Breastfed", "Breastfed"),
    ("Moth_Smoke_BefBirth", "Mom smoked before birth"),
    ("Alc_BefBirth", "Mom drank before birth"),
    ("Doctor_0to3", "Regular doctor's visits, age 0–3"),
    ("Dentist_0to3", "Ever been to dentist, age 0–3"),
    ("Moth_WeightChange", "Weight change during pregnancy"),
    ("Illness_1stYr", "Child illness, age 0–1"),
    ("Premature", "Premature birth"),
    ("Insurance_0to3", "Private health insurance, age 0–3"),
    ("Medicaid_0to3", "Medicaid, age 0–3"),
    ("PreTreatIndex", "Pre-treatment index"),
]

sample_cov = df[df["Sample90_2"] == 1].copy()

rows = []

# --- Attrited 行 ---
rows.append({
    "Variable": "Attrited",
    "Label": "Attrited",
    "coef_HS": attrit_coef_HS,
    "se_HS": attrit_se_HS,
    "coef_Pre": attrit_coef_Pre,
    "se_Pre": attrit_se_Pre,
    "ControlMean": attrit_ctrl_mean,
    "se_Control": attrit_ctrl_se,
    "N": attrit_N,
})

# --- PPVTat3 行（单独写，方便控制顺序） ---
ppvt_coef_HS, ppvt_se_HS, ppvt_coef_Pre, ppvt_se_Pre, ppvt_N = \
    fe_regression(sample_cov, "PPVTat3", add_controls=True)

ctrl_series_ppvt = sample_cov.loc[sample_cov["PreK_FE"] == 3, "PPVTat3"]
ppvt_ctrl_mean = ctrl_series_ppvt.mean()
ppvt_ctrl_se = ctrl_series_ppvt.std(ddof=1) / np.sqrt(ctrl_series_ppvt.count())

rows.append({
    "Variable": "PPVTat3",
    "Label": "PPVT score, age 3",
    "coef_HS": ppvt_coef_HS,
    "se_HS": ppvt_se_HS,
    "coef_Pre": ppvt_coef_Pre,
    "se_Pre": ppvt_se_Pre,
    "ControlMean": ppvt_ctrl_mean,
    "se_Control": ppvt_ctrl_se,
    "N": ppvt_N,
})

# --- 其余协变量 ---
for var, label in covariate_vars:
    if var == "PPVTat3":
        continue  # 已经单独处理

    coef_HS, se_HS, coef_Pre, se_Pre, n_obs = fe_regression(
        sample_cov, var, add_controls=True
    )

    ctrl_series = sample_cov.loc[sample_cov["PreK_FE"] == 3, var]
    ctrl_mean = ctrl_series.mean()
    ctrl_se = ctrl_series.std(ddof=1) / np.sqrt(ctrl_series.count())

    rows.append({
        "Variable": var,
        "Label": label,
        "coef_HS": coef_HS,
        "se_HS": se_HS,
        "coef_Pre": coef_Pre,
        "se_Pre": se_Pre,
        "ControlMean": ctrl_mean,
        "se_Control": ctrl_se,
        "N": n_obs,
    })

table2 = pd.DataFrame(rows)

# 先保存一份原始结果（不排序）
# table2.to_csv("table2_results.csv", index=False)
# print("Table 2 raw results saved to table2_results.csv")

# ------------------------------------------------------------
# 5. 按论文 Table 2 的顺序重排 & 输出 ordered 版本
# ------------------------------------------------------------
order = [
    "Attrited",
    "PPVTat3",
    "logBW",
    "VLow_BW",
    "Res_0to3",
    "HealthCond_before",
    "FirstBorn",
    "Male",
    "Age2_Yr104",
    "HOME_Pct_0to3",
    "Father_HH_0to3",
    "GMom_0to3",
    "MomCare",
    "RelCare",
    "NonRelCare",
    "Breastfed",
    "Doctor_0to3",
    "Dentist_0to3",
    "Moth_WeightChange",
    "Illness_1stYr",
    "Premature",
    "Insurance_0to3",
    "Medicaid_0to3",
    "LogInc_0to3",
    "LogIncAt3",
    "Moth_HrsWorked_BefBirth",
    "Moth_HrsWorked_0to1",
    "Moth_Smoke_BefBirth",
    "Alc_BefBirth",
    "PreTreatIndex",
]

label_map = {
    "Attrited": "Attrited",
    "PPVTat3": "PPVT score, age 3",
    "logBW": "ln(birth weight)",
    "VLow_BW": "Very low BW (<3.31 lbs)",
    "Res_0to3": "In mother's HH, 0–3",
    "HealthCond_before": "Pre-existing health limitation",
    "FirstBorn": "Firstborn",
    "Male": "Male",
    "Age2_Yr104": "Age in 2004 (in years)",
    "HOME_Pct_0to3": "HOME score, age 0–3",
    "Father_HH_0to3": "Father in HH, 0–3",
    "GMom_0to3": "Grandmother in HH, 0–3",
    "MomCare": "Maternal care, age 0–3",
    "RelCare": "Relative care, age 0–3",
    "NonRelCare": "Nonrelative care, age 0–3",
    "Breastfed": "Breastfed",
    "Doctor_0to3": "Regular doctor's visits, age 0–3",
    "Dentist_0to3": "Ever been to dentist, age 0–3",
    "Moth_WeightChange": "Weight change during pregnancy",
    "Illness_1stYr": "Child illness, age 0–1",
    "Premature": "Premature birth",
    "Insurance_0to3": "Private health insurance, age 0–3",
    "Medicaid_0to3": "Medicaid, age 0–3",
    "LogInc_0to3": "ln(income), age 0–3",
    "LogIncAt3": "ln(income), age 3",
    "Moth_HrsWorked_BefBirth": "Mom avg hours worked, year before birth",
    "Moth_HrsWorked_0to1": "Mom avg hours worked, age 0–1",
    "Moth_Smoke_BefBirth": "Mom smoked before birth",
    "Alc_BefBirth": "Mom drank before birth",
    "PreTreatIndex": "Pre-treatment index",
}

table2["Variable"] = pd.Categorical(table2["Variable"],
                                    categories=order,
                                    ordered=True)
table2 = table2.sort_values("Variable").reset_index(drop=True)
table2["Label"] = table2["Variable"].map(label_map)

cols = [
    "Label",
    "Variable",
    "coef_HS", "se_HS",
    "coef_Pre", "se_Pre",
    "ControlMean", "se_Control",
    "N"
]
table2_ordered = table2[cols]

table2_ordered.to_csv("table2_results.csv", index=False)
print("Ordered Table 2 saved to table2_results.csv")
