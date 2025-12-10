import numpy as np
import pandas as pd

# 0. 读入数据 & 负值→缺失（ChildID-hhID 之间的所有变量）
dta_path = "113563-V1/data_Deming_2008_0217.dta"  # 按需修改路径
df = pd.read_stata(dta_path)

cols = list(df.columns)
start = cols.index("ChildID")
end = cols.index("hhID")
for c in cols[start:end + 1]:
    s = df[c]
    mask = s < 0
    if mask.any():
        df.loc[mask, c] = np.nan

# copy 一份
d = df.copy()

# 1. 构造 Age2_Mo*（先用 Age_Mo，再用 PPVTAge 补，然后用±24 个月补）
years = [86, 88, 90, 92, 94, 96, 98, 100, 102, 104]

for y in years:
    d[f"Age2_Mo{y}"] = d[f"Age_Mo{y}"]
    mask = d[f"Age2_Mo{y}"].isna() & d[f"PPVTAge{y}"].notna()
    d.loc[mask, f"Age2_Mo{y}"] = d.loc[mask, f"PPVTAge{y}"]

# 逐条对 “+/-24 个月” 的替换
d.loc[(d["Age2_Mo86"].isna()) & (d["Age2_Mo88"] >= 25) & d["Age2_Mo88"].notna(), "Age2_Mo86"] = d["Age2_Mo88"] - 24
d.loc[(d["Age2_Mo88"].isna()) & d["Age2_Mo86"].notna(), "Age2_Mo88"] = d["Age2_Mo86"] + 24
d.loc[(d["Age2_Mo88"].isna()) & (d["Age2_Mo90"] >= 25) & d["Age2_Mo90"].notna(), "Age2_Mo88"] = d["Age2_Mo90"] - 24
d.loc[(d["Age2_Mo90"].isna()) & d["Age2_Mo88"].notna(), "Age2_Mo90"] = d["Age2_Mo88"] + 24
d.loc[(d["Age2_Mo90"].isna()) & (d["Age2_Mo92"] >= 25) & d["Age2_Mo92"].notna(), "Age2_Mo90"] = d["Age2_Mo92"] - 24
d.loc[(d["Age2_Mo92"].isna()) & d["Age2_Mo90"].notna(), "Age2_Mo92"] = d["Age2_Mo90"] + 24
d.loc[(d["Age2_Mo92"].isna()) & (d["Age2_Mo94"] >= 25) & d["Age2_Mo94"].notna(), "Age2_Mo92"] = d["Age2_Mo94"] - 24
d.loc[(d["Age2_Mo94"].isna()) & d["Age2_Mo92"].notna(), "Age2_Mo94"] = d["Age2_Mo92"] + 24
d.loc[(d["Age2_Mo94"].isna()) & (d["Age2_Mo96"] >= 25) & d["Age2_Mo96"].notna(), "Age2_Mo94"] = d["Age2_Mo96"] - 24
d.loc[(d["Age2_Mo96"].isna()) & d["Age2_Mo94"].notna(), "Age2_Mo96"] = d["Age2_Mo94"] + 24
d.loc[(d["Age2_Mo96"].isna()) & (d["Age2_Mo98"] >= 25) & d["Age2_Mo98"].notna(), "Age2_Mo96"] = d["Age2_Mo98"] - 24
d.loc[(d["Age2_Mo98"].isna()) & d["Age2_Mo96"].notna(), "Age2_Mo98"] = d["Age2_Mo96"] + 24
d.loc[(d["Age2_Mo98"].isna()) & (d["Age2_Mo100"] >= 25) & d["Age2_Mo100"].notna(), "Age2_Mo98"] = d["Age2_Mo100"] - 24
d.loc[(d["Age2_Mo100"].isna()) & d["Age2_Mo98"].notna(), "Age2_Mo100"] = d["Age2_Mo98"] + 24
d.loc[(d["Age2_Mo100"].isna()) & (d["Age2_Mo102"] >= 25) & d["Age2_Mo102"].notna(), "Age2_Mo100"] = d["Age2_Mo102"] - 24
d.loc[(d["Age2_Mo102"].isna()) & d["Age2_Mo100"].notna(), "Age2_Mo102"] = d["Age2_Mo100"] + 24
d.loc[(d["Age2_Mo102"].isna()) & (d["Age2_Mo104"] >= 25) & d["Age2_Mo104"].notna(), "Age2_Mo102"] = d["Age2_Mo104"] - 24
d.loc[(d["Age2_Mo104"].isna()) & d["Age2_Mo102"].notna(), "Age2_Mo104"] = d["Age2_Mo102"] + 24

# 2. Elig1_*, Elig2_* （“至少两个 54 个月以上的兄弟姐妹” 规则）
#    最终逻辑：Age>47且家中Age>47的孩子数>1→Elig=1，其余=0
for y in [86, 88, 90]:
    grp = d.groupby("MotherID")
    count1 = grp[f"Age_Mo{y}"].transform(lambda s: ((s > 47) & s.notna()).sum())
    count2 = grp[f"Age2_Mo{y}"].transform(lambda s: ((s > 47) & s.notna()).sum())

    d[f"Elig1_{y}"] = np.where(
        (d[f"Age_Mo{y}"] > 47) & d[f"Age_Mo{y}"].notna() & (count1 > 1),
        1.0,
        0.0,
    )
    d[f"Elig2_{y}"] = np.where(
        (d[f"Age2_Mo{y}"] > 47) & d[f"Age2_Mo{y}"].notna() & (count2 > 1),
        1.0,
        0.0,
    )

# 去掉“在资格前死亡的孩子”
for x in [86, 88, 90]:
    dead = d[f"Res{x}"] == 8
    d.loc[dead, f"Elig1_{x}"] = np.nan
    d.loc[dead, f"Elig2_{x}"] = np.nan

# 3. 按照Rule 2生成HS2_90/Pre2_90/None2_90
for y in [1, 2]:
    elig = d[f"Elig{y}_90"]

    # rowmax(Ever_HS88 Ever_HS90) & rowmax(Ever_Preschool88 Ever_Preschool90)
    hs = d[["Ever_HS88", "Ever_HS90"]].max(axis=1, skipna=True)
    pre = d[["Ever_Preschool88", "Ever_Preschool90"]].max(axis=1, skipna=True)

    d[f"HS{y}_90"] = np.where(elig == 1, hs, np.nan)
    d[f"Pre{y}_90"] = np.where(elig == 1, pre, np.nan)

    # “replace HS = 0 if Elig==1 & HS!=1”
    d.loc[(elig == 1) & (d[f"HS{y}_90"] != 1), f"HS{y}_90"] = 0
    d.loc[(elig == 1) & (d[f"Pre{y}_90"] != 1), f"Pre{y}_90"] = 0

    # rule：同时 HS=1 和 Pre=1 时当作 HS
    d.loc[d[f"HS{y}_90"] == 1, f"Pre{y}_90"] = 0

    # None = Elig==1 且不在 HS/Pre 中
    d[f"None{y}_90"] = np.where(elig == 1, 1.0, np.nan)
    d.loc[elig == 0, f"None{y}_90"] = np.nan
    d.loc[(d[f"HS{y}_90"] == 1) | (d[f"Pre{y}_90"] == 1), f"None{y}_90"] = 0

# 4. FE 样本：HS2_FE90/Pre2_FE90/None2_FE90
grp = d.groupby("MotherID")
# Num2Elig90：家里Age2_Mo90>47且非缺失的孩子数
d["Num2Elig90"] = grp["Age2_Mo90"].transform(lambda s: ((s > 47) & s.notna()).sum())

hs_count = grp["HS2_90"].transform(lambda s: (s == 1).sum())
pre_count = grp["Pre2_90"].transform(lambda s: (s == 1).sum())
none_count = grp["None2_90"].transform(lambda s: (s == 1).sum())

temp = np.where(d["HS2_90"] == 1, hs_count, np.nan)
temp2 = np.where(d["Pre2_90"] == 1, pre_count, np.nan)
temp3 = np.where(d["None2_90"] == 1, none_count, np.nan)

# 如果所有eligible的孩子都在同一类，则该类temp设为missing（家庭没有 variation）
temp = np.where(d["Num2Elig90"] == temp, np.nan, temp)
temp2 = np.where(d["Num2Elig90"] == temp2, np.nan, temp2)
temp3 = np.where(d["Num2Elig90"] == temp3, np.nan, temp3)

d["HS2_FE90"] = np.where(~np.isnan(temp), 1.0, np.nan)
d["Pre2_FE90"] = np.where(~np.isnan(temp2), 1.0, np.nan)
d["None2_FE90"] = np.where(~np.isnan(temp3), 1.0, np.nan)

# 互斥处理（只保留一个=1，其余改为0）
d.loc[(d["Pre2_FE90"] == 1) | (d["None2_FE90"] == 1), "HS2_FE90"] = 0
d.loc[(d["HS2_FE90"] == 1) | (d["None2_FE90"] == 1), "Pre2_FE90"] = 0
d.loc[(d["HS2_FE90"] == 1) | (d["Pre2_FE90"] == 1), "None2_FE90"] = 0

# 5. PreK & PreK_FE（1=Head Start, 2=Other Preschool, 3=None）
d["PreK"] = np.nan
d.loc[d["HS2_90"] == 1, "PreK"] = 1
d.loc[d["Pre2_90"] == 1, "PreK"] = 2
d.loc[d["None2_90"] == 1, "PreK"] = 3

d["PreK_FE"] = np.nan
d.loc[d["HS2_FE90"] == 1, "PreK_FE"] = 1
d.loc[d["Pre2_FE90"] == 1, "PreK_FE"] = 2
d.loc[d["None2_FE90"] == 1, "PreK_FE"] = 3

# 6. 种族 dummies
d["Hispanic"] = (d["Race_Child"] == 1).astype(float)
d["Black"] = (d["Race_Child"] == 2).astype(float)
d["White"] = (d["Race_Child"] == 3).astype(float)
d["NonBlack"] = ((d["Race_Child"] != 2) & d["Race_Child"].notna()).astype(float)

# 7. Permanent income（NetFamInc* 乘 CPI 系数，然后 rowmean）
mult = {
    78: 2.82,
    79: 2.54,
    80: 2.24,
    81: 2.03,
    82: 1.90,
    83: 1.85,
    84: 1.78,
    85: 1.71,
    86: 1.68,
    87: 1.62,
    88: 1.55,
    89: 1.48,
    90: 1.41,
    91: 1.35,
    92: 1.31,
    93: 1.27,
    95: 1.21,
    97: 1.15,
    99: 1.10,
    101: 1.04,
}
for y, m in mult.items():
    d[f"NetFamInc{y}"] = d[f"NetFamInc{y}"] * m

perm_cols = [
    "NetFamInc78",
    "NetFamInc79",
    "NetFamInc80",
    "NetFamInc81",
    "NetFamInc82",
    "NetFamInc83",
    "NetFamInc84",
    "NetFamInc85",
    "NetFamInc86",
    "NetFamInc87",
    "NetFamInc88",
    "NetFamInc89",
    "NetFamInc90",
    "NetFamInc91",
    "NetFamInc92",
    "NetFamInc93",
    "NetFamInc95",
    "NetFamInc97",
    "NetFamInc99",
    "NetFamInc101",
    "NetFamInc103",
]
d["PermInc"] = d[perm_cols].mean(axis=1, skipna=True)

# 8. 母亲教育：MomDropout / MomSomeColl
moth_cols = [c for c in d.columns if c.startswith("HighGrade_Moth")]
for c in moth_cols:
    d.loc[d[c] == 95, c] = np.nan

d["MothED"] = d[moth_cols].max(axis=1, skipna=True)

d["MomDropout"] = np.where(d["MothED"].isna(), np.nan, np.where(d["MothED"] < 12, 1.0, 0.0))
d["MomSomeColl"] = np.where(d["MothED"].isna(), np.nan, np.where(d["MothED"] >= 13, 1.0, 0.0))

# 9. Maternal AFQT：AgeAFQT_std
d["AgeAFQT"] = d["AFQT_Pct81_REV"]

scale_map = {
    14: 28.79544,
    15: 32.86273,
    16: 32.86273,
    17: 36.3544,
    18: 33.45777,
    19: 36.84,
    20: 41.84536,
    21: 40.95177,
    22: 42.82069,
}

for age, denom in scale_map.items():
    mask = d["Age_Mom79"] == age
    d.loc[mask, "AgeAFQT"] = d.loc[mask, "AgeAFQT"] * (35.60881 / denom)

afqt = d["AgeAFQT"]
d["AgeAFQT_std"] = (afqt - afqt.mean(skipna=True)) / afqt.std(skipna=True, ddof=1)

# 10. 模仿 tabexport 的 (mean, sd, count) by PreK
def make_table1_for_group(df_all: pd.DataFrame, race_var: str, use_fe: bool):
    """race_var: 'Black' 或 'NonBlack'
       use_fe: False → 用 PreK；True → 用 PreK_FE
    """
    group_var = "PreK_FE" if use_fe else "PreK"
    sub = df_all[df_all[race_var] == 1].copy()
    sub = sub[~sub[group_var].isna()]

    vars_list = ["PermInc", "MomDropout", "MomSomeColl", "AgeAFQT_std", "HighGrade_GMom79"]

    grp = sub.groupby(group_var)

    pieces = []
    for var in vars_list:
        g = grp[var].agg(["mean", "std", "count"])
        g.columns = pd.MultiIndex.from_product([[var], g.columns])
        pieces.append(g)

    out = pd.concat(pieces, axis=1)

    # 列顺序：PreK=1,2,3
    out = out.reindex(index=[1.0, 2.0, 3.0])
    return out


# 11. 生成 Table 1的两个CSV（NonBlack/Black，各含普通样本&FE子样本）
nonblack_main = make_table1_for_group(d, "NonBlack", use_fe=False)
nonblack_fe = make_table1_for_group(d, "NonBlack", use_fe=True)

black_main = make_table1_for_group(d, "Black", use_fe=False)
black_fe = make_table1_for_group(d, "Black", use_fe=True)

# 把普通样本和 FE 样本上下拼起来，方便在Excel对照
nb_out = pd.concat(
    {"Full_sample": nonblack_main, "FE_subsample": nonblack_fe},
    axis=0,
)
bl_out = pd.concat(
    {"Full_sample": black_main, "FE_subsample": black_fe},
    axis=0,
)

nb_out.to_csv("table1_NonBlack.csv")
bl_out.to_csv("table1_Black.csv")

d.to_csv("deming_cleaned_data.csv", index=False)
print("Done. table1_NonBlack.csv and table1_Black.csv")





# # 依照 Deming_2008_0217.do 中对应代码：
# # - 样本年龄修正与 Elig 规则
# # - Head Start / Preschool / None 分类（PreK, PreK_FE）
# # - 种族变量
# # - PermInc / MomDropout / MomSomeColl / AgeAFQT_std / HighGrade_GMom79
# # - 按 PreK & PreK_FE × 种族汇总 mean / sd / count

# import pandas as pd
# import numpy as np


# def clean_negatives(d):
#     """
#     对应 do 文件：
#     foreach var of varlist ChildID- hhID { replace `var'=. if `var'<0; };
#     """
#     cols = list(d.columns)
#     start_idx = cols.index("ChildID")
#     end_idx = cols.index("hhID")
#     for col in cols[start_idx : end_idx + 1]:
#         d.loc[d[col] < 0, col] = np.nan
#     return d


# def create_age2_and_elig(d):
#     """
#     对应 do 文件中：
#     - 创建 Age2_Mo*（用 PPVTAge 填补）以及用相邻年份 ±24 个月补缺
#     - 创建 Age_Yr* / Age2_Yr*
#     - 创建 Elig1_y / Elig2_y（y in 86,88,90）
#     - 用死亡信息 Res* 将 Elig 设为缺失
#     """
#     years = [86, 88, 90, 92, 94, 96, 98, 100, 102, 104]

#     # 1) Age2_Mo 使用 Age_Mo 和 PPVTAge
#     for y in years:
#         d[f"Age2_Mo{y}"] = d[f"Age_Mo{y}"]
#         ppvt_col = f"PPVTAge{y}"
#         if ppvt_col in d.columns:
#             mask = d[f"Age2_Mo{y}"].isna() & d[ppvt_col].notna()
#             d.loc[mask, f"Age2_Mo{y}"] = d.loc[mask, ppvt_col]

#     # 2) 按 do 文件顺序用相邻年份 ±24 个月补缺
#     def rep(cond, target, values):
#         mask = cond & d[target].isna()
#         d.loc[mask, target] = values[mask]

#     rep((d["Age2_Mo86"].isna()) & (d["Age2_Mo88"] >= 25) & d["Age2_Mo88"].notna(), "Age2_Mo86", d["Age2_Mo88"] - 24)
#     rep(d["Age2_Mo88"].isna() & d["Age2_Mo86"].notna(), "Age2_Mo88", d["Age2_Mo86"] + 24)
#     rep(d["Age2_Mo88"].isna() & (d["Age2_Mo90"] >= 25) & d["Age2_Mo90"].notna(), "Age2_Mo88", d["Age2_Mo90"] - 24)
#     rep(d["Age2_Mo90"].isna() & d["Age2_Mo88"].notna(), "Age2_Mo90", d["Age2_Mo88"] + 24)
#     rep(d["Age2_Mo90"].isna() & (d["Age2_Mo92"] >= 25) & d["Age2_Mo92"].notna(), "Age2_Mo90", d["Age2_Mo92"] - 24)
#     rep(d["Age2_Mo92"].isna() & d["Age2_Mo90"].notna(), "Age2_Mo92", d["Age2_Mo90"] + 24)
#     rep(d["Age2_Mo92"].isna() & (d["Age2_Mo94"] >= 25) & d["Age2_Mo94"].notna(), "Age2_Mo92", d["Age2_Mo94"] - 24)
#     rep(d["Age2_Mo94"].isna() & d["Age2_Mo92"].notna(), "Age2_Mo94", d["Age2_Mo92"] + 24)
#     rep(d["Age2_Mo94"].isna() & (d["Age2_Mo96"] >= 25) & d["Age2_Mo96"].notna(), "Age2_Mo94", d["Age2_Mo96"] - 24)
#     rep(d["Age2_Mo96"].isna() & d["Age2_Mo94"].notna(), "Age2_Mo96", d["Age2_Mo94"] + 24)
#     rep(d["Age2_Mo96"].isna() & (d["Age2_Mo98"] >= 25) & d["Age2_Mo98"].notna(), "Age2_Mo96", d["Age2_Mo98"] - 24)
#     rep(d["Age2_Mo98"].isna() & d["Age2_Mo96"].notna(), "Age2_Mo98", d["Age2_Mo96"] + 24)
#     rep(d["Age2_Mo98"].isna() & (d["Age2_Mo100"] >= 25) & d["Age2_Mo100"].notna(), "Age2_Mo98", d["Age2_Mo100"] - 24)
#     rep(d["Age2_Mo100"].isna() & d["Age2_Mo98"].notna(), "Age2_Mo100", d["Age2_Mo98"] + 24)
#     rep(d["Age2_Mo100"].isna() & (d["Age2_Mo102"] >= 25) & d["Age2_Mo102"].notna(), "Age2_Mo100", d["Age2_Mo102"] - 24)
#     rep(d["Age2_Mo102"].isna() & d["Age2_Mo100"].notna(), "Age2_Mo102", d["Age2_Mo100"] + 24)
#     rep(d["Age2_Mo102"].isna() & (d["Age2_Mo104"] >= 25) & d["Age2_Mo104"].notna(), "Age2_Mo102", d["Age2_Mo104"] - 24)
#     rep(d["Age2_Mo104"].isna() & d["Age2_Mo102"].notna(), "Age2_Mo104", d["Age2_Mo102"] + 24)

#     # 3) 按月龄生成 Age_Yr* / Age2_Yr*
#     for x in years:
#         d[f"Age_Yr{x}"] = np.where(d[f"Age_Mo{x}"] < 12, 0, np.nan)
#         d[f"Age2_Yr{x}"] = np.where(d[f"Age2_Mo{x}"] < 12, 0, np.nan)
#         for yr in range(1, 38):
#             cond1 = (d[f"Age_Mo{x}"] >= 12 * yr) & (d[f"Age_Mo{x}"] < 12 * yr + 12)
#             cond2 = (d[f"Age2_Mo{x}"] >= 12 * yr) & (d[f"Age2_Mo{x}"] < 12 * yr + 12)
#             d.loc[cond1, f"Age_Yr{x}"] = yr
#             d.loc[cond2, f"Age2_Yr{x}"] = yr

#     # 4) 创建 Elig1_y / Elig2_y （家庭中 age>47 月的孩子数 ≥2）
#     for y in [86, 88, 90]:
#         grp = d.groupby("MotherID")
#         count1 = grp[f"Age_Mo{y}"].transform(lambda s: ((s > 47) & s.notna()).sum())
#         count2 = grp[f"Age2_Mo{y}"].transform(lambda s: ((s > 47) & s.notna()).sum())
#         d[f"Elig1_{y}"] = np.where(count1 > 1, 1.0, 0.0)
#         d[f"Elig2_{y}"] = np.where(count2 > 1, 1.0, 0.0)

#     # 5) 排除在 eligibility 前死亡的孩子（对 86/88/90）
#     for x in [86, 88, 90]:
#         dead = (d[f"Res{x}"] == 8)
#         d.loc[dead, f"Elig1_{x}"] = np.nan
#         d.loc[dead, f"Elig2_{x}"] = np.nan

#     # 同时生成 Num2Elig90（FE 构造要用）
#     grp = d.groupby("MotherID")
#     d["Num2Elig90"] = grp["Age2_Mo90"].transform(lambda s: ((s > 47) & s.notna()).sum())

#     return d


# def create_prek_and_prek_fe(d):
#     """
#     对应 do 文件中：
#     - 基于 Elig1_90 / Elig2_90 + Ever_HS88/90 + Ever_Preschool88/90
#       创建 HS1_90/HS2_90, Pre1_90/Pre2_90, None1_90/None2_90
#     - 基于家庭内部是否存在“不同学前状态”的兄弟姐妹，创建 HS2_FE90 / Pre2_FE90 / None2_FE90
#     - 最终 PreK / PreK_FE（1=HS, 2=其他幼儿园, 3=不读幼儿园）
#     """

#     # 1) 创建 HS1_90 / HS2_90, Pre1_90 / Pre2_90, None1_90 / None2_90
#     for y in [1, 2]:
#         elig_col = f"Elig{y}_90"
#         hs_rowmax = d[["Ever_HS88", "Ever_HS90"]].max(axis=1, skipna=True)
#         pre_rowmax = d[["Ever_Preschool88", "Ever_Preschool90"]].max(axis=1, skipna=True)

#         hs_var = f"HS{y}_90"
#         pre_var = f"Pre{y}_90"
#         none_var = f"None{y}_90"

#         d[hs_var] = np.where(d[elig_col] == 1, hs_rowmax, np.nan)
#         d[pre_var] = np.where(d[elig_col] == 1, pre_rowmax, np.nan)

#         # Elig==1 且不为1 的设为0
#         d.loc[(d[elig_col] == 1) & (d[hs_var] != 1), hs_var] = 0
#         d.loc[(d[elig_col] == 1) & (d[pre_var] != 1), pre_var] = 0
#         # 同时 HS=1 时，Pre 归 0（双报算 Head Start）
#         d.loc[d[hs_var] == 1, pre_var] = 0

#         # None = 只有 Elig==1 且既非 HS 也非 Pre
#         d[none_var] = np.where(d[elig_col] == 1, 1.0, np.nan)
#         d.loc[d[elig_col] == 0, none_var] = np.nan
#         d.loc[(d[hs_var] == 1) | (d[pre_var] == 1), none_var] = 0

#     # 2) 创建 FE 样本分类（只用规则2：HS2_90/Pre2_90/None2_90）
#     x = 2
#     y = 90
#     hs_col = f"HS{x}_{y}"
#     pre_col = f"Pre{x}_{y}"
#     none_col = f"None{x}_{y}"
#     num_col = f"Num{x}Elig{y}"

#     grp = d.groupby("MotherID")
#     count_hs = grp[hs_col].transform(lambda s: (s == 1).sum())
#     count_pre = grp[pre_col].transform(lambda s: (s == 1).sum())
#     count_none = grp[none_col].transform(lambda s: (s == 1).sum())

#     # 只对该类别==1 的孩子记录家庭里的计数，其他设为缺失
#     temp = np.where(d[hs_col] == 1, count_hs, np.nan).astype(float)
#     temp2 = np.where(d[pre_col] == 1, count_pre, np.nan).astype(float)
#     temp3 = np.where(d[none_col] == 1, count_none, np.nan).astype(float)

#     # 如果某户所有 eligible 孩子都在同一类别，则该类别计数==Num2Elig90，设为缺失 -> 不构成 FE
#     temp[temp == d[num_col]] = np.nan
#     temp2[temp2 == d[num_col]] = np.nan
#     temp3[temp3 == d[num_col]] = np.nan

#     hs_fe = np.where(~np.isnan(temp), 1.0, np.nan)
#     pre_fe = np.where(~np.isnan(temp2), 1.0, np.nan)
#     none_fe = np.where(~np.isnan(temp3), 1.0, np.nan)

#     # 保证互斥：有一个为1，其它必须0
#     hs_fe = np.where((pre_fe == 1) | (none_fe == 1), 0.0, hs_fe)
#     pre_fe = np.where((hs_fe == 1) | (none_fe == 1), 0.0, pre_fe)
#     none_fe = np.where((hs_fe == 1) | (pre_fe == 1), 0.0, none_fe)

#     d["HS2_FE90"] = hs_fe
#     d["Pre2_FE90"] = pre_fe
#     d["None2_FE90"] = none_fe

#     # 3) 最终 PreK / PreK_FE（1=HS, 2=Pre, 3=None）
#     d["PreK"] = np.nan
#     d.loc[d["HS2_90"] == 1, "PreK"] = 1
#     d.loc[d["Pre2_90"] == 1, "PreK"] = 2
#     d.loc[d["None2_90"] == 1, "PreK"] = 3

#     d["PreK_FE"] = np.nan
#     d.loc[d["HS2_FE90"] == 1, "PreK_FE"] = 1
#     d.loc[d["Pre2_FE90"] == 1, "PreK_FE"] = 2
#     d.loc[d["None2_FE90"] == 1, "PreK_FE"] = 3

#     return d


# def create_race_and_covariates_for_table1(d):
#     """
#     对应 do 文件中：
#     - gen byte Hispanic/Black/White/NonBlack
#     - 永久收入 PermInc（NetFamInc* 调整为 2004 年）
#     - 母亲教育 MomDropout / MomSomeColl（基于 HighGrade_Moth*）
#     - 年龄调整 AFQT + 标准化 AgeAFQT_std
#     - 外祖母教育 HighGrade_GMom79 直接使用
#     """
#     # 种族 dummies
#     d["Hispanic"] = (d["Race_Child"] == 1).astype(float)
#     d["Black"] = (d["Race_Child"] == 2).astype(float)
#     d["White"] = (d["Race_Child"] == 3).astype(float)
#     d["NonBlack"] = ((d["Race_Child"] != 2) & d["Race_Child"].notna()).astype(float)

#     # 永久收入（调整到 2004 美元）
#     income_years = [78, 79, 80, 81, 82, 83, 84, 85, 86,
#                     87, 88, 89, 90, 91, 92, 93, 95, 97, 99, 101]
#     multipliers = {
#         78: 2.82, 79: 2.54, 80: 2.24, 81: 2.03, 82: 1.90,
#         83: 1.85, 84: 1.78, 85: 1.71, 86: 1.68, 87: 1.62,
#         88: 1.55, 89: 1.48, 90: 1.41, 91: 1.35, 92: 1.31,
#         93: 1.27, 95: 1.21, 97: 1.15, 99: 1.10, 101: 1.04
#     }
#     for y in income_years:
#         col = f"NetFamInc{y}"
#         d[col] = d[col] * multipliers[y]

#     perm_cols = [f"NetFamInc{y}" for y in income_years]
#     d["PermInc"] = d[perm_cols].mean(axis=1, skipna=True)

#     # 母亲教育：把 HighGrade_Moth* 中的 95 视为缺失，然后取 rowmax
#     moth_cols = [c for c in d.columns if c.startswith("HighGrade_Moth")]
#     for col in moth_cols:
#         d.loc[d[col] == 95, col] = np.nan
#     d["MothED"] = d[moth_cols].max(axis=1, skipna=True)

#     d["MomDropout"] = np.where(d["MothED"].isna(), np.nan,
#                                np.where(d["MothED"] < 12, 1.0, 0.0))
#     d["MomHS"] = np.where(d["MothED"].isna(), np.nan,
#                           np.where(d["MothED"] == 12, 1.0, 0.0))
#     d["MomSomeColl"] = np.where(d["MothED"].isna(), np.nan,
#                                 np.where(d["MothED"] >= 13, 1.0, 0.0))

#     # 年龄调整 AFQT
#     d["AgeAFQT"] = d["AFQT_Pct81_REV"]
#     d.loc[d["Age_Mom79"] == 14, "AgeAFQT"] = d["AgeAFQT"] * (35.60881 / 28.79544)
#     d.loc[d["Age_Mom79"] == 15, "AgeAFQT"] = d["AgeAFQT"] * (35.60881 / 32.86273)
#     d.loc[d["Age_Mom79"] == 16, "AgeAFQT"] = d["AgeAFQT"] * (35.60881 / 32.86273)

#     # 标准化 AgeAFQT_std（mean=0, sd=1）
#     afqt = d["AgeAFQT"]
#     d["AgeAFQT_std"] = (afqt - afqt.mean(skipna=True)) / afqt.std(ddof=1, skipna=True)

#     return d


# def summarize_table1(d, group_var, race_var):
#     """
#     模拟 do 文件中：
#     tabexport PermInc MomDropout MomSomeColl AgeAFQT_std HighGrade_GMom79 if `x'==1, by(PreK/PreK_FE) s(mean sd count)

#     返回一个 MultiIndex 列的 DataFrame：
#       index: PreK (1=HS,2=Pre,3=None)
#       columns: (变量名, ['mean','std','count'])
#     """
#     vars_table1 = ["PermInc", "MomDropout", "MomSomeColl", "AgeAFQT_std", "HighGrade_GMom79"]

#     sub = d[d[race_var] == 1].copy()
#     sub = sub[~sub[group_var].isna()]
#     grp = sub.groupby(group_var)

#     stats = {}
#     for v in vars_table1:
#         g = grp[v].agg(["mean", "std", "count"])
#         stats[v] = g

#     out = pd.concat(stats, axis=1)
#     # 排序 PreK=1,2,3（如果存在）
#     try:
#         out = out.sort_index()
#     except Exception:
#         pass
#     return out


# def main():
#     # 读入原始数据
#     d = pd.read_stata("113563-V1/data_Deming_2008_0217.dta")

#     # 1) 负值视为缺失（对应 ChildID-hhID）
#     d = clean_negatives(d)

#     # 2) 年龄修正 + Elig + Num2Elig90
#     d = create_age2_and_elig(d)

#     # 3) PreK / PreK_FE
#     d = create_prek_and_prek_fe(d)

#     # 4) 种族 + Table1 所需协变量
#     d = create_race_and_covariates_for_table1(d)

#     # 5) 生成四个表：Black/NonBlack × PreK/PreK_FE
#     black_main = summarize_table1(d, "PreK", "Black")
#     nonblack_main = summarize_table1(d, "PreK", "NonBlack")
#     black_fe = summarize_table1(d, "PreK_FE", "Black")
#     nonblack_fe = summarize_table1(d, "PreK_FE", "NonBlack")

#     # 打印到终端
#     pd.set_option("display.width", 180)
#     pd.set_option("display.max_columns", 40)

#     print("\n=== Table 1: Black × PreK (main sample) ===")
#     print(black_main)
#     print("\n=== Table 1: NonBlack × PreK (main sample) ===")
#     print(nonblack_main)
#     print("\n=== Table 1: Black × PreK_FE (FE sample) ===")
#     print(black_fe)
#     print("\n=== Table 1: NonBlack × PreK_FE (FE sample) ===")
#     print(nonblack_fe)

#     # 同时保存为 csv，方便你和论文里的 Table 1 一行一列对照
#     black_main.to_csv("table1_Black.csv")
#     nonblack_main.to_csv("table1_NonBlack.csv")
#     black_fe.to_csv("table1_Black_FE.csv")
#     nonblack_fe.to_csv("table1_NonBlack_FE.csv")


# if __name__ == "__main__":
#     main()