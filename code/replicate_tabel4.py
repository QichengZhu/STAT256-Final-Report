
import pandas as pd
import numpy as np
import statsmodels.api as sm

# Input file from previous step
DATA_PATH = "deming_table2_data.csv"

def main():
    print("Loading data...")
    df = pd.read_csv(DATA_PATH)

    # =========================================================================
    # 0. IMPUTE COVARIATES BY RACE & GENDER (MIMIC STATA IMPUTE)
    # =========================================================================
    covariate_base = [
        "Res_0to3", "HealthCond_before", "VLow_BW", "logBW",
        "LogInc_0to3", "LogIncAt3", "FirstBorn", "PPVTat3",
        "HOME_Pct_0to3", "Moth_HrsWorked_BefBirth", "Moth_HrsWorked_Avg_0to3",
        "Moth_HrsWorked_0to1", "Father_HH_0to3", "GMom_0to3",
        "MomCare", "RelCare", "NonRelCare",
        "Moth_Smoke_BefBirth", "Alc_BefBirth", "Breastfed",
        "Doctor_0to3", "Dentist_0to3", "Moth_WeightChange",
        "Illness_1stYr", "Premature", "Insurance_0to3", "Medicaid_0to3"
    ]
    # Keep only covariates actually in the data
    covariate_base = [c for c in covariate_base if c in df.columns]

    # Predictors used for imputation: race & gender (as in Stata)
    impute_predictors = [c for c in ["Male", "Black", "Hispanic"] if c in df.columns]

    if "Sample90_2" in df.columns:
        sample_mask = df["Sample90_2"] == 1
    else:
        # Fallback: if Sample90_2 is missing (shouldn't happen if data is correct)
        sample_mask = pd.Series(False, index=df.index)

    for x in covariate_base:
        imp_col = f"{x}_imp"
        miss_col = f"{x}_miss"

        # Start imputed value as original
        df[imp_col] = df[x]

        # -------- Imputation: based on race & gender only, within Sample90_2 == 1 --------
        if impute_predictors:
            tmp = df.loc[sample_mask, impute_predictors + [x]].copy()
            group_means = tmp.groupby(impute_predictors)[x].transform("mean")
            df.loc[sample_mask, "_impute_mean_tmp"] = group_means

            mask_missing_to_fill = (
                sample_mask
                & df[x].isna()
                & df["_impute_mean_tmp"].notna()
            )
            df.loc[mask_missing_to_fill, imp_col] = df.loc[mask_missing_to_fill, "_impute_mean_tmp"]
            df.drop(columns=["_impute_mean_tmp"], inplace=True)
        else:
            # If we somehow have no race/gender predictors, fall back to overall mean in sample
            mean_val = df.loc[sample_mask, x].mean()
            mask_missing_to_fill = sample_mask & df[x].isna()
            df.loc[mask_missing_to_fill, imp_col] = mean_val

        # -------- Missing indicator, following Stata logic --------
        df[miss_col] = np.nan
        # gen x_miss = 1 if x==. & x_imp!=.
        cond1 = df[x].isna() & df[imp_col].notna()
        df.loc[cond1, miss_col] = 1
        # replace x_miss = 0 if x_miss!=1 & x_imp!=.
        cond0 = df[imp_col].notna() & df[miss_col].ne(1)
        df.loc[cond0, miss_col] = 0

    # =========================================================================
    # 0B. CREATE AGE-ADJUSTED MATERNAL AFQT & IMPUTE MISSING (FROM .DO FILE)
    # =========================================================================
    # Stata:
    # gen AgeAFQT = AFQT_Pct81_REV
    # replace AgeAFQT = AgeAFQT*(35.60881/28.79544) if Age_Mom79==14
    # ...
    # egen AgeAFQT_std = std(AgeAFQT), mean(0) std(1)
    # impute AgeAFQT_std Black Hispanic Age_Moth_Birth, gen(impAFQT_std)

    if "AFQT_Pct81_REV" in df.columns:
        df["AgeAFQT"] = df["AFQT_Pct81_REV"]

        if "Age_Mom79" in df.columns:
            # factors: 35.60881 / denom
            factors = {
                14: 35.60881 / 28.79544,
                15: 35.60881 / 32.86273,
                16: 35.60881 / 32.86273,
                17: 35.60881 / 36.3544,
                18: 35.60881 / 33.45777,
                19: 35.60881 / 36.84,
                20: 35.60881 / 41.84536,
                21: 35.60881 / 40.95177,
                22: 35.60881 / 42.82069
            }
            for age_val, factor in factors.items():
                mask = df["Age_Mom79"] == age_val
                df.loc[mask, "AgeAFQT"] = df.loc[mask, "AgeAFQT"] * factor

        # Standardize AgeAFQT to mean 0, sd 1
        mean_afqt = df["AgeAFQT"].mean(skipna=True)
        std_afqt = df["AgeAFQT"].std(ddof=1, skipna=True)
        if pd.notnull(std_afqt) and std_afqt > 0:
            df["AgeAFQT_std"] = (df["AgeAFQT"] - mean_afqt) / std_afqt
        else:
            df["AgeAFQT_std"] = np.nan
    else:
        print("Warning: AFQT_Pct81_REV not in data; AgeAFQT_std will be missing.")
        df["AgeAFQT_std"] = np.nan

    # ---- Impute AgeAFQT_std using Black, Hispanic, Age_Moth_Birth (regression imputation) ----
    if "AgeAFQT_std" in df.columns:
        # Start with original values
        df["impAFQT_std"] = df["AgeAFQT_std"]

        predictors_afqt = [c for c in ["Black", "Hispanic", "Age_Moth_Birth"] if c in df.columns]
        if predictors_afqt:
            # rows with observed AFQT and predictors
            mask_obs = df["AgeAFQT_std"].notna()
            for c in predictors_afqt:
                mask_obs &= df[c].notna()

            if mask_obs.any():
                X_train = df.loc[mask_obs, predictors_afqt]
                X_train = sm.add_constant(X_train)
                y_train = df.loc[mask_obs, "AgeAFQT_std"]

                try:
                    afqt_model = sm.OLS(y_train, X_train).fit()
                    # rows to impute
                    mask_miss = df["AgeAFQT_std"].isna()
                    for c in predictors_afqt:
                        mask_miss &= df[c].notna()

                    if mask_miss.any():
                        X_miss = df.loc[mask_miss, predictors_afqt]
                        X_miss = sm.add_constant(X_miss, has_constant="add")
                        y_pred = afqt_model.predict(X_miss)
                        df.loc[mask_miss, "impAFQT_std"] = y_pred
                except Exception as e:
                    print(f"AFQT imputation regression failed: {e}")
                    # if regression fails, impAFQT_std stays as original AgeAFQT_std
        else:
            # no predictors available; impAFQT_std == AgeAFQT_std
            pass

    # =========================================================================
    # 1. DATA PREP: Create AgeTest and Test Composites (Wide Format)
    # =========================================================================
    
    # Define years for the loop
    years = [86, 88, 90, 92, 94, 96, 98, 100, 102, 104]
    
    # Create AgeTest_Mo (copy of PPVTAge) and AgeTest_Yr
    for y in years:
        ppvt_age_col = f"PPVTAge{y}"
        mo_col = f"AgeTest_Mo{y}"
        yr_col = f"AgeTest_Yr{y}"
        
        if ppvt_age_col in df.columns:
            df[mo_col] = df[ppvt_age_col]
            df[yr_col] = 0
            for age_yr in range(1, 36):
                mask = (df[mo_col] >= 12 * age_yr) & (df[mo_col] < 12 * (age_yr + 1))
                df.loc[mask, yr_col] = age_yr

    # Create Test Score Composite (only for even years per Stata: 86(2)104)
    for y in range(86, 105, 2):
        tests = [f"PPVT_Pct{y}", f"PIATMT_Pct{y}", f"PIATRR_Pct{y}"]
        existing_tests = [t for t in tests if t in df.columns]
        
        if existing_tests:
            df[f"Test_Pct{y}"] = df[existing_tests].mean(axis=1, skipna=True)

    # =========================================================================
    # 2. RESHAPE TO LONG FORMAT
    # =========================================================================
    print("Reshaping to long format...")
    
    # Static columns to keep
    static_cols = [
        "ChildID", "MotherID", "Sample90_2", "Attrit", "Age2_Yr104", 
        "Black", "NonBlack", "Male", "PermInc",
        "AgeAFQT_std", "impAFQT_std",
        "MomDropout", "MomSomeColl", "HS2_FE90", "Pre2_FE90", "PreK_FE"
    ]
    
    # Pre-treatment covariates for regression: use *_imp and *_miss versions
    covariates_list = []
    for c in covariate_base:
        imp_col = f"{c}_imp"
        miss_col = f"{c}_miss"
        if imp_col in df.columns:
            covariates_list.append(imp_col)
        if miss_col in df.columns:
            covariates_list.append(miss_col)

    covariates_list = [c for c in covariates_list if c in df.columns]
    static_cols += covariates_list

    # Filter existing columns
    keep_cols = [c for c in static_cols if c in df.columns]
    
    # Variables to reshape (only scores that exist)
    stubnames = ["Test_Pct", "AgeTest_Yr"]
    for t in ["PPVT_Pct", "PIATMT_Pct", "PIATRR_Pct", "PPVT_Raw", "PIATMT_Raw", "PIATRR_Raw"]:
        if f"{t}86" in df.columns:
            stubnames.append(t)
    
    # Need to subset to keep_cols + wide columns before reshaping
    wide_cols = []
    for stub in stubnames:
        for y in years:
            col = f"{stub}{y}"
            if col in df.columns:
                wide_cols.append(col)
    
    # df_subset = df[list(set(keep_cols + wide_cols))].copy()
    cols = list(dict.fromkeys(keep_cols + wide_cols))  # Preserve order
    df_subset = df[cols].copy()

    # Reshape
    long_df = pd.wide_to_long(
        df_subset, 
        stubnames=stubnames, 
        i="ChildID", 
        j="year", 
        sep=""
    ).reset_index()

    # Filter Sample
    long_df = long_df[long_df["Sample90_2"] == 1].copy()

    # Filter Age Range (5 to 14)
    long_df = long_df[(long_df["AgeTest_Yr"] >= 5) & (long_df["AgeTest_Yr"] <= 14)].copy()

    # =========================================================================
    # 3. STANDARDIZATION & VARIABLE CREATION
    # =========================================================================
    print("Creating standardized scores and interaction terms...")

    # Age Groups
    long_df["Group_5to6"] = (long_df["AgeTest_Yr"] < 7).astype(int)
    long_df["Group_7to10"] = ((long_df["AgeTest_Yr"] >= 7) & (long_df["AgeTest_Yr"] <= 10)).astype(int)
    long_df["Group_11to14"] = (long_df["AgeTest_Yr"] >= 11).astype(int)
    long_df["Group_5to14"] = 1  # For loop facilitation

    # Standardize individual test components within age groups, then create composite
    for grp_name in ["5to6", "7to10", "11to14"]:
        grp_col = f"Group_{grp_name}"
        mask = long_df[grp_col] == 1
        
        for test in ["PPVT_Pct", "PIATMT_Pct", "PIATRR_Pct"]:
            if test in long_df.columns:
                vals = long_df.loc[mask, test]
                if not vals.empty and vals.std() > 0:
                    mean_val = vals.mean()
                    std_val = vals.std(ddof=1)
                    long_df.loc[mask, f"{test}_std_{grp_name}"] = (vals - mean_val) / std_val
        
        std_cols = [f"{t}_std_{grp_name}" for t in ["PPVT_Pct", "PIATMT_Pct", "PIATRR_Pct"] 
                   if f"{t}_std_{grp_name}" in long_df.columns]
        if std_cols:
            temp = long_df.loc[mask, std_cols].mean(axis=1, skipna=True)
            if temp.std() > 0:
                long_df.loc[mask, f"Test_std_{grp_name}"] = (temp - temp.mean()) / temp.std(ddof=1)

    long_df["Test_std"] = np.nan
    for grp_name in ["5to6", "7to10", "11to14"]:
        mask = long_df[f"Group_{grp_name}"] == 1
        std_col = f"Test_std_{grp_name}"
        if std_col in long_df.columns:
            long_df.loc[mask, "Test_std"] = long_df.loc[mask, std_col]

    # Treatment interactions
    for grp_suffix in ["5to6", "7to10", "11to14"]:
        grp_col = f"Group_{grp_suffix}"
        
        long_df[f"HS_{grp_suffix}"] = 0.0
        mask_hs = (long_df["HS2_FE90"] == 1) & (long_df[grp_col] == 1)
        long_df.loc[mask_hs, f"HS_{grp_suffix}"] = 1.0
        
        long_df[f"Pre_{grp_suffix}"] = 0.0
        mask_pre = (long_df["Pre2_FE90"] == 1) & (long_df[grp_col] == 1)
        long_df.loc[mask_pre, f"Pre_{grp_suffix}"] = 1.0
    
    # Also create for full 5 to 14
    long_df["HS_5to14"] = ((long_df["HS2_FE90"] == 1) & (long_df["Group_5to14"] == 1)).astype(float)
    long_df["Pre_5to14"] = ((long_df["Pre2_FE90"] == 1) & (long_df["Group_5to14"] == 1)).astype(float)
    
    # Create MomHS if missing
    if "MomHS" not in long_df.columns:
        long_df["MomHS"] = ((long_df["MomDropout"] == 0) & (long_df["MomSomeColl"] == 0)).astype(float)
        long_df.loc[long_df["MomDropout"].isna(), "MomHS"] = np.nan
    
    # Create lowAFQT indicator for Table 4 subgroup analysis
    long_df["lowAFQT"] = (
        (long_df["impAFQT_std"] <= -1) &
        (long_df["impAFQT_std"].notna())
    ).astype(int)

    # =========================================================================
    # 3B. CREATE SUBGROUP TREATMENT INTERACTIONS (Black / Male / lowAFQT)
    # =========================================================================
    subgroup_list = ["Black", "Male", "lowAFQT"]
    age_groups = ["5to6", "7to10", "11to14"]

    for g in subgroup_list:
        if g not in long_df.columns:
            continue  # Skip if subgroup column is missing

        g_ind = long_df[g]           # 1 = 该组, 0 = Non 该组, NaN = 缺失
        g_non = 1 - g_ind            # Non 该组，遇到 NaN 会保持 NaN

        # 按年龄段的 HS / Pre
        for ag in age_groups:
            hs_base = f"HS_{ag}"
            pre_base = f"Pre_{ag}"
            if hs_base in long_df.columns:
                long_df[f"HS_{g}_{ag}"] = long_df[hs_base] * g_ind
                long_df[f"HS_Non{g}_{ag}"] = long_df[hs_base] * g_non
            if pre_base in long_df.columns:
                long_df[f"Pre_{g}_{ag}"] = long_df[pre_base] * g_ind
                long_df[f"Pre_Non{g}_{ag}"] = long_df[pre_base] * g_non

        # 5-14 整体 HS / Pre
        if "HS_5to14" in long_df.columns:
            long_df[f"HS_{g}"] = long_df["HS_5to14"] * g_ind
            long_df[f"HS_Non{g}"] = long_df["HS_5to14"] * g_non
        if "Pre_5to14" in long_df.columns:
            long_df[f"Pre_{g}"] = long_df["Pre_5to14"] * g_ind
            long_df[f"Pre_Non{g}"] = long_df["Pre_5to14"] * g_non

    # Standardize PermInc
    if "PermInc" in long_df.columns and "PermInc_std" not in long_df.columns:
        mu = long_df["PermInc"].mean()
        sigma = long_df["PermInc"].std()
        if sigma > 0:
            long_df["PermInc_std"] = (long_df["PermInc"] - mu) / sigma
    
    # Create dummies
    year_dummies = pd.get_dummies(long_df["year"], prefix="yr", drop_first=True).astype(float)
    age_dummies = pd.get_dummies(long_df["AgeTest_Yr"], prefix="age", drop_first=True).astype(float)
    
    long_df = pd.concat([long_df, year_dummies, age_dummies], axis=1)
   
    # Age2_Yr104 dummies（用于子组回归，模仿 Stata 的 i.Age2_Yr104）
    if "Age2_Yr104" in long_df.columns:
        age2_dummies = pd.get_dummies(long_df["Age2_Yr104"], prefix="age2", drop_first=True).astype(float)
        long_df = pd.concat([long_df, age2_dummies], axis=1)
    else:
        age2_dummies = pd.DataFrame(index=long_df.index)

    # 子组回归用的控制变量：Male + i.Age2_Yr104 + Covariates（对应 Stata）
    subgroup_controls = ["Male"]
    subgroup_controls += list(age2_dummies.columns)
    subgroup_controls += covariates_list
    subgroup_controls = [c for c in subgroup_controls if c in long_df.columns]

    # =========================================================================
    # 4. REGRESSION MODELS
    # =========================================================================
    
    vars_interest = [
        "HS_5to6", "HS_7to10", "HS_11to14", 
        "Pre_5to6", "Pre_7to10", "Pre_11to14"
    ]
    
    # Base controls
    base_controls = ["Male"]
    base_controls += list(year_dummies.columns) + list(age_dummies.columns)

    
    # =========================================================================
    # 6. TABLE 4 - OVERALL (前三列：按年龄段；第四列：5-14 汇总)
    # =========================================================================
    print("\nRunning Table 4 - Overall FE with covariates (age-specific + 5-14)...")

    # ---------- 回归 A：HS_5to6 / HS_7to10 / HS_11to14 ----------
    t4_X_vars = vars_interest + base_controls + covariates_list
    req_cols_t4 = ["Test_std", "MotherID"] + t4_X_vars
    req_cols_t4 = [c for c in req_cols_t4 if c in long_df.columns]

    t4_temp = long_df.dropna(subset=req_cols_t4).copy()
    if t4_temp.empty:
        print("Warning: Table 4 overall (age-specific) has no valid observations")
        return
    y_t4 = t4_temp["Test_std"]
    X_t4 = t4_temp[t4_X_vars]

    grp_t4 = t4_temp.groupby("MotherID")
    y_dm_t4 = y_t4 - grp_t4["Test_std"].transform("mean")
    X_dm_t4 = X_t4 - grp_t4[X_t4.columns].transform("mean")

    std_vals_t4 = X_dm_t4.std()
    valid_cols_t4 = std_vals_t4[std_vals_t4 > 1e-8].index.tolist()
    X_dm_t4 = X_dm_t4[valid_cols_t4]

    model_t4 = sm.OLS(y_dm_t4, X_dm_t4)
    res_t4 = model_t4.fit(cov_type="cluster",
                          cov_kwds={"groups": t4_temp["MotherID"]})

    # ---------- 回归 B：HS_5to14 / Pre_5to14 ----------
    pooled_vars = ["HS_5to14", "Pre_5to14"]
    pooled_X_vars = pooled_vars + base_controls + covariates_list
    pooled_req_cols = ["Test_std", "MotherID"] + pooled_X_vars
    pooled_req_cols = [c for c in pooled_req_cols if c in long_df.columns]

    pooled_temp = long_df.dropna(subset=pooled_req_cols).copy()
    if pooled_temp.empty:
        print("Warning: Table 4 pooled 5-14 has no valid observations")
        return

    y_p = pooled_temp["Test_std"]
    X_p = pooled_temp[pooled_X_vars]

    grp_p = pooled_temp.groupby("MotherID")
    y_dm_p = y_p - grp_p["Test_std"].transform("mean")
    X_dm_p = X_p - grp_p[X_p.columns].transform("mean")

    std_vals_p = X_dm_p.std()
    valid_cols_p = std_vals_p[std_vals_p > 1e-8].index.tolist()
    X_dm_p = X_dm_p[valid_cols_p]

    model_p = sm.OLS(y_dm_p, X_dm_p)
    res_p = model_p.fit(cov_type="cluster",
                        cov_kwds={"groups": pooled_temp["MotherID"]})

    # ---------- 组装成 4 列：5-6, 7-10, 11-14, 5-14 ----------
    cols = ["5to6", "7to10", "11to14", "5to14"]
    table4_dict = {c: {} for c in cols}

    # 3 个年龄段：来自 res_t4
    age_map = {
        "5to6": ("HS_5to6", "Pre_5to6"),
        "7to10": ("HS_7to10", "Pre_7to10"),
        "11to14": ("HS_11to14", "Pre_11to14"),
    }
    for label, (hs_var, pre_var) in age_map.items():
        table4_dict[label]["HS_coef"] = res_t4.params.get(hs_var, np.nan)
        table4_dict[label]["HS_se"]   = res_t4.bse.get(hs_var, np.nan)
        table4_dict[label]["Pre_coef"] = res_t4.params.get(pre_var, np.nan)
        table4_dict[label]["Pre_se"]   = res_t4.bse.get(pre_var, np.nan)

    # 5-14：来自 res_p
    table4_dict["5to14"]["HS_coef"]  = res_p.params.get("HS_5to14", np.nan)
    table4_dict["5to14"]["HS_se"]    = res_p.bse.get("HS_5to14", np.nan)
    table4_dict["5to14"]["Pre_coef"] = res_p.params.get("Pre_5to14", np.nan)
    table4_dict["5to14"]["Pre_se"]   = res_p.bse.get("Pre_5to14", np.nan)

    # p 值：只对前三个年龄段做（对应 Stata 的 test）
    try:
        f_all = res_t4.f_test("HS_5to6 = HS_7to10, HS_7to10 = HS_11to14")
        p_all = float(f_all.pvalue)
    except Exception:
        p_all = np.nan

    try:
        p_5to6 = float(res_t4.f_test("HS_5to6 = Pre_5to6").pvalue)
    except Exception:
        p_5to6 = np.nan
    try:
        p_7to10 = float(res_t4.f_test("HS_7to10 = Pre_7to10").pvalue)
    except Exception:
        p_7to10 = np.nan
    try:
        p_11to14 = float(res_t4.f_test("HS_11to14 = Pre_11to14").pvalue)
    except Exception:
        p_11to14 = np.nan
    try:
        p_5to14 = float(res_p.f_test("HS_5to14 = Pre_5to14").pvalue)
    except Exception:
        p_5to14 = np.nan

    # 把 p 值和 R2/N 也放进去（你可以只放一次或每列都复制）
    meta_row = pd.Series({
        "p_AllAgeEqual": p_all,
        "p_HS_eq_Pre_5to6": p_5to6,
        "p_HS_eq_Pre_7to10": p_7to10,
        "p_HS_eq_Pre_11to14": p_11to14,
        "p_HS_eq_Pre_5to14": p_5to14,
        "R2_within_age": res_t4.rsquared,
        "N_age": int(res_t4.nobs),
        "R2_within_5to14": res_p.rsquared,
        "N_5to14": int(res_p.nobs),
    }, name="meta")

    table4_overall = pd.DataFrame(table4_dict)
    print("\n" + "="*80)
    print("TABLE 4 - OVERALL (columns: 5-6, 7-10, 11-14, 5-14)")
    print("="*80)
    print(table4_overall)
    print("\nMeta info:")
    print(meta_row)

    table4_overall.to_csv("table4_overall_with_5to14.csv")
    meta_row.to_csv("table4_overall_meta.csv")
    print("\nSaved to table4_overall_with_5to14.csv & table4_overall_meta.csv")

    # =========================================================================
    # 7. TABLE 4 - SUBGROUPS (Black / Male / lowAFQT)
    # =========================================================================
    print("\nRunning Table 4 - Subgroups: Black / Male / lowAFQT ...")

    def fe_cluster(y_var, x_vars, controls, data, group_col="MotherID"):
        needed = [y_var, group_col] + x_vars + controls
        needed = [c for c in needed if c in data.columns]
        tmp = data.dropna(subset=needed).copy()
        if tmp.empty: return None

        y = tmp[y_var]
        X = tmp[x_vars + controls]

        grp = tmp.groupby(group_col)
        y_dm = y - grp[y_var].transform("mean")
        X_dm = X - grp[X.columns].transform("mean")

        std_vals = X_dm.std()
        keep = std_vals[std_vals > 1e-8].index.tolist()
        X_dm = X_dm[keep]
        if X_dm.empty: return None

        # Helper to safely f_test inside the function context if needed, 
        # but we return result wrapper to do it outside
        return sm.OLS(y_dm, X_dm).fit(cov_type="cluster", cov_kwds={"groups": tmp[group_col]})

    # Define F-test helper
    def ftest_p(res, lhs, rhs):
        if lhs in res.params.index and rhs in res.params.index:
            try:
                return float(res.f_test(f"{lhs} = {rhs}").pvalue)
            except Exception:
                return np.nan
        return np.nan

    subgroup_list = ["Black", "Male", "lowAFQT"]
    age_groups = ["5to6", "7to10", "11to14"]

    # ---------- 7A. By Age (Existing Logic OK) ----------
    sub_age_cols = {}
    for g in subgroup_list:
        x_vars_g = []
        for ag in age_groups:
            x_vars_g += [f"HS_{g}_{ag}", f"HS_Non{g}_{ag}"]
            # Stata includes Pre interactions in the regression model even if not shown in table?
            # Stata code: xtreg ... Pre_`g'_5to6 ...
            # YES: We must include Pre variables in the regression controls to avoid bias
            x_vars_g += [f"Pre_{g}_{ag}", f"Pre_Non{g}_{ag}"]

        x_vars_g = [v for v in x_vars_g if v in long_df.columns]
        
        # We only want to EXTRACT HS coefficients for the table, but we use ALL in regression
        res_g = fe_cluster("Test_std", x_vars_g, subgroup_controls, long_df)
        if res_g is None: continue

        col = {}
        # Only save HS coefficients as per Stata output request (though Stata code creates scalars for tests)
        for ag in age_groups:
            # Save Coefs
            for prefix in [f"HS_{g}", f"HS_Non{g}"]:
                var = f"{prefix}_{ag}"
                col[f"{var}_coef"] = res_g.params.get(var, np.nan)
                col[f"{var}_se"] = res_g.bse.get(var, np.nan)
            
            # Save P-value: HS_g = HS_Nong
            lhs, rhs = f"HS_{g}_{ag}", f"HS_Non{g}_{ag}"
            col[f"p_{g}_vs_Non{g}_{ag}"] = ftest_p(res_g, lhs, rhs)

        col["R2_within"] = res_g.rsquared
        col["N"] = int(res_g.nobs)
        sub_age_cols[g] = col

    if sub_age_cols:
        pd.DataFrame(sub_age_cols).to_csv("table4_subgroups_by_age.csv")

    # ---------- 7B. 5-14 Pooled (FIXED) ----------
    sub_514_cols = {}

    for g in subgroup_list:
        # FIXED: Added Pre variables to the list
        x_vars_g514 = [f"HS_{g}", f"HS_Non{g}", f"Pre_{g}", f"Pre_Non{g}"]
        x_vars_g514 = [v for v in x_vars_g514 if v in long_df.columns]
        
        if len(x_vars_g514) < 4:
            print(f"Skipping {g} 5-14 due to missing vars")
            continue

        res_g514 = fe_cluster("Test_std", x_vars_g514, subgroup_controls, long_df)
        if res_g514 is None: continue

        col = {}
        # Save coefficients for all 4
        for v in x_vars_g514:
            col[f"{v}_coef"] = res_g514.params.get(v, np.nan)
            col[f"{v}_se"] = res_g514.bse.get(v, np.nan)

        # FIXED: Added the 3 hypothesis tests required by Stata
        # 1. HS_g = Pre_g
        col[f"p_HS_eq_Pre_{g}"] = ftest_p(res_g514, f"HS_{g}", f"Pre_{g}")
        
        # 2. HS_Nong = Pre_Nong
        col[f"p_HS_eq_Pre_Non{g}"] = ftest_p(res_g514, f"HS_Non{g}", f"Pre_Non{g}")
        
        # 3. HS_g = HS_Nong
        col[f"p_HS_{g}_eq_HS_Non{g}"] = ftest_p(res_g514, f"HS_{g}", f"HS_Non{g}")

        col["R2_within"] = res_g514.rsquared
        col["N"] = int(res_g514.nobs)

        sub_514_cols[g] = col

    if sub_514_cols:
        table4_sub_514 = pd.DataFrame(sub_514_cols)
        print("\n" + "="*80)
        print("TABLE 4 - SUBGROUPS 5-14 (Corrected)")
        print("="*80)
        print(table4_sub_514)
        table4_sub_514.to_csv("table4_subgroups_5to14.csv")

if __name__ == "__main__":
    main()
