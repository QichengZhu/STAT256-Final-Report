
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

    results_table = []

    def run_model(model_name, X_vars, use_fe=False):
        """Run OLS or FE regression"""
        req_cols = ["Test_std", "MotherID"] + vars_interest + X_vars
        req_cols = [c for c in req_cols if c in long_df.columns]
        
        temp = long_df.dropna(subset=req_cols).copy()
        
        if temp.empty:
            print(f"Warning: {model_name} has no valid observations")
            return
        
        y = temp["Test_std"]
        X_vars_exist = [v for v in (vars_interest + X_vars) if v in temp.columns]
        X = temp[X_vars_exist]
        
        if use_fe:
            grp = temp.groupby("MotherID")
            y_dm = y - grp["Test_std"].transform("mean")
            X_dm = X - grp[X.columns].transform("mean")
            
            std_vals = X_dm.std()
            valid_cols = std_vals[std_vals > 1e-8].index.tolist()
            X_dm = X_dm[valid_cols]
            
            if X_dm.empty or len(X_dm.columns) == 0:
                print(f"Warning: {model_name} - all variables dropped in FE")
                return
            
            model = sm.OLS(y_dm, X_dm)
            res = model.fit(cov_type='cluster', cov_kwds={'groups': temp['MotherID']})
            
        else:
            X = sm.add_constant(X)
            model = sm.OLS(y, X)
            res = model.fit(cov_type='cluster', cov_kwds={'groups': temp['MotherID']})

        # Test equality of HS coefficients across age groups
        p_val_diff = np.nan
        try:
            hypotheses = "HS_5to6 = HS_7to10, HS_7to10 = HS_11to14"
            f_test = res.f_test(hypotheses)
            p_val_diff = f_test.pvalue
        except Exception as e:
            print(f"F-test failed for {model_name}: {e}")

        row = {"Model": model_name, "N": int(res.nobs), "R2": res.rsquared, "P_Equality": p_val_diff}
        for v in vars_interest:
            if v in res.params:
                row[f"{v}_coef"] = res.params[v]
                row[f"{v}_se"] = res.bse[v]
            else:
                row[f"{v}_coef"] = np.nan
                row[f"{v}_se"] = np.nan
        
        results_table.append(row)
    
    print("\nRunning Regression 1: OLS, No Controls...")
    run_model("1_OLS_NoControls", base_controls, use_fe=False)

    print("Running Regression 2: OLS + Pre-Treat Covariates...")
    run_model("2_OLS_PreTreat", base_controls + covariates_list, use_fe=False)

    print("Running Regression 3: OLS + Income/AFQT/Ed...")
    ses_vars = ["PermInc_std", "impAFQT_std", "MomHS", "MomSomeColl"]
    ses_vars = [v for v in ses_vars if v in long_df.columns]
    run_model("3_OLS_Full", base_controls + covariates_list + ses_vars, use_fe=False)

    print("Running Regression 4: FE, No Covariates...")
    run_model("4_FE_NoControls", base_controls, use_fe=True)

    print("Running Regression 5: FE + Pre-Treat Covariates...")
    run_model("5_FE_PreTreat", base_controls + covariates_list, use_fe=True)
    
    # =========================================================================
    # 5. OUTPUT
    # =========================================================================
    if results_table:
        out_df = pd.DataFrame(results_table)
        out_df = out_df.set_index("Model").T
        
        print("\n" + "="*80)
        print("TABLE 3 REPLICATION RESULTS")
        print("="*80)
        print(out_df)
        
        out_df.to_csv("table3_results.csv")
        print("\nSaved to table3_results.csv")
    else:
        print("No results generated - check data and variable names")

if __name__ == "__main__":
    main()
