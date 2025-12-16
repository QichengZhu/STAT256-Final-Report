import re
import pandas as pd
import numpy as np
import statsmodels.api as sm

# Input file from previous step
DATA_PATH = "../data/deming_table2_data.csv"

# ------------------------------------------------------------
# Helpers: standardize, FE-OLS with clustered SEs
# ------------------------------------------------------------
def zscore(s: pd.Series) -> pd.Series:
    mu = s.mean(skipna=True)
    sd = s.std(ddof=1, skipna=True)
    if pd.isna(sd) or sd <= 0:
        return pd.Series(np.nan, index=s.index)
    return (s - mu) / sd

def fe_ols_cluster(df: pd.DataFrame, y: str, x_vars: list, controls: list, group: str = "MotherID"):
    needed = [y, group] + x_vars + controls
    needed = [c for c in needed if c in df.columns]
    tmp = df.dropna(subset=needed).copy()
    if tmp.empty:
        return None

    yv = tmp[y]
    X = tmp[[c for c in (x_vars + controls) if c in tmp.columns]].astype(float)

    g = tmp.groupby(group)
    y_dm = yv - g[y].transform("mean")
    X_dm = X - g[X.columns].transform("mean")

    # drop near-constant columns after demeaning (common with FE)
    keep = X_dm.std(ddof=1) > 1e-8
    X_dm = X_dm.loc[:, keep]
    if X_dm.shape[1] == 0:
        return None

    res = sm.OLS(y_dm, X_dm).fit(cov_type="cluster", cov_kwds={"groups": tmp[group]})
    return res

def safe_f_pval(res, hypothesis: str) -> float:
    try:
        return float(res.f_test(hypothesis).pvalue)
    except Exception:
        return np.nan

# ------------------------------------------------------------
# Subgroup flags + subgroup treatment dummies in WIDE data
# (mirrors Stata: NonMale, lowAFQT, NonlowAFQT and HS_g/Pre_g)
# ------------------------------------------------------------
def create_subgroup_flags_wide(wide_df: pd.DataFrame) -> pd.DataFrame:
    df = wide_df.copy()

    # NonMale
    if "Male" in df.columns:
        df["NonMale"] = (df["Male"] == 0).astype(int)
    # lowAFQT uses *imputed* AFQT (Stata: impAFQT_std<=-1 & !=.)
    if "impAFQT_std" in df.columns:
        df["lowAFQT"] = ((df["impAFQT_std"] <= -1) & df["impAFQT_std"].notna()).astype(int)
        df["NonlowAFQT"] = (df["lowAFQT"] == 0).astype(int)

    # NonBlack: if not provided, derive from Black when available
    if "NonBlack" not in df.columns and "Black" in df.columns:
        df["NonBlack"] = (df["Black"] == 0).astype(int)

    # Build HS_g and Pre_g exactly like Stata loop:
    # gen x_g = 1 if x2_FE90==1 & g==1 ; else 0 if x2_FE90!=.
    groups = []
    for g in ["Male", "NonMale", "Black", "NonBlack", "lowAFQT", "NonlowAFQT"]:
        if g in df.columns:
            groups.append(g)

    for g in groups:
        for x in ["HS", "Pre"]:
            treat_col = f"{x}2_FE90"
            out = f"{x}_{g}"
            if treat_col not in df.columns:
                continue
            df[out] = np.nan
            df.loc[(df[treat_col] == 1) & (df[g] == 1), out] = 1
            # set to 0 for everyone else with non-missing treatment status
            df.loc[df[out].isna() & df[treat_col].notna(), out] = 0

    return df

# ------------------------------------------------------------
# Construct non-test + long-term outcomes in WIDE data
# (robust to missing columns: uses whatever exists)
# ------------------------------------------------------------
def create_nontest_outcomes_wide(wide_df: pd.DataFrame) -> pd.DataFrame:
    """
    Construct non-test and long-term outcomes in WIDE data (one row per child),
    following the Stata do-file logic as closely as possible.

    Outputs (when inputs exist):
      Repeat, LD, HSGrad, HSGrad_GED, someCollAtt, Idle, Crime, TeenPreg, PoorHealth
      plus intermediate: Repeat94/Repeat96/Repeat98, PosWages, InSchool, Prison*, etc.
    """
    df = wide_df.copy()

    # ============================================================
    # Grade repetition (Repeat): mimic Stata exactly
    # ============================================================
    # Fix random error: replace Repeat92=. if Repeat92==6
    if "Repeat92" in df.columns:
        df.loc[df["Repeat92"] == 6, "Repeat92"] = np.nan

    # --- Repeat94 ---
    rep94_cols = [
        "Repeat_K94", "Repeat_194", "Repeat_294", "Repeat_394", "Repeat_494",
        "Repeat_594", "Repeat_694", "Repeat_794", "Repeat_894"
    ]
    rep94_cols = [c for c in rep94_cols if c in df.columns]
    if rep94_cols:
        df["Repeat94"] = df[rep94_cols].max(axis=1, skipna=True)
        if "Repeat_None94" in df.columns:
            # replace Repeat94=0 if Repeat_None94==1 & Repeat94!=1;
            df.loc[(df["Repeat_None94"] == 1) & (df["Repeat94"] != 1), "Repeat94"] = 0

    # --- Repeat96 ---
    rep96_cols = [
        "Repeat_K96", "Repeat_196", "Repeat_296", "Repeat_396", "Repeat_496",
        "Repeat_596", "Repeat_696", "Repeat_796", "Repeat_896",
        "Repeat_996", "Repeat_1096", "Repeat_1196", "Repeat_1296"
    ]
    rep96_cols = [c for c in rep96_cols if c in df.columns]
    if rep96_cols:
        df["Repeat96"] = df[rep96_cols].max(axis=1, skipna=True)
        if "Repeat_None96" in df.columns:
            # replace Repeat96=0 if Repeat_None96==1 & Repeat96!=1;
            df.loc[(df["Repeat_None96"] == 1) & (df["Repeat96"] != 1), "Repeat96"] = 0

    # --- Repeat98 ---
    # Stata:
    # rename Repeat_None98 RepeatNone98;
    # drop Repeat_YA98;
    # foreach var of varlist Repeat_*98 { gen byte `var'_temp=1 if `var'==1; };
    # egen Repeat98=rowmax(Repeat_*98_temp);
    # replace Repeat98=0 if RepeatNone98==1;
    #
    # We'll replicate with regex selecting Repeat_<digits>98 (e.g., Repeat_198 ... Repeat_998),
    # and ignore Repeat_YA98 if present.
    if "Repeat_None98" in df.columns:
        rep98_cols = [c for c in df.columns if re.match(r"^Repeat_\d+98$", c)]
        # exclude Repeat_YA98 explicitly (even though it won't match regex if named Repeat_YA98)
        rep98_cols = [c for c in rep98_cols if c != "Repeat_YA98"]
        if rep98_cols:
            # temp = 1 if ==1 else NaN (so max acts like rowmax of 1's)
            tmp = (df[rep98_cols] == 1).astype(float)
            tmp = tmp.where(df[rep98_cols].notna(), np.nan)
            df["Repeat98"] = tmp.max(axis=1, skipna=True)
            df.loc[df["Repeat_None98"] == 1, "Repeat98"] = 0

    # Final Repeat = rowmax across available year-level Repeat variables (as Stata)
    rep_years = ["Repeat88", "Repeat90", "Repeat92", "Repeat94", "Repeat96", "Repeat98", "Repeat100", "Repeat102", "Repeat104"]
    rep_years = [c for c in rep_years if c in df.columns]
    if rep_years:
        df["Repeat"] = df[rep_years].max(axis=1, skipna=True)

    # ============================================================
    # Learning disability (LD): mimic Stata exactly
    # ============================================================
    # Stata:
    # forvalues x=86(2)100:
    #   gen tempLDx=LDx
    #   replace tempLDx=0 if HealthCondx!=. & tempLDx!=1
    # egen LD=rowmax(tempLD*)
    # exclude LD diagnosed before age 5:
    #   replace LD_before=1 if tempLDx==1 & Age2_Yrx<5
    #   replace LD=. if LD_before==1
    tempLD_cols = []
    for y in range(86, 102, 2):
        ld = f"LD{y}"
        hc = f"HealthCond{y}"
        tmp = f"tempLD{y}"
        if ld in df.columns:
            df[tmp] = df[ld]
            if hc in df.columns:
                df.loc[df[hc].notna() & (df[tmp] != 1), tmp] = 0
            tempLD_cols.append(tmp)

    if tempLD_cols:
        df["LD"] = df[tempLD_cols].max(axis=1, skipna=True)

        # Exclude LD before age 5
        df["LD_before"] = np.nan
        for y in range(86, 102, 2):
            tmp = f"tempLD{y}"
            age = f"Age2_Yr{y}"
            if tmp in df.columns and age in df.columns:
                df.loc[(df[tmp] == 1) & df[age].notna() & (df[age] < 5), "LD_before"] = 1

        df.loc[df["LD_before"] == 1, "LD"] = np.nan

        # Clean up temps
        df.drop(columns=[c for c in tempLD_cols if c in df.columns], inplace=True)
        if "LD_before" in df.columns:
            df.drop(columns=["LD_before"], inplace=True)

    # ============================================================
    # Long-term outcomes from YA variables (these were "matching" already)
    # ============================================================
    # HSGrad / someCollAtt from YA_Educ104
    if "YA_Educ104" in df.columns:
        df["HSGrad"] = np.where(df["YA_Educ104"].notna(), (df["YA_Educ104"] >= 12).astype(int), np.nan)
        df["someCollAtt"] = np.where(df["YA_Educ104"].notna(), (df["YA_Educ104"] >= 13).astype(int), np.nan)

    # GED exclusion: egen GED=rowmax(GED*) ; HSGrad_GED=. if HSGrad==1 & GED==2
    if "HSGrad" in df.columns:
        ged_cols = [c for c in df.columns if c.startswith("GED")]
        if ged_cols:
            GED = df[ged_cols].max(axis=1, skipna=True)
            df["HSGrad_GED"] = df["HSGrad"].copy()
            df.loc[(df["HSGrad_GED"] == 1) & (GED == 2), "HSGrad_GED"] = np.nan
        else:
            df["HSGrad_GED"] = df.get("HSGrad", np.nan)

    # someCollAtt alternative using HighGrade_Att*
    highgrade_cols = [c for c in df.columns if c.startswith("HighGrade_Att")]
    if "someCollAtt" in df.columns and highgrade_cols:
        HighGradeAtt = df[highgrade_cols].max(axis=1, skipna=True)
        df.loc[(HighGradeAtt > 12) & HighGradeAtt.notna(), "someCollAtt"] = 1

    # ============================================================
    # Idle: mimic Stata (including Wages_Est logic!)
    # ============================================================
    if "YA_LastInterview" in df.columns:
        # Backfilled InSchool
        if "InSchool104" in df.columns:
            df["InSchool"] = df["InSchool104"]
            for y in [102, 100, 98, 96, 94]:
                col = f"InSchool{y}"
                if col in df.columns:
                    df.loc[df["InSchool"].isna() & (df["YA_LastInterview"] == y), "InSchool"] = df[col]

        # Backfilled Wages (actual)
        if "Wages104" in df.columns:
            df["Wages"] = df["Wages104"]
            for y in [102, 100, 98, 96, 94]:
                col = f"Wages{y}"
                if col in df.columns:
                    df.loc[df["Wages"].isna() & (df["YA_LastInterview"] == y), "Wages"] = df[col]

        # Create PosWagesYY for 104/102/100/98/96/94 (Stata does loop)
        for y in [104, 102, 100, 98, 96, 94]:
            wy = f"Wages{y}"
            py = f"PosWages{y}"
            if wy in df.columns:
                df[py] = np.where(df[wy].notna(), (df[wy] > 0).astype(float), np.nan)

        # PosWages overall with estimate overrides (Stata logic)
        if "PosWages104" in df.columns:
            df["PosWages"] = df["PosWages104"]

            # If respondent did not know wages but reported estimate:
            # replace PosWages=0 if PosWages104==. & Wages_Est104==1
            # replace PosWages=1 if PosWages104==. & Wages_Est104>1
            if "Wages_Est104" in df.columns:
                df.loc[df["PosWages"].isna() & (df["Wages_Est104"] == 1), "PosWages"] = 0
                df.loc[df["PosWages"].isna() & df["Wages_Est104"].notna() & (df["Wages_Est104"] > 1), "PosWages"] = 1

            # Backfill to 2002 / 2000 using their own estimate codes
            if "YA_LastInterview" in df.columns:
                if "PosWages102" in df.columns:
                    mask = df["PosWages"].isna() & (df["YA_LastInterview"] == 2002)
                    df.loc[mask, "PosWages"] = df.loc[mask, "PosWages102"]
                    if "Wages_Est102" in df.columns:
                        mask2 = df["PosWages"].isna() & (df["YA_LastInterview"] == 2002)
                        df.loc[mask2 & (df["Wages_Est102"] == 1), "PosWages"] = 0
                        df.loc[mask2 & df["Wages_Est102"].notna() & (df["Wages_Est102"] > 1), "PosWages"] = 1

                if "PosWages100" in df.columns:
                    mask = df["PosWages"].isna() & (df["YA_LastInterview"] == 2000)
                    df.loc[mask, "PosWages"] = df.loc[mask, "PosWages100"]
                    if "Wages_Est100" in df.columns:
                        mask2 = df["PosWages"].isna() & (df["YA_LastInterview"] == 2000)
                        df.loc[mask2 & (df["Wages_Est100"] == 1), "PosWages"] = 0
                        df.loc[mask2 & df["Wages_Est100"].notna() & (df["Wages_Est100"] > 1), "PosWages"] = 1

                # Earlier years: no estimate code
                for y, last in [(98, 1998), (96, 1996), (94, 1994)]:
                    py = f"PosWages{y}"
                    if py in df.columns:
                        mask = df["PosWages"].isna() & (df["YA_LastInterview"] == last)
                        df.loc[mask, "PosWages"] = df.loc[mask, py]

        # Idle definition
        if "InSchool" in df.columns and "PosWages" in df.columns:
            df["Idle"] = np.nan
            df.loc[(df["InSchool"] == 0) & (df["PosWages"] == 0), "Idle"] = 1
            df.loc[df["Idle"].isna() & df["InSchool"].notna(), "Idle"] = 0

    # ============================================================
    # Crime: mimic Stata (construct PrisonYY from ResidYY==5)
    # ============================================================
    # Convicted / Probation / Sentenced are rowmax across waves
    conv_cols = [c for c in df.columns if c.startswith("Convicted")]
    prob_cols = [c for c in df.columns if c.startswith("Probation")]
    sent_cols = [c for c in df.columns if c.startswith("Sentenced")]

    Convicted = df[conv_cols].max(axis=1, skipna=True) if conv_cols else None
    Probation = df[prob_cols].max(axis=1, skipna=True) if prob_cols else None
    Sentenced = df[sent_cols].max(axis=1, skipna=True) if sent_cols else None

    # PrisonYY from ResidYY==5, then Prison=rowmax(PrisonYY)
    prison_year_cols = []
    for y in [94, 96, 98, 100, 102, 104]:
        ry = f"Resid{y}"
        py = f"Prison{y}"
        if ry in df.columns:
            df[py] = np.nan
            df.loc[df[ry] == 5, py] = 1
            df.loc[df[ry].notna() & df[py].isna(), py] = 0
            prison_year_cols.append(py)

    Prison = df[prison_year_cols].max(axis=1, skipna=True) if prison_year_cols else None

    # Crime = rowmax(Convicted Probation Sentenced Prison)
    parts = []
    for s in [Convicted, Probation, Sentenced, Prison]:
        if s is not None:
            parts.append(s)
    if parts:
        df["Crime"] = pd.concat(parts, axis=1).max(axis=1, skipna=True)

    # ============================================================
    # TeenPreg (same as your version)
    # ============================================================
    if "Ageat1stBirth" in df.columns:
        df["TeenPreg"] = np.nan
        df.loc[df["Ageat1stBirth"].notna(), "TeenPreg"] = (df["Ageat1stBirth"] < 20).astype(int)
        if "YA_NumKids" in df.columns:
            df.loc[df["TeenPreg"].isna() & df["YA_NumKids"].notna(), "TeenPreg"] = 0

    # ============================================================
    # PoorHealth (same as your version)
    # ============================================================
    healthrep_cols = [c for c in df.columns if c.startswith("Health_Report")]
    if healthrep_cols:
        HealthReport = df[healthrep_cols].mean(axis=1, skipna=True)
        df["PoorHealth"] = np.where(HealthReport.notna(), (HealthReport < 3).astype(int), np.nan)

    return df

# ------------------------------------------------------------
# Build indices used in Table 4 col(5) and col(6)
# - Nontest score index: Repeat + LD  (reverse sign so higher is "good")
# - Long-term index: HSGrad, someCollAtt, Idle, Crime, TeenPreg, PoorHealth (reverse "bad" ones)
# ------------------------------------------------------------
def create_indices_wide(wide_df: pd.DataFrame) -> pd.DataFrame:
    df = wide_df.copy()

    # Standardize component outcomes (within Sample90_2==1 is handled outside)
    for v in ["Repeat", "LD", "HSGrad", "someCollAtt", "Idle", "Crime", "TeenPreg", "PoorHealth", "HSGrad_GED"]:
        if v in df.columns:
            df[f"{v}_std"] = zscore(df[v])

    # Reverse sign so positive = good (as Stata)
    for bad in ["Repeat", "LD", "Idle", "Crime", "TeenPreg", "PoorHealth"]:
        col = f"{bad}_std"
        if col in df.columns:
            df[col] = -df[col]

    # Nontest index (school-age): mean(Repeat_std, LD_std), then restandardize
    nontest_parts = [c for c in ["Repeat_std", "LD_std"] if c in df.columns]
    if nontest_parts:
        tmp = df[nontest_parts].mean(axis=1, skipna=True)
        df["Noncog_raw"] = tmp
        df["Noncog_std"] = zscore(tmp)

    # Long-term index (young adult): mean(HSGrad_std, someCollAtt_std, Idle_std, Crime_std, TeenPreg_std, PoorHealth_std)
    adult_parts = [c for c in ["HSGrad_std", "someCollAtt_std", "Idle_std", "Crime_std", "TeenPreg_std", "PoorHealth_std"]
                  if c in df.columns]
    if adult_parts:
        tmp = df[adult_parts].mean(axis=1, skipna=True)
        df["Sum_Adult_raw"] = tmp
        df["Sum_Adult"] = zscore(tmp)

    return df

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
        
    # ============================================================
    # CREATE WIDE DF
    # ============================================================
    wide_df = df.copy()
    
    
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
    wide_covariates_list = covariates_list.copy()
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
    
    wide_covariates_list = covariates_list.copy()

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
     
    # =========================================================================
    # 8. TABLE 4 col(5)(6): NON-TEST index + LONG-TERM index (WIDE)
    # =========================================================================
    print("\nCreating non-test and long-term outcomes (wide) ...")

    # Match Stata restriction: Sample90_2==1
    if "Sample90_2" in wide_df.columns:
        wide_df = wide_df[wide_df["Sample90_2"] == 1].copy()

    # Build subgroup flags + HS_g/Pre_g
    wide_df = create_subgroup_flags_wide(wide_df)

    # Build outcomes and indices
    wide_df = create_nontest_outcomes_wide(wide_df)
    wide_df = create_indices_wide(wide_df)

    # Controls for WIDE regressions: Male + i.Age2_Yr104 + covariates_list
    wide_controls = []
    if "Male" in wide_df.columns:
        wide_controls.append("Male")
    if "Age2_Yr104" in wide_df.columns:
        age2dum = pd.get_dummies(wide_df["Age2_Yr104"], prefix="age2", drop_first=True).astype(float)
        wide_df = pd.concat([wide_df, age2dum], axis=1)
        wide_controls += list(age2dum.columns)
    wide_controls += [c for c in wide_covariates_list if c in wide_df.columns]

    # ---- Overall: Noncog_std and Sum_Adult ----
    out_rows = []
    for yvar, label in [("Noncog_std", "Table4_col5_NontestIndex"),
                        ("Sum_Adult", "Table4_col6_LongTermIndex")]:
        if yvar not in wide_df.columns:
            continue
        res = fe_ols_cluster(
            wide_df, y=yvar,
            x_vars=["HS2_FE90", "Pre2_FE90"],
            controls=wide_controls,
            group="MotherID"
        )
        if res is None:
            continue
        out_rows.append({
            "block": label,
            "HS_coef": res.params.get("HS2_FE90", np.nan),
            "HS_se":   res.bse.get("HS2_FE90", np.nan),
            "Pre_coef": res.params.get("Pre2_FE90", np.nan),
            "Pre_se":   res.bse.get("Pre2_FE90", np.nan),
            "p_HS_eq_Pre": safe_f_pval(res, "HS2_FE90 = Pre2_FE90"),
            "R2_within": res.rsquared,
            "N": int(res.nobs)
        })

    table4_non_test_long = pd.DataFrame(out_rows)
    table4_non_test_long.to_csv("table4_col5_col6_overall.csv", index=False)
    print("Saved: table4_col5_col6_overall.csv")

    # ---- Subgroups (Table 4 Panels B–D, columns 5–6 style) ----
    # Regression form: y ~ HS_g + HS_Nong + Pre_g + Pre_Nong + controls + FE
    subgroups = ["Black", "Male", "lowAFQT"]
    sub_rows = []
    for yvar, label in [("Noncog_std", "col5_NontestIndex"),
                        ("Sum_Adult", "col6_LongTermIndex")]:
        if yvar not in wide_df.columns:
            continue

        for g in subgroups:
            need = [f"HS_{g}", f"HS_Non{g}", f"Pre_{g}", f"Pre_Non{g}"]
            need = [v for v in need if v in wide_df.columns]
            if len(need) < 4:
                continue

            res = fe_ols_cluster(wide_df, y=yvar, x_vars=need, controls=wide_controls, group="MotherID")
            if res is None:
                continue

            sub_rows.append({
                "outcome": label,
                "group": g,
                f"HS_{g}_coef": res.params.get(f"HS_{g}", np.nan),
                f"HS_{g}_se":   res.bse.get(f"HS_{g}", np.nan),
                f"HS_Non{g}_coef": res.params.get(f"HS_Non{g}", np.nan),
                f"HS_Non{g}_se":   res.bse.get(f"HS_Non{g}", np.nan),
                "p_group_eq_nongroup": safe_f_pval(res, f"HS_{g} = HS_Non{g}"),
                "p_HS_eq_Pre_group": safe_f_pval(res, f"HS_{g} = Pre_{g}"),
                "p_HS_eq_Pre_nongroup": safe_f_pval(res, f"HS_Non{g} = Pre_Non{g}"),
                "R2_within": res.rsquared,
                "N": int(res.nobs),
            })

    table4_sub_col56 = pd.DataFrame(sub_rows)
    table4_sub_col56.to_csv("table4_col5_col6_subgroups.csv", index=False)
    print("Saved: table4_col5_col6_subgroups.csv")

    # =========================================================================
    # 9. TABLE 5: Individual outcomes (overall + subgroups), WIDE FE
    # =========================================================================
    outcomes = ["Repeat", "LD", "HSGrad", "HSGrad_GED", "someCollAtt", "Idle", "Crime", "TeenPreg", "PoorHealth"]
    outcomes = [o for o in outcomes if o in wide_df.columns]

    t5_rows = []
    for o in outcomes:
        # Overall
        res = fe_ols_cluster(wide_df, y=o, x_vars=["HS2_FE90", "Pre2_FE90"], controls=wide_controls, group="MotherID")
        if res is not None:
            t5_rows.append({
                "outcome": o,
                "spec": "overall",
                "HS_coef": res.params.get("HS2_FE90", np.nan),
                "HS_se": res.bse.get("HS2_FE90", np.nan),
                "Pre_coef": res.params.get("Pre2_FE90", np.nan),
                "Pre_se": res.bse.get("Pre2_FE90", np.nan),
                "p_HS_eq_Pre": safe_f_pval(res, "HS2_FE90 = Pre2_FE90"),
                "N": int(res.nobs)
            })

        # Subgroups
        for g in ["Black", "Male", "lowAFQT"]:
            xg = [f"HS_{g}", f"HS_Non{g}", f"Pre_{g}", f"Pre_Non{g}"]
            xg = [v for v in xg if v in wide_df.columns]
            if len(xg) < 4:
                continue
            resg = fe_ols_cluster(wide_df, y=o, x_vars=xg, controls=wide_controls, group="MotherID")
            if resg is None:
                continue
            t5_rows.append({
                "outcome": o,
                "spec": f"subgroup_{g}",
                f"HS_{g}_coef": resg.params.get(f"HS_{g}", np.nan),
                f"HS_{g}_se":   resg.bse.get(f"HS_{g}", np.nan),
                f"HS_Non{g}_coef": resg.params.get(f"HS_Non{g}", np.nan),
                f"HS_Non{g}_se":   resg.bse.get(f"HS_Non{g}", np.nan),
                "p_HS_eq_Pre_group": safe_f_pval(resg, f"HS_{g} = Pre_{g}"),
                "p_HS_eq_Pre_nongroup": safe_f_pval(resg, f"HS_Non{g} = Pre_Non{g}"),
                "p_HS_group_eq_HS_nongroup": safe_f_pval(resg, f"HS_{g} = HS_Non{g}"),
                "N": int(resg.nobs)
            })

    table5 = pd.DataFrame(t5_rows)
    table5.to_csv("table5_individual_outcomes.csv", index=False)
    print("Saved: table5_individual_outcomes.csv")

if __name__ == "__main__":
    main()
