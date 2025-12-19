"""
IPW re-analysis for Deming (2009) Head Start sibling FE design.

What this does
--------------
1) Loads the "analysis-ready" child-level CSV you already use for Tables 2–5.
2) Constructs the same young-adult and non-test indices used in Table 4 col(5)(6),
   using the wide (one-row-per-child) logic.
3) Estimates propensity scores for a binary contrast (default: HS vs Other Preschool).
4) Builds stabilized IPW weights + trims extreme weights.
5) Runs a *weighted sibling fixed-effects regression* (within-MotherID) with cluster-robust SEs.
6) Prints effect estimates + simple balance diagnostics (SMD before/after weighting).

Usage
-----
python ipw_reanalysis_headstart.py \
  --data ../data/deming_table2_data.csv \
  --contrast hs_vs_pre \
  --outcome Sum_Adult \
  --trim 0.01

Contrasts (binary)
------------------
- hs_vs_pre : HS2_FE90==1 vs Pre2_FE90==1   (drops None)
- hs_vs_none: HS2_FE90==1 vs None2_FE90==1  (drops Pre)
- hs_vs_nonhs: HS2_FE90==1 vs (Pre2_FE90==1 or None2_FE90==1)

Outcomes
--------
- Sum_Adult   : long-term (young adult) index (positive = good)
- Noncog_std  : school-age non-test index (positive = good)
- Or any raw component outcome created by create_nontest_outcomes_wide()
  e.g., Repeat, LD, HSGrad, someCollAtt, Idle, Crime, TeenPreg, PoorHealth

Notes
-----
- This is NOT a randomized experiment. IPW here relies on an "as-if unconfounded
  given observed covariates + sibling FE" story. You should justify this in your writeup:
  within-family comparisons remove all shared family-level confounding, and IPW aims to
  address remaining within-family selection on observed child-specific covariates.
"""
from __future__ import annotations

import argparse
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm


# ----------------------------
# Utilities
# ----------------------------
def zscore(s: pd.Series) -> pd.Series:
    mu = s.mean(skipna=True)
    sd = s.std(ddof=1, skipna=True)
    if pd.isna(sd) or sd <= 0:
        return pd.Series(np.nan, index=s.index)
    return (s - mu) / sd


def safe_logit_fit(X: pd.DataFrame, y: pd.Series):
    """GLM Binomial is more stable than Logit for some separations."""
    Xc = sm.add_constant(X, has_constant="add")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = sm.GLM(y.astype(float), Xc.astype(float), family=sm.families.Binomial())
        res = model.fit(maxiter=200, disp=0)
    return res


def winsorize_weights(w: pd.Series, trim: float) -> pd.Series:
    if trim <= 0:
        return w
    lo, hi = w.quantile(trim), w.quantile(1 - trim)
    return w.clip(lower=lo, upper=hi)


def smd(x: pd.Series, t: pd.Series, w: pd.Series | None = None) -> float:
    """Standardized mean difference between treated/control for a covariate."""
    m = (~x.isna()) & (~t.isna())
    x = x[m]
    t = t[m]
    if w is not None:
        w = w[m]

    if x.empty:
        return np.nan

    if w is None:
        mt = x[t == 1].mean()
        mc = x[t == 0].mean()
        vt = x[t == 1].var(ddof=1)
        vc = x[t == 0].var(ddof=1)
    else:
        wt = w[t == 1]
        wc = w[t == 0]
        xt = x[t == 1]
        xc = x[t == 0]
        mt = np.average(xt, weights=wt) if len(xt) else np.nan
        mc = np.average(xc, weights=wc) if len(xc) else np.nan
        vt = np.average((xt - mt) ** 2, weights=wt) if len(xt) else np.nan
        vc = np.average((xc - mc) ** 2, weights=wc) if len(xc) else np.nan

    denom = np.sqrt((vt + vc) / 2) if (vt is not None and vc is not None) else np.nan
    if denom is None or np.isnan(denom) or denom == 0:
        return np.nan
    return float((mt - mc) / denom)


# ----------------------------
# Outcomes (wide)
# ----------------------------
def create_nontest_outcomes_wide(wide_df: pd.DataFrame) -> pd.DataFrame:
    """
    Minimal, robust version: constructs the outcomes needed for Table 4 col(5)(6).
    It uses whatever columns exist; missing inputs -> missing outputs.

    Outcomes:
      Repeat, LD, HSGrad, someCollAtt, Idle, Crime, TeenPreg, PoorHealth
    """
    df = wide_df.copy()

    # Repeat: if already present, keep; else try rowmax of RepeatYY
    if "Repeat" not in df.columns:
        rep_years = [c for c in df.columns if c.startswith("Repeat") and c[6:].isdigit()]
        if rep_years:
            df["Repeat"] = df[rep_years].max(axis=1, skipna=True)

    # LD: if already present keep; else rowmax of LDYY
    if "LD" not in df.columns:
        ld_years = [c for c in df.columns if c.startswith("LD") and c[2:].isdigit()]
        if ld_years:
            df["LD"] = df[ld_years].max(axis=1, skipna=True)

    # Education: YA_Educ104 thresholding
    if "YA_Educ104" in df.columns:
        df["HSGrad"] = np.where(df["YA_Educ104"].notna(), (df["YA_Educ104"] >= 12).astype(float), np.nan)
        df["someCollAtt"] = np.where(df["YA_Educ104"].notna(), (df["YA_Educ104"] >= 13).astype(float), np.nan)

    # Idle (simplified): InSchool104==0 and Wages104<=0
    if "InSchool104" in df.columns and "Wages104" in df.columns:
        posw = np.where(df["Wages104"].notna(), (df["Wages104"] > 0).astype(float), np.nan)
        df["Idle"] = np.where(
            df["InSchool104"].notna() & (df["InSchool104"] == 0) & (posw == 0),
            1.0,
            np.where(df["InSchool104"].notna(), 0.0, np.nan),
        )

    # Crime (simplified): rowmax across conviction/probation/sentenced/prison proxies
    conv = [c for c in df.columns if c.startswith("Convicted")]
    prob = [c for c in df.columns if c.startswith("Probation")]
    sent = [c for c in df.columns if c.startswith("Sentenced")]
    prison = []
    for y in [94, 96, 98, 100, 102, 104]:
        ry = f"Resid{y}"
        if ry in df.columns:
            py = f"Prison{y}"
            df[py] = np.where(df[ry].notna(), (df[ry] == 5).astype(float), np.nan)
            prison.append(py)

    parts = []
    for cols in [conv, prob, sent, prison]:
        if cols:
            parts.append(df[cols].max(axis=1, skipna=True))
    if parts:
        df["Crime"] = pd.concat(parts, axis=1).max(axis=1, skipna=True)

    # TeenPreg
    if "Ageat1stBirth" in df.columns:
        df["TeenPreg"] = np.where(df["Ageat1stBirth"].notna(), (df["Ageat1stBirth"] < 20).astype(float), np.nan)
        if "YA_NumKids" in df.columns:
            df.loc[df["TeenPreg"].isna() & df["YA_NumKids"].notna(), "TeenPreg"] = 0.0

    # PoorHealth from Health_Report*
    hcols = [c for c in df.columns if c.startswith("Health_Report")]
    if hcols:
        hr = df[hcols].mean(axis=1, skipna=True)
        df["PoorHealth"] = np.where(hr.notna(), (hr < 3).astype(float), np.nan)

    return df


def create_indices_wide(wide_df: pd.DataFrame) -> pd.DataFrame:
    """Construct Noncog_std and Sum_Adult with 'good' direction (bad outcomes negated)."""
    df = wide_df.copy()

    comps = ["Repeat", "LD", "HSGrad", "someCollAtt", "Idle", "Crime", "TeenPreg", "PoorHealth"]
    for v in comps:
        if v in df.columns:
            df[f"{v}_std"] = zscore(df[v])

    # reverse sign for "bad" ones so higher = better
    for bad in ["Repeat", "LD", "Idle", "Crime", "TeenPreg", "PoorHealth"]:
        c = f"{bad}_std"
        if c in df.columns:
            df[c] = -df[c]

    # Non-test index: mean(Repeat_std, LD_std) then standardize
    non_parts = [c for c in ["Repeat_std", "LD_std"] if c in df.columns]
    if non_parts:
        raw = df[non_parts].mean(axis=1, skipna=True)
        df["Noncog_std"] = zscore(raw)

    # Long-term index: mean(HSGrad_std, someCollAtt_std, Idle_std, Crime_std, TeenPreg_std, PoorHealth_std) then standardize
    adult_parts = [c for c in ["HSGrad_std", "someCollAtt_std", "Idle_std", "Crime_std", "TeenPreg_std", "PoorHealth_std"] if c in df.columns]
    if adult_parts:
        raw = df[adult_parts].mean(axis=1, skipna=True)
        df["Sum_Adult"] = zscore(raw)

    return df


# ----------------------------
# Treatment construction
# ----------------------------
def build_binary_contrast(df: pd.DataFrame, contrast: str):
    """
    Returns (t, keep_mask) where t is 0/1, keep_mask selects rows used.
    Requires FE-sample treatment indicators to exist.
    """
    need = ["HS2_FE90", "Pre2_FE90"]
    for c in need:
        if c not in df.columns:
            raise KeyError(f"Missing required column: {c}")

    hs = df["HS2_FE90"]
    pre = df["Pre2_FE90"]
    none = df["None2_FE90"] if "None2_FE90" in df.columns else None

    if contrast == "hs_vs_pre":
        keep = (hs == 1) | (pre == 1)
        t = np.where(hs == 1, 1.0, np.where(pre == 1, 0.0, np.nan))
    elif contrast == "hs_vs_none":
        if none is None:
            raise KeyError("Need None2_FE90 for hs_vs_none contrast.")
        keep = (hs == 1) | (none == 1)
        t = np.where(hs == 1, 1.0, np.where(none == 1, 0.0, np.nan))
    elif contrast == "hs_vs_nonhs":
        if none is None:
            # if None2_FE90 missing, treat "nonhs" as Pre only
            keep = (hs == 1) | (pre == 1)
            t = np.where(hs == 1, 1.0, np.where(pre == 1, 0.0, np.nan))
        else:
            keep = (hs == 1) | (pre == 1) | (none == 1)
            nonhs = ((pre == 1) | (none == 1)).astype(float)
            t = np.where(hs == 1, 1.0, np.where(nonhs == 1, 0.0, np.nan))
    else:
        raise ValueError(f"Unknown contrast: {contrast}")

    return pd.Series(t, index=df.index), keep


# ----------------------------
# Weighted sibling FE regression
# ----------------------------
def weighted_within_fe(df: pd.DataFrame, y: str, t: str, controls: list[str], w: str, group: str = "MotherID"):
    """
    Weighted within transformation:
      y~ = y - E_w[y | group]
      X~ = X - E_w[X | group]
    then run WLS on demeaned data with cluster-robust SEs by group.
    """
    cols = [y, t, w, group] + controls
    cols = [c for c in cols if c in df.columns]
    tmp = df[cols].dropna().copy()
    if tmp.empty:
        raise ValueError("No complete cases after dropping NA.")

    # Build design matrix
    X = tmp[[t] + controls].astype(float)
    yv = tmp[y].astype(float)
    wt = tmp[w].astype(float)

    # weighted group means
    g = tmp[group]
    # numerator and denominator for each group
    denom = wt.groupby(g).transform("sum")
    # guard: denom 0 (shouldn't happen)
    denom = denom.replace(0, np.nan)

    ybar = (yv * wt).groupby(g).transform("sum") / denom
    Xbar = (X.mul(wt, axis=0)).groupby(g).transform("sum").div(denom, axis=0)

    y_dm = yv - ybar
    X_dm = X - Xbar

    # drop near-constant columns after demeaning
    keep = X_dm.std(ddof=1) > 1e-8
    X_dm = X_dm.loc[:, keep]
    if t not in X_dm.columns:
        raise ValueError("Treatment column dropped as near-constant after demeaning (no within-family variation).")

    res = sm.WLS(y_dm, X_dm, weights=wt).fit(
        cov_type="cluster",
        cov_kwds={"groups": g},
    )
    return res, tmp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default="../data/deming_table2_data.csv")
    ap.add_argument("--contrast", type=str, default="hs_vs_none", choices=["hs_vs_pre", "hs_vs_none", "hs_vs_nonhs"])
    ap.add_argument("--outcome", type=str, default="Sum_Adult")
    ap.add_argument("--trim", type=float, default=0.01, help="Winsorize weights at [trim, 1-trim]. Set 0 to disable.")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    np.random.seed(args.seed)

    print(f"Loading data: {args.data}")
    df = pd.read_csv(args.data)

    # Baseline sample restriction used throughout your replication:
    # - Sample90_2 == 1
    # - Attrit == 0 (if present)
    if "Sample90_2" in df.columns:
        df = df[df["Sample90_2"] == 1].copy()
    if "Attrit" in df.columns:
        df = df[df["Attrit"] == 0].copy()

    # Build outcomes + indices in wide df
    df = create_nontest_outcomes_wide(df)
    df = create_indices_wide(df)
    
    # --- Maternal AFQT subgroup (Deming definition) ---
    if "impAFQT_std" in df.columns:
        df["lowAFQT"] = ((df["impAFQT_std"] <= -1) & df["impAFQT_std"].notna()).astype(int)

    if args.outcome not in df.columns:
        raise KeyError(f"Outcome '{args.outcome}' not found/constructed. Available: {sorted(df.columns)}")

    # Construct binary treatment and keep rows
    t, keep = build_binary_contrast(df, args.contrast)
    df = df.loc[keep].copy()
    df["T"] = t.loc[df.index].astype(float)

    # Covariates for propensity model:
    # - use all *_imp and *_miss covariates if present (mirrors your replication setup)
    # - plus baseline demographics that vary within family (Male, birth order, etc.) if present
    covs = []
    covs += [c for c in df.columns if c.endswith("_imp")]
    covs += [c for c in df.columns if c.endswith("_miss")]
    for c in ["Male", "Black", "Hispanic", "NonBlack", "FirstBorn", "Age2_Yr104", "PermInc", "impAFQT_std"]:
        if c in df.columns and c not in covs:
            covs.append(c)

    # Expand Age2_Yr104 as dummies for flexibility (optional)
    if "Age2_Yr104" in df.columns:
        dums = pd.get_dummies(df["Age2_Yr104"], prefix="age2", drop_first=True)
        for c in dums.columns:
            df[c] = dums[c].astype(float)
            covs.append(c)
        covs = [c for c in covs if c != "Age2_Yr104"]

    # Drop obviously empty covariates
    covs = [c for c in covs if c in df.columns and df[c].notna().any()]
    covs = sorted(set(covs))
    
    if "impAFQT_std" in df.columns and "impAFQT_std" not in covs:
        covs.append("impAFQT_std")
        covs = sorted(set(covs))

    # Build complete-case dataset for propensity model
    need = ["T", "MotherID", args.outcome] + [c for c in covs if c != "impAFQT_std"]
    need = [c for c in need if c in df.columns]
    tmp_ps = df[need].dropna().copy()
    if "impAFQT_std" in tmp_ps.columns:
        tmp_ps["lowAFQT"] = ((tmp_ps["impAFQT_std"] <= -1) & tmp_ps["impAFQT_std"].notna()).astype(int)
    if tmp_ps.empty:
        raise ValueError("No complete cases for propensity model after dropping NA.")

    # Fit propensity: P(T=1 | X)
    X = tmp_ps[covs].astype(float)
    y = tmp_ps["T"].astype(int)

    ps_res = safe_logit_fit(X, y)
    e = ps_res.predict(sm.add_constant(X, has_constant="add"))
    e = np.clip(e, 1e-4, 1 - 1e-4)

    tmp_ps["ps"] = e

    # Stabilized weights
    p_t = y.mean()
    sw = np.where(y == 1, p_t / e, (1 - p_t) / (1 - e))
    tmp_ps["w"] = sw
    tmp_ps["w"] = winsorize_weights(tmp_ps["w"], args.trim)

    print("\nPropensity model fit:")
    print(ps_res.summary().as_text())

    print("\nWeights summary (after trimming):")
    print(tmp_ps["w"].describe(percentiles=[0.01, 0.05, 0.5, 0.95, 0.99]).to_string())

    # Balance diagnostics: SMD before vs after weighting (top 20 by |SMD|)
    smds = []
    for c in covs:
        s0 = smd(tmp_ps[c], tmp_ps["T"])
        s1 = smd(tmp_ps[c], tmp_ps["T"], w=tmp_ps["w"])
        smds.append((c, s0, s1))
    bal = pd.DataFrame(smds, columns=["covariate", "smd_unweighted", "smd_weighted"])
    bal["abs_unw"] = bal["smd_unweighted"].abs()
    bal = bal.sort_values("abs_unw", ascending=False)
    print("\nTop covariates by |SMD| (unweighted -> weighted):")
    print(bal.head(20)[["covariate", "smd_unweighted", "smd_weighted"]].to_string(index=False))

    # Weighted sibling FE regression: outcome ~ T + (optional) within-varying controls
    # For IPW, typically you don't also control for X (double-robust would).
    # Here we keep a *very small* set of optional within-family controls if present.
    optional_controls = [c for c in ["Male", "FirstBorn"] if c in tmp_ps.columns]

    res, used = weighted_within_fe(
        tmp_ps,
        y=args.outcome,
        t="T",
        controls=optional_controls,
        w="w",
        group="MotherID",
    )

    print("\nWeighted sibling FE result:")
    print(res.summary().as_text())
    
    
    # ============================================================
    # Subgroup IPW-FE estimates (HS vs None) for Sum_Adult
    # ============================================================
    subgroups = {
        "Black": ("Black", 1),
        "NonBlack": ("NonBlack", 1),
        "Hispanic": ("Hispanic", 1),
        "Male": ("Male", 1),
        "Female": ("Male", 0),
        "LowAFQT": ("lowAFQT", 1),
        "NonLowAFQT": ("lowAFQT", 0),
    }

    # (optional) create lowAFQT if it doesn't exist but impAFQT_std does
    if "lowAFQT" not in tmp_ps.columns and "impAFQT_std" in tmp_ps.columns:
        tmp_ps["lowAFQT"] = ((tmp_ps["impAFQT_std"] <= -1) & tmp_ps["impAFQT_std"].notna()).astype(int)

    rows = []
    for name, (col, val) in subgroups.items():
        if col not in tmp_ps.columns:
            continue

        sub = tmp_ps[tmp_ps[col] == val].copy()
        # need within-family variation in T to identify FE effect
        fam_var = sub.groupby("MotherID")["T"].transform("nunique")
        sub = sub[fam_var >= 2].copy()
        if sub.empty:
            continue

        try:
            sub_res, _ = weighted_within_fe(
                sub,
                y=args.outcome,
                t="T",
                controls=optional_controls,   # same controls as overall
                w="w",
                group="MotherID",
            )
        except Exception as e:
            print(f"Subgroup {name} failed: {e}")
            continue

        rows.append({
            "subgroup": name,
            "beta": float(sub_res.params.get("T", np.nan)),
            "se": float(sub_res.bse.get("T", np.nan)),
            "p": float(sub_res.pvalues.get("T", np.nan)),
            "N": int(sub_res.nobs),
            "families": int(sub["MotherID"].nunique()),
        })

    sub_table = pd.DataFrame(rows).sort_values("subgroup")
    print("\nSubgroup IPW-FE estimates (same weights/spec as overall):")
    print(sub_table.to_string(index=False))

    out_sub = f"ipw_{args.contrast}_{args.outcome}_subgroups.csv"
    sub_table.to_csv(out_sub, index=False)
    print(f"Saved: {out_sub}")

    # Save outputs
    out_prefix = f"ipw_{args.contrast}_{args.outcome}"
    bal.to_csv(f"{out_prefix}_balance.csv", index=False)
    tmp_ps[["MotherID", "T", "ps", "w", args.outcome] + optional_controls].to_csv(f"{out_prefix}_analysis_data.csv", index=False)

    # A small, clean one-line summary:
    coef = res.params.get("T", np.nan)
    se = res.bse.get("T", np.nan)
    pval = res.pvalues.get("T", np.nan)
    print(f"\nIPW-FE estimate for {args.outcome} ({args.contrast}): "
          f"beta={coef:.4f}, se={se:.4f}, p={pval:.4g}, N={int(res.nobs)}")
    print(f"Saved: {out_prefix}_balance.csv, {out_prefix}_analysis_data.csv")


if __name__ == "__main__":
    main()
