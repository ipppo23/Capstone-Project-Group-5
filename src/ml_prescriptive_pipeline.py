import re
import textwrap
from pathlib import Path
import numpy as np
import pandas as pd
import argparse

parser = argparse.ArgumentParser(
    description="Run the Capstone ML + Prescriptive pipeline to produce TASK_*.csv outputs."
)

THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parent.parent if THIS_FILE.parent.name.lower() == "src" else THIS_FILE.parent

default_raw = PROJECT_ROOT / "Raw Data" / "RQ1"
default_out = PROJECT_ROOT / "outputs_capstone"

parser.add_argument(
    "--project_root",
    type=str,
    default=str(PROJECT_ROOT),
    help="Project root folder (the folder that contains Raw Data/, outputs_capstone/, and src/)",
)
parser.add_argument(
    "--raw_dir",
    type=str,
    default=str(default_raw),
    help="Folder containing your raw StatCan CSVs (e.g., .../Raw Data/RQ1)",
)
parser.add_argument(
    "--out_dir",
    type=str,
    default=str(default_out),
    help="Output folder where /stage and /tasks will be created",
)

args = parser.parse_args()

PROJECT_ROOT = Path(args.project_root).expanduser().resolve()
RAW_DIR = Path(args.raw_dir).expanduser().resolve()
OUT_DIR = Path(args.out_dir).expanduser().resolve()
STAGE_DIR = OUT_DIR / "stage"
TASK_DIR = OUT_DIR / "tasks"

for d in [OUT_DIR, STAGE_DIR, TASK_DIR]:
    d.mkdir(parents=True, exist_ok=True)

print(f"PROJECT_ROOT = {PROJECT_ROOT}")
print(f"RAW_DIR      = {RAW_DIR}")
print(f"OUT_DIR      = {OUT_DIR}")
print(f"TASK_DIR     = {TASK_DIR}")

INDUSTRY_OPP_WEIGHTS = {"demand": 0.50, "vacancies": 0.30, "wage": 0.20}
OCCUPATION_OPP_WEIGHTS = {"demand": 0.45, "wage": 0.30, "vacancies": 0.25}
PROVINCE_SCORE_WEIGHTS = {"industry": 0.35, "occupation": 0.25, "education": 0.20, "supply": 0.20}

def candidate_search_dirs(raw_dir: Path, project_root: Path):
    dirs = []
    for p in [
        raw_dir,
        project_root / "Raw Data" / "RQ1",
        project_root / "Raw Data",
        project_root,
    ]:
        if p not in dirs:
            dirs.append(p)
    return dirs

def pick_file(search_dirs, candidates):
    for base in search_dirs:
        if not base.exists():
            continue
        for c in candidates:
            p = base / c
            if p.exists():
                return p

    for c in candidates:
        m = re.search(r"(\d{10})", c)
        if not m:
            continue
        pid = m.group(1)
        for base in search_dirs:
            if not base.exists():
                continue
            hits = sorted(base.rglob(f"*{pid}*.csv"))
            if hits:
                return hits[0]
    return None

SEARCH_DIRS = candidate_search_dirs(RAW_DIR, PROJECT_ROOT)
print("Search directories:")
for d in SEARCH_DIRS:
    print(f"  - {d}")

FILES = {
    "edu_lfs":        pick_file(SEARCH_DIRS, ["1410002001.csv", "1410002001_databaseLoadingData.csv"]),
    "monthly_vac":    pick_file(SEARCH_DIRS, ["1410037101.csv", "1410037101_databaseLoadingData.csv"]),
    "jvws_ind":       pick_file(SEARCH_DIRS, ["1410044201.csv", "1410044201_databaseLoadingData.csv"]),
    "jvws_occ_unit":  pick_file(SEARCH_DIRS, ["1410044401.csv", "1410044401_databaseLoadingData.csv"]),
    "jvws_occ_broad": pick_file(SEARCH_DIRS, ["1410044301.csv", "1410044301_databaseLoadingData.csv"]),
    "supply_proxy":   pick_file(SEARCH_DIRS, ["3710019601.csv", "3710019601_databaseLoadingData.csv", "3710019601_databaseLoadingData (1).csv"]),
}

required = ["edu_lfs", "jvws_ind", "supply_proxy"]
missing_required = [k for k in required if FILES.get(k) is None]
if missing_required:
    snapshot = []
    for d in SEARCH_DIRS:
        if d.exists():
            try:
                snapshot.append(f"\n[{d}]\n" + "\n".join(sorted(p.name for p in d.glob("*.csv"))[:50]))
            except Exception:
                pass
    raise FileNotFoundError(
        "Missing required files:\n"
        + "\n".join([f"- {k}" for k in missing_required])
        + "\n\nSearched in:\n"
        + "\n".join([f"- {d}" for d in SEARCH_DIRS])
        + ("\n\nCSV snapshot:" + "".join(snapshot) if snapshot else "")
    )

if FILES["jvws_occ_unit"] is None and FILES["jvws_occ_broad"] is None:
    raise FileNotFoundError("Missing occupation file: need 1410044401 OR 1410044301.")

print("Files detected:")
for k, v in FILES.items():
    print(f"  - {k}: {str(v) if v is not None else '(not found)'}")

CAN_GEOS = [
    "Canada",
    "Newfoundland and Labrador",
    "Prince Edward Island",
    "Nova Scotia",
    "New Brunswick",
    "Quebec",
    "Ontario",
    "Manitoba",
    "Saskatchewan",
    "Alberta",
    "British Columbia",
    "Yukon",
    "Northwest Territories",
    "Nunavut",
]

def year_from_ref(x):
    if pd.isna(x): return np.nan
    s = str(x)
    m = re.search(r"(\d{4})", s)
    return int(m.group(1)) if m else np.nan

def refkey(x):
    if pd.isna(x): return np.nan
    s = str(x)
    m = re.match(r"(\d{4})(?:-(\d{2}))?", s)
    if not m: return np.nan
    y = int(m.group(1))
    mo = int(m.group(2)) if m.group(2) else 1
    return y * 100 + mo

def find_col(df, tokens):
    tokens = tokens if isinstance(tokens, (list, tuple)) else [tokens]
    for c in df.columns:
        cl = c.lower()
        if any(t.lower() in cl for t in tokens):
            return c
    return None

def norm01(s: pd.Series):
    s = pd.to_numeric(s, errors="coerce")
    mn, mx = s.min(skipna=True), s.max(skipna=True)
    if pd.isna(mn) or pd.isna(mx) or mx == mn:
        return pd.Series(np.zeros(len(s)), index=s.index)
    return (s - mn) / (mx - mn)

def weighted_score_from_norms(df, weighted_cols):
    out = []
    for _, row in df.iterrows():
        numerator = 0.0
        denom = 0.0
        for col, w in weighted_cols:
            val = pd.to_numeric(row.get(col), errors="coerce")
            if pd.notna(val):
                numerator += float(w) * float(val)
                denom += float(w)
        out.append((numerator / denom) if denom > 0 else np.nan)
    return pd.Series(out, index=df.index, dtype=float)

def filter_geo(df):
    if df is None or df.empty:
        return df
    if "GEO" not in df.columns:
        return df
    return df[df["GEO"].isin(CAN_GEOS)].copy()

def audit_df(name, df):
    df = df.copy()
    yr_rng = (None, None)
    if "REF_DATE" in df.columns:
        yrs = df["REF_DATE"].apply(year_from_ref).dropna().astype(int)
        if len(yrs):
            yr_rng = (int(yrs.min()), int(yrs.max()))
    geo_n = df["GEO"].nunique() if "GEO" in df.columns else None
    print(f"\n=== {name} ===")
    print("shape:", df.shape)
    if yr_rng[0] is not None:
        print("year range:", yr_rng[0], "→", yr_rng[1], "| max_year =", yr_rng[1])
    print("unique GEO:", geo_n)
    if "GEO" in df.columns:
        print("top GEO values:\n", df["GEO"].value_counts().head(5).to_string())
    return df

def pivot_stats(df, idx_cols, stat_col="Statistics", val_col="VALUE"):
    out = df.copy()
    out[val_col] = pd.to_numeric(out[val_col], errors="coerce")
    wide = out.pivot_table(index=idx_cols, columns=stat_col, values=val_col, aggfunc="mean").reset_index()
    wide.columns = [str(c) for c in wide.columns]
    return wide

def annual_features(wide, group_cols, year_col="YEAR", ref_col="REFKEY", prefix=""):
    """
    Build annual features from monthly/quarterly:
      - metric_mean, metric_std across periods within year
      - metric_growth = last - first (within year)
    Standard outputs:
      demand_mean/std/growth, vacancies_mean/std/growth, wage_mean/std/growth
    """
    df = wide.copy()
    metric_cols = [c for c in df.columns if c not in (group_cols + [year_col, ref_col])]

    agg = df.groupby(group_cols + [year_col], as_index=False)[metric_cols].agg(["mean", "std"])
    agg.columns = ["_".join([x for x in col if x]) for col in agg.columns.to_flat_index()]
    for g in group_cols + [year_col]:
        cand = [c for c in agg.columns if c.startswith(g + "_")]
        if cand:
            agg.rename(columns={cand[0]: g}, inplace=True)

    df_sorted = df.sort_values(group_cols + [year_col, ref_col])
    first = df_sorted.groupby(group_cols + [year_col], as_index=False)[metric_cols].first()
    last  = df_sorted.groupby(group_cols + [year_col], as_index=False)[metric_cols].last()
    growth = first[group_cols + [year_col]].copy()
    for m in metric_cols:
        growth[f"{m}_growth"] = pd.to_numeric(last[m], errors="coerce") - pd.to_numeric(first[m], errors="coerce")

    out = agg.merge(growth, on=group_cols + [year_col], how="left")

    def pick_mean(token):
        for c in out.columns:
            if token in c.lower() and c.lower().endswith("_mean"):
                return c
        return None

    def pick_std(token):
        for c in out.columns:
            if token in c.lower() and c.lower().endswith("_std"):
                return c
        return None

    def pick_growth(token):
        for c in out.columns:
            if token in c.lower() and c.lower().endswith("_growth"):
                return c
        return None

    vacrate_mean = pick_mean("job vacancy rate")
    vac_mean     = pick_mean("job vacancies")
    wage_mean    = pick_mean("average offered hourly wage")

    vacrate_std  = pick_std("job vacancy rate")
    vac_std      = pick_std("job vacancies")
    wage_std     = pick_std("average offered hourly wage")

    vacrate_g    = pick_growth("job vacancy rate")
    vac_g        = pick_growth("job vacancies")
    wage_g       = pick_growth("average offered hourly wage")

    out[f"{prefix}demand_mean"]   = out[vacrate_mean] if vacrate_mean else out[vac_mean]
    out[f"{prefix}demand_std"]    = out[vacrate_std]  if vacrate_std  else out[vac_std]
    out[f"{prefix}demand_growth"] = out[vacrate_g]    if vacrate_g    else out[vac_g]

    out[f"{prefix}vacancies_mean"]   = out[vac_mean]
    out[f"{prefix}vacancies_std"]    = out[vac_std]
    out[f"{prefix}vacancies_growth"] = out[vac_g]

    out[f"{prefix}wage_mean"]   = out[wage_mean] if wage_mean else np.nan
    out[f"{prefix}wage_std"]    = out[wage_std]  if wage_std  else np.nan
    out[f"{prefix}wage_growth"] = out[wage_g]    if wage_g    else np.nan

    return out

def safe_mean_topn(df, score_col, n=3):
    if df is None or df.empty or score_col not in df.columns:
        return np.nan
    x = df[score_col].dropna().sort_values(ascending=False).head(n)
    return float(x.mean()) if len(x) else np.nan

def max_year_df(df):
    return int(np.nanmax(df["REF_DATE"].apply(year_from_ref)))

def extract_bracket_code(label: str):
    m = re.search(r"\[(.*?)\]\s*$", str(label))
    return m.group(1).strip() if m else None

def is_total_all_occupations(label: str) -> bool:
    s = str(label).strip().lower()
    return s == "total, all occupations"

def clean_wrap(s: str, width=86):
    return "\n".join(textwrap.wrap(str(s), width=width, break_long_words=False))

def short_label(s: str, width=90):
    return textwrap.shorten(str(s), width=width, placeholder="…")

def backtest_ridge_vs_naive(df_feat, group_cols, feature_cols, target_cols, feature_year,
                            name="", strict_next_year=True, min_train_rows=200):
    """
    Backtest: input YEAR=feature_year-1 -> predict target in feature_year
    Compare Ridge vs Naive (y_next_hat = y_now).
    Returns a DataFrame with metrics and ridge_wins flag per target.
    """
    try:
        from sklearn.linear_model import Ridge
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    except Exception as e:
        print(f"[{name}] sklearn unavailable ({e}). Skipping backtest.")
        return pd.DataFrame()

    df = df_feat.copy()

    for c in set(feature_cols + target_cols):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.sort_values(group_cols + ["YEAR"])
    year_next = df.groupby(group_cols)["YEAR"].shift(-1)

    out_rows = []
    for t in target_cols:
        y_next = df.groupby(group_cols)[t].shift(-1)
        if strict_next_year:
            y_next = y_next.where(year_next == (df["YEAR"] + 1))
        df[f"{t}_next"] = y_next

        rows = df[df["YEAR"] <= feature_year - 1].dropna(subset=[f"{t}_next"]).copy()
        test = rows[rows["YEAR"] == feature_year - 1].copy()
        train = rows[rows["YEAR"] <= feature_year - 2].copy()

        test = test.dropna(subset=[t])
        train = train.dropna(subset=[t])

        if len(train) < min_train_rows or len(test) < 10:
            out_rows.append({
                "dataset": name, "target": t, "ridge_wins": False,
                "ridge_MAE": np.nan, "naive_MAE": np.nan,
                "ridge_RMSE": np.nan, "naive_RMSE": np.nan,
                "ridge_R2": np.nan, "naive_R2": np.nan,
                "n_train": int(len(train)), "n_test": int(len(test)),
                "note": "too_few_rows"
            })
            continue

        X_cols = [c for c in feature_cols if c in df.columns and not train[c].isna().all()]
        if not X_cols:
            out_rows.append({
                "dataset": name, "target": t, "ridge_wins": False,
                "ridge_MAE": np.nan, "naive_MAE": np.nan,
                "ridge_RMSE": np.nan, "naive_RMSE": np.nan,
                "ridge_R2": np.nan, "naive_R2": np.nan,
                "n_train": int(len(train)), "n_test": int(len(test)),
                "note": "no_features"
            })
            continue

        med = train[X_cols].median(numeric_only=True).fillna(0.0)

        def prep(d):
            X = d[X_cols].copy()
            for c in X_cols:
                X[c] = pd.to_numeric(X[c], errors="coerce")
            return X.fillna(med).fillna(0.0)

        ytr = pd.to_numeric(train[f"{t}_next"], errors="coerce").values
        yte = pd.to_numeric(test[f"{t}_next"], errors="coerce").values

        model = Ridge(alpha=1.0)
        model.fit(prep(train), ytr)
        pred_r = model.predict(prep(test))

        pred_n = pd.to_numeric(test[t], errors="coerce").values

        ridge_mae = float(mean_absolute_error(yte, pred_r))
        naive_mae = float(mean_absolute_error(yte, pred_n))
        ridge_rmse = float(np.sqrt(mean_squared_error(yte, pred_r)))
        naive_rmse = float(np.sqrt(mean_squared_error(yte, pred_n)))
        ridge_r2 = float(r2_score(yte, pred_r))
        naive_r2 = float(r2_score(yte, pred_n))

        out_rows.append({
            "dataset": name, "target": t,
            "ridge_wins": bool(ridge_mae < naive_mae),
            "ridge_MAE": ridge_mae, "naive_MAE": naive_mae,
            "ridge_RMSE": ridge_rmse, "naive_RMSE": naive_rmse,
            "ridge_R2": ridge_r2, "naive_R2": naive_r2,
            "n_train": int(len(train)), "n_test": int(len(test)),
            "note": ""
        })

    return pd.DataFrame(out_rows)

def forecast_next_with_selection(df_feat, group_cols, feature_cols, target_cols,
                                 feature_year, pred_year, eval_table=None,
                                 name="", strict_next_year=True,
                                 min_train_rows=200):
    """
    Forecast pred_year using data at feature_year:
      - trains Ridge on (<=feature_year-2) to predict next year
      - uses Ridge only if eval_table says ridge_wins=True for that target
      - otherwise uses Naive (y_pred = y_now)
    Returns dataframe with *_pred and *_pred_model for each target.
    """
    try:
        from sklearn.linear_model import Ridge
    except Exception as e:
        print(f"[{name}] sklearn unavailable ({e}). Falling back to naive.")
        eval_table = None

    df = df_feat.copy()
    for c in set(feature_cols + target_cols):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.sort_values(group_cols + ["YEAR"])
    year_next = df.groupby(group_cols)["YEAR"].shift(-1)

    for t in target_cols:
        y_next = df.groupby(group_cols)[t].shift(-1)
        if strict_next_year:
            y_next = y_next.where(year_next == (df["YEAR"] + 1))
        df[f"{t}_next"] = y_next

    rows_all = df[df["YEAR"] <= feature_year - 1].copy()
    pred_input = df[df["YEAR"] == feature_year].copy()
    if pred_input.empty:
        print(f"\n[{name}] WARNING: No rows for FEATURE_YEAR={feature_year}. Using last-available-year naive.")
        pred_input = df.groupby(group_cols).tail(1).copy()
        pred_input["YEAR"] = feature_year

    out = pred_input[group_cols].copy()
    out["FEATURE_YEAR"] = feature_year
    out["PRED_YEAR"] = pred_year

    eval_map = {}
    if eval_table is not None and not eval_table.empty:
        for _, r in eval_table.iterrows():
            eval_map[str(r["target"])] = bool(r.get("ridge_wins", False))

    X_cols_base = [c for c in feature_cols if c in df.columns]
    X_cols_base = [c for c in X_cols_base if not rows_all[c].isna().all()]

    def prep_X(d, X_cols, med):
        X = d[X_cols].copy()
        for c in X_cols:
            X[c] = pd.to_numeric(X[c], errors="coerce")
        return X.fillna(med).fillna(0.0)

    for t in target_cols:
        rows = rows_all.dropna(subset=[f"{t}_next"]).copy()
        test = rows[rows["YEAR"] == feature_year - 1].dropna(subset=[t]).copy()
        train = rows[rows["YEAR"] <= feature_year - 2].dropna(subset=[t]).copy()

        def use_naive():
            out[f"{t}_pred"] = pd.to_numeric(pred_input[t], errors="coerce").values
            out[f"{t}_pred_model"] = "naive"

        ridge_should = eval_map.get(str(t), False) if eval_map else False

        if len(train) < min_train_rows or not X_cols_base:
            use_naive()
            continue

        if not ridge_should:
            use_naive()
            continue

        X_cols = [c for c in X_cols_base if not train[c].isna().all()]
        if not X_cols:
            use_naive()
            continue

        med = train[X_cols].median(numeric_only=True).fillna(0.0)

        try:
            model = Ridge(alpha=1.0)
            model.fit(prep_X(train, X_cols, med), pd.to_numeric(train[f"{t}_next"], errors="coerce").values)
            out[f"{t}_pred"] = model.predict(prep_X(pred_input, X_cols, med))
            out[f"{t}_pred_model"] = "ridge"
        except Exception:
            use_naive()

    return out

def backtest_confusion_and_metrics(df_feat, group_cols, feature_cols, target,
                                   feature_year, name="", strict_next_year=True,
                                   mode="direction", quantile=0.67):
    """
    Confusion matrix evaluation for a regression forecast by binarizing the outcome.

    mode:
      - "direction": class=1 if y_next > y_now
      - "high_vs_low": class=1 if y_next >= quantile(y_next)

    Robust fixes:
      - drops rows where y_now / y_next / y_score are NaN
      - confusion_matrix uses labels=[0,1] (always 2x2)
      - ROC-AUC computed only when both classes exist and scores finite
    """
    try:
        from sklearn.linear_model import Ridge
        from sklearn.metrics import (
            confusion_matrix, accuracy_score, precision_score, recall_score,
            f1_score, roc_auc_score
        )
    except Exception:
        return None

    df = df_feat.copy()
    for c in set(feature_cols + [target]):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.sort_values(group_cols + ["YEAR"])
    year_next = df.groupby(group_cols)["YEAR"].shift(-1)

    y_next = df.groupby(group_cols)[target].shift(-1)
    if strict_next_year:
        y_next = y_next.where(year_next == (df["YEAR"] + 1))
    df["y_next"] = y_next

    rows = df[df["YEAR"] <= feature_year - 1].dropna(subset=["y_next"]).copy()
    test = rows[rows["YEAR"] == feature_year - 1].copy()
    train = rows[rows["YEAR"] <= feature_year - 2].copy()

    test = test.dropna(subset=[target])
    train = train.dropna(subset=[target])

    if test.empty or train.empty:
        return None

    X_cols = [c for c in feature_cols if c in df.columns and not train[c].isna().all()]
    if not X_cols:
        return None

    med = train[X_cols].median(numeric_only=True).fillna(0.0)

    def prep(d):
        X = d[X_cols].copy()
        for c in X_cols:
            X[c] = pd.to_numeric(X[c], errors="coerce")
        return X.fillna(med).fillna(0.0)

    m = Ridge(alpha=1.0)
    m.fit(prep(train), pd.to_numeric(train["y_next"], errors="coerce").values)

    y_now = pd.to_numeric(test[target], errors="coerce").values
    y_true_next = pd.to_numeric(test["y_next"], errors="coerce").values
    y_score = m.predict(prep(test))
    y_score_naive = y_now.copy()

    mask = np.isfinite(y_now) & np.isfinite(y_true_next) & np.isfinite(y_score) & np.isfinite(y_score_naive)
    y_now = y_now[mask]
    y_true_next = y_true_next[mask]
    y_score = y_score[mask]
    y_score_naive = y_score_naive[mask]

    if len(y_now) == 0:
        return None

    if mode == "direction":
        y_true = (y_true_next > y_now).astype(int)
        y_pred = (y_score > y_now).astype(int)
        y_pred_naive = (y_score_naive > y_now).astype(int)
        threshold_used = "per-row(y_now)"
    elif mode == "high_vs_low":
        thr = np.nanquantile(y_true_next, quantile)
        y_true = (y_true_next >= thr).astype(int)
        y_pred = (y_score >= thr).astype(int)
        y_pred_naive = (y_score_naive >= thr).astype(int)
        threshold_used = float(thr)
    else:
        raise ValueError("mode must be 'direction' or 'high_vs_low'")

    from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

    def pack(tag, yt, yp, yscore):
        cm = confusion_matrix(yt, yp, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()

        acc = accuracy_score(yt, yp)
        prec = precision_score(yt, yp, zero_division=0)
        rec = recall_score(yt, yp, zero_division=0)
        f1 = f1_score(yt, yp, zero_division=0)

        auc = np.nan
        if len(np.unique(yt)) == 2 and np.isfinite(yscore).all():
            auc = roc_auc_score(yt, yscore)

        return {
            "dataset": name,
            "target": target,
            "mode": mode,
            "model": tag,
            "threshold_used": threshold_used,
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
            "accuracy": float(acc),
            "precision": float(prec),
            "recall": float(rec),
            "f1": float(f1),
            "roc_auc": float(auc) if auc == auc else np.nan,
            "n_test": int(len(yt)),
            "backtest_input_year": int(feature_year - 1),
            "backtest_target_year": int(feature_year),
        }

    ridge_res = pack("ridge", y_true, y_pred, y_score)
    naive_res = pack("naive", y_true, y_pred_naive, y_score_naive)

    print(f"\n[{name}] CONFUSION MATRIX ({mode}) — target={target} — backtest {feature_year-1}->{feature_year}")
    print("Ridge [[TN FP],[FN TP]] =", [[ridge_res["tn"], ridge_res["fp"]],[ridge_res["fn"], ridge_res["tp"]]])
    print("Naive [[TN FP],[FN TP]] =", [[naive_res["tn"], naive_res["fp"]],[naive_res["fn"], naive_res["tp"]]])

    return ridge_res, naive_res

def qualitative_sanity_ranges(pred_df, observed_df, pred_col, obs_col):
    """
    % of predictions below observed min and above observed max (over observed window).
    """
    obs = pd.to_numeric(observed_df[obs_col], errors="coerce")
    mn, mx = obs.min(skipna=True), obs.max(skipna=True)

    p = pd.to_numeric(pred_df[pred_col], errors="coerce")
    p = p.dropna()
    if len(p) == 0 or pd.isna(mn) or pd.isna(mx):
        return np.nan, np.nan

    return float((p < mn).mean()), float((p > mx).mean())

def weight_sensitivity(prov_rank_components, base_weights, n_sims=500, noise=0.05, seed=0):
    """
    Randomly perturb weights around base weights and compute stability:
      - avg_rank
      - p_top1
      - p_top3
    """
    rng = np.random.default_rng(seed)
    geos = prov_rank_components["GEO"].tolist()
    comp_cols = list(base_weights.keys())

    df = prov_rank_components.copy()
    for c in comp_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df[comp_cols] = df[comp_cols].fillna(df[comp_cols].median(numeric_only=True)).fillna(0.0)

    rank_accum = {g: [] for g in geos}
    top1_accum = {g: 0 for g in geos}
    top3_accum = {g: 0 for g in geos}

    w0 = np.array([base_weights[c] for c in comp_cols], dtype=float)

    for _ in range(n_sims):
        w = w0 + rng.normal(0, noise, size=len(w0))
        w = np.clip(w, 0.001, None)
        w = w / w.sum()

        score = np.zeros(len(df))
        for i, c in enumerate(comp_cols):
            score += w[i] * df[c].to_numpy()

        ranks = pd.Series(score, index=df["GEO"]).rank(ascending=False, method="min")
        for g in geos:
            r = int(ranks.loc[g])
            rank_accum[g].append(r)
        top1 = ranks.idxmin()
        top1_accum[top1] += 1
        top3 = ranks.nsmallest(3).index.tolist()
        for g in top3:
            top3_accum[g] += 1

    out = []
    for g in geos:
        out.append({
            "GEO": g,
            "avg_rank": float(np.mean(rank_accum[g])),
            "p_top1": float(top1_accum[g] / n_sims),
            "p_top3": float(top3_accum[g] / n_sims),
        })
    return pd.DataFrame(out).sort_values("avg_rank", ascending=True)

edu = pd.read_csv(FILES["edu_lfs"], low_memory=False)
ind = pd.read_csv(FILES["jvws_ind"], low_memory=False)
supply = pd.read_csv(FILES["supply_proxy"], low_memory=False)
monthly = pd.read_csv(FILES["monthly_vac"], low_memory=False) if FILES["monthly_vac"] else None

occ_score_path = FILES["jvws_occ_broad"] if FILES["jvws_occ_broad"] else FILES["jvws_occ_unit"]
occ_score = pd.read_csv(occ_score_path, low_memory=False)

audit_df("LFS Education Outcomes (1410002001)", edu)
audit_df("JVWS Industry (1410044201)", ind)
audit_df("Supply Proxy (3710019601)", supply)
if monthly is not None:
    audit_df("Monthly Vacancies (1410037101)", monthly)
audit_df(f"JVWS Occupation SCORING ({occ_score_path.name})", occ_score)

edu_f = filter_geo(edu)
ind_f = filter_geo(ind)
supply_f = filter_geo(supply)
occ_score_f = filter_geo(occ_score)
monthly_f = filter_geo(monthly) if monthly is not None else None

edu_f.to_csv(STAGE_DIR / "edu_lfs_filtered.csv", index=False)
ind_f.to_csv(STAGE_DIR / "jvws_industry_filtered.csv", index=False)
supply_f.to_csv(STAGE_DIR / "supply_proxy_filtered.csv", index=False)
occ_score_f.to_csv(STAGE_DIR / "jvws_occ_scoring_filtered.csv", index=False)
if monthly_f is not None:
    monthly_f.to_csv(STAGE_DIR / "monthly_vac_filtered.csv", index=False)

print("\nStaged files in:", STAGE_DIR)

edu = pd.read_csv(STAGE_DIR / "edu_lfs_filtered.csv", low_memory=False)
ind = pd.read_csv(STAGE_DIR / "jvws_industry_filtered.csv", low_memory=False)
supply = pd.read_csv(STAGE_DIR / "supply_proxy_filtered.csv", low_memory=False)
occ = pd.read_csv(STAGE_DIR / "jvws_occ_scoring_filtered.csv", low_memory=False)

print("\nScoring occupation table:", "1410044301 (broad)" if "1410044301" in occ_score_path.name else ("1410044401 (unit)" if "1410044401" in occ_score_path.name else occ_score_path.name))

geo_sets = [set(df["GEO"].dropna().unique()) for df in [edu, ind, supply, occ]]
COMMON_GEO = sorted(set.intersection(*geo_sets))
COMMON_PROVINCES = [g for g in COMMON_GEO if g != "Canada"]
print("\nCOMMON_PROVINCES used for ranking:", COMMON_PROVINCES)

maxy = min(max_year_df(ind), max_year_df(edu), max_year_df(supply), max_year_df(occ))
FEATURE_YEAR = maxy
PRED_YEAR = FEATURE_YEAR + 1
print(f"\nFEATURE_YEAR={FEATURE_YEAR}  ->  PRED_YEAR={PRED_YEAR}")

ind = ind[ind["GEO"].isin(COMMON_GEO)].copy()
ind["YEAR"] = ind["REF_DATE"].apply(year_from_ref)
ind["REFKEY"] = ind["REF_DATE"].apply(refkey)

naics_col = find_col(ind, ["naics", "north american industry"])
if naics_col is None:
    raise ValueError("Could not detect NAICS column in industry file.")

ind_wide = pivot_stats(ind, idx_cols=["GEO", naics_col, "YEAR", "REFKEY"], stat_col="Statistics", val_col="VALUE")
ind_feat = annual_features(ind_wide, group_cols=["GEO", naics_col], year_col="YEAR", ref_col="REFKEY")
ind_feat = ind_feat[ind_feat["GEO"].isin(COMMON_GEO)].copy()

occ = occ[occ["GEO"].isin(COMMON_GEO)].copy()
occ["YEAR"] = occ["REF_DATE"].apply(year_from_ref)
occ["REFKEY"] = occ["REF_DATE"].apply(refkey)

occ_col = find_col(occ, ["national occupational classification", "occupational", "noc"])
if occ_col is None:
    raise ValueError("Could not detect occupation column in occupation file.")

if "Job vacancy characteristics" in occ.columns:
    keep_char = "Type of work, all types"
    if keep_char in occ["Job vacancy characteristics"].unique():
        occ = occ[occ["Job vacancy characteristics"] == keep_char].copy()

occ_wide = pivot_stats(occ, idx_cols=["GEO", occ_col, "YEAR", "REFKEY"], stat_col="Statistics", val_col="VALUE")
occ_feat = annual_features(occ_wide, group_cols=["GEO", occ_col], year_col="YEAR", ref_col="REFKEY")
occ_feat = occ_feat[occ_feat["GEO"].isin(COMMON_GEO)].copy()

den = occ_feat.groupby(["GEO", "YEAR"])["vacancies_mean"].transform("sum").replace(0, np.nan)
occ_feat["demand_mean"] = occ_feat["vacancies_mean"] / den
occ_feat = occ_feat.sort_values(["GEO", occ_col, "YEAR"])
occ_feat["demand_growth"] = occ_feat.groupby(["GEO", occ_col])["demand_mean"].diff().fillna(0.0)
occ_feat["demand_std"] = np.nan

chk = occ_feat.groupby(["GEO", "YEAR"])["demand_mean"].sum().dropna()
print("Occupation demand-share sum (should be ~1):", float(chk.median()) if len(chk) else None)

edu = edu[edu["GEO"].isin(COMMON_GEO)].copy()
edu["YEAR"] = edu["REF_DATE"].apply(year_from_ref)

for col, val in [("Gender", "Total - Gender"), ("Age group", "15 years and over")]:
    if col in edu.columns:
        edu = edu[edu[col] == val].copy()

edu_wide = (edu
    .pivot_table(index=["GEO", "Educational attainment", "YEAR"],
                 columns="Labour force characteristics",
                 values="VALUE",
                 aggfunc="mean")
    .reset_index())
edu_wide.columns = [str(c) for c in edu_wide.columns]

if "Employment rate" not in edu_wide.columns:
    raise ValueError("Education table missing 'Employment rate'. Re-export 1410002001 including Employment rate.")

edu_wide["employment_rate"] = pd.to_numeric(edu_wide["Employment rate"], errors="coerce")
edu_wide["unemployment_rate"] = pd.to_numeric(edu_wide["Unemployment rate"], errors="coerce") if "Unemployment rate" in edu_wide.columns else np.nan

supply = supply[supply["GEO"].isin(COMMON_GEO)].copy()
supply["YEAR"] = supply["REF_DATE"].apply(year_from_ref)

neet = supply[
    (supply.get("Age group", "") == "Total, 15 to 29 years") &
    (supply.get("Gender", "") == "Total - Gender") &
    (supply.get("Educational attainment level", "") == "Total, all education levels") &
    (supply.get("Labour force and education status", "") == "Sub-total, not in employment, education or training (NEET)")
].copy()
neet = neet.groupby(["GEO", "YEAR"], as_index=False)["VALUE"].mean().rename(columns={"VALUE": "neet_prop"})

ind_targets = ["demand_mean", "vacancies_mean", "wage_mean"]
ind_features = ["demand_mean", "demand_std", "demand_growth",
                "vacancies_mean", "vacancies_std", "vacancies_growth",
                "wage_mean", "wage_std", "wage_growth"]

occ_targets = ["vacancies_mean", "wage_mean"]
occ_features = ["demand_mean", "demand_growth",
                "vacancies_mean", "vacancies_std", "vacancies_growth",
                "wage_mean", "wage_std", "wage_growth"]

edu_targets = ["employment_rate", "unemployment_rate"]
edu_features = ["employment_rate", "unemployment_rate"]

neet_targets = ["neet_prop"]
neet_features = ["neet_prop"]

EVAL_ind = backtest_ridge_vs_naive(ind_feat, ["GEO", naics_col], ind_features, ind_targets, FEATURE_YEAR, name="Industry", min_train_rows=400)
EVAL_occ = backtest_ridge_vs_naive(occ_feat, ["GEO", occ_col], occ_features, occ_targets, FEATURE_YEAR, name="Occupation", min_train_rows=200)

if not EVAL_ind.empty:
    print(f"\n[Industry] Backtest (input {FEATURE_YEAR-1} -> target {FEATURE_YEAR})")
    print(EVAL_ind[["target","ridge_wins","ridge_MAE","naive_MAE","ridge_RMSE","naive_RMSE","ridge_R2","naive_R2","n_train","n_test"]].to_string(index=False))

if not EVAL_occ.empty:
    print(f"\n[Occupation] Backtest (input {FEATURE_YEAR-1} -> target {FEATURE_YEAR})")
    print(EVAL_occ[["target","ridge_wins","ridge_MAE","naive_MAE","ridge_RMSE","naive_RMSE","ridge_R2","naive_R2","n_train","n_test"]].to_string(index=False))


(EVAL_ind.drop(columns=["dataset"], errors="ignore")
 .to_csv(TASK_DIR / "EVAL_backtest_regression_industry.csv", index=False))
(EVAL_occ.drop(columns=["dataset"], errors="ignore")
 .to_csv(TASK_DIR / "EVAL_backtest_regression_occupation.csv", index=False))
print(f"\nSaved: {TASK_DIR / 'EVAL_backtest_regression_industry.csv'}")
print(f"Saved: {TASK_DIR / 'EVAL_backtest_regression_occupation.csv'}")

ind_pred = forecast_next_with_selection(ind_feat, ["GEO", naics_col], ind_features, ind_targets,
                                        FEATURE_YEAR, PRED_YEAR, eval_table=EVAL_ind, name="Industry", min_train_rows=400)

occ_pred = forecast_next_with_selection(occ_feat, ["GEO", occ_col], occ_features, occ_targets,
                                        FEATURE_YEAR, PRED_YEAR, eval_table=EVAL_occ, name="Occupation", min_train_rows=200)

edu_pred = forecast_next_with_selection(edu_wide, ["GEO", "Educational attainment"], edu_features, edu_targets,
                                        FEATURE_YEAR, PRED_YEAR, eval_table=pd.DataFrame(), name="Education", min_train_rows=50)

neet_pred = forecast_next_with_selection(neet, ["GEO"], neet_features, neet_targets,
                                         FEATURE_YEAR, PRED_YEAR, eval_table=pd.DataFrame(), name="Supply(NEET)", min_train_rows=20)

ind_pred_out = ind_pred[ind_pred["GEO"].isin(COMMON_PROVINCES)].copy()
occ_pred_out = occ_pred[occ_pred["GEO"].isin(COMMON_PROVINCES)].copy()
edu_pred_out = edu_pred[edu_pred["GEO"].isin(COMMON_PROVINCES)].copy()
neet_pred_out = neet_pred[neet_pred["GEO"].isin(COMMON_PROVINCES)].copy()

denp = occ_pred_out.groupby(["GEO"])["vacancies_mean_pred"].transform("sum").replace(0, np.nan)
occ_pred_out["demand_mean_pred"] = occ_pred_out["vacancies_mean_pred"] / denp

ind_pred_out.to_csv(TASK_DIR / "TASK_B_industry_predictions.csv", index=False)
occ_pred_out.to_csv(TASK_DIR / "TASK_B2_occupation_predictions.csv", index=False)
edu_pred_out.to_csv(TASK_DIR / "TASK_C_education_predictions.csv", index=False)
neet_pred_out.to_csv(TASK_DIR / "TASK_C2_supply_neet_predictions.csv", index=False)

print("\n Wrote TASK_B/TASK_B2/TASK_C (+NEET) to:", TASK_DIR)

def build_opp_industry(df):
    d = df.copy()
    d["d_norm"] = norm01(d["demand_mean_pred"])
    d["v_norm"] = norm01(d["vacancies_mean_pred"])
    d["w_norm"] = norm01(d["wage_mean_pred"]) if "wage_mean_pred" in d.columns else np.nan
    d["opp_score"] = weighted_score_from_norms(
        d,
        [
            ("d_norm", INDUSTRY_OPP_WEIGHTS["demand"]),
            ("v_norm", INDUSTRY_OPP_WEIGHTS["vacancies"]),
            ("w_norm", INDUSTRY_OPP_WEIGHTS["wage"]),
        ],
    )
    return d

def build_opp_occupation(df):
    d = df.copy()
    d["d_norm"] = norm01(d["demand_mean_pred"])
    d["v_norm"] = norm01(d["vacancies_mean_pred"])
    d["w_norm"] = norm01(d["wage_mean_pred"]) if "wage_mean_pred" in d.columns else np.nan
    d["opp_score"] = weighted_score_from_norms(
        d,
        [
            ("d_norm", OCCUPATION_OPP_WEIGHTS["demand"]),
            ("w_norm", OCCUPATION_OPP_WEIGHTS["wage"]),
            ("v_norm", OCCUPATION_OPP_WEIGHTS["vacancies"]),
        ],
    )
    return d

ind_scored = build_opp_industry(ind_pred_out)
occ_scored = build_opp_occupation(occ_pred_out)

occ_scored["_occ_code"] = occ_scored[occ_col].astype(str).apply(extract_bracket_code)
DROP_TOTAL_ALL_OCC = True
DROP_TOPLEVEL_NOC_0 = True

mask = pd.Series(True, index=occ_scored.index)
if DROP_TOTAL_ALL_OCC:
    mask &= ~occ_scored[occ_col].astype(str).apply(is_total_all_occupations)
if DROP_TOPLEVEL_NOC_0:
    mask &= ~occ_scored["_occ_code"].isin(["0", "00"])
occ_scored = occ_scored[mask].copy()

top10_ind = (ind_scored.sort_values(["GEO", "opp_score"], ascending=[True, False])
             .groupby("GEO").head(10).copy())
top10_occ = (occ_scored.sort_values(["GEO", "opp_score"], ascending=[True, False])
             .groupby("GEO").head(10).copy())

top10_ind.to_csv(TASK_DIR / "TASK_D_top10_industry_opp_per_province.csv", index=False)
top10_occ.to_csv(TASK_DIR / "TASK_D2_top10_occupation_opp_per_province.csv", index=False)

edu_sc = edu_pred_out.copy()
edu_sc["employment_rate_pred"] = pd.to_numeric(edu_sc["employment_rate_pred"], errors="coerce")
edu_sc["edu_norm"] = norm01(edu_sc["employment_rate_pred"])

best_edu = (edu_sc.sort_values(["GEO", "employment_rate_pred"], ascending=[True, False])
            .groupby("GEO").head(1).copy())
best_edu.to_csv(TASK_DIR / "TASK_E_top_education_opp_per_province.csv", index=False)

neet_sc = neet_pred_out.copy()
neet_sc["neet_prop_pred"] = pd.to_numeric(neet_sc["neet_prop_pred"], errors="coerce")
neet_sc["supply_score"] = 1.0 - norm01(neet_sc["neet_prop_pred"])

print(" Wrote TASK_D/TASK_D2/TASK_E.")

prov_rows = []
for prov in COMMON_PROVINCES:
    ind_p = ind_scored[ind_scored["GEO"] == prov]
    occ_p = occ_scored[occ_scored["GEO"] == prov]
    edu_p = best_edu[best_edu["GEO"] == prov]
    sup_p = neet_sc[neet_sc["GEO"] == prov]

    industry_score = safe_mean_topn(ind_p, "opp_score", n=3)
    occupation_score = safe_mean_topn(occ_p, "opp_score", n=3)
    education_score = float(edu_p["edu_norm"].iloc[0]) if not edu_p.empty else np.nan
    supply_score = float(sup_p["supply_score"].iloc[0]) if not sup_p.empty else np.nan

    final_score = (PROVINCE_SCORE_WEIGHTS["industry"]*industry_score + PROVINCE_SCORE_WEIGHTS["occupation"]*occupation_score + PROVINCE_SCORE_WEIGHTS["education"]*education_score + PROVINCE_SCORE_WEIGHTS["supply"]*supply_score)

    top3_ind = ind_p.sort_values("opp_score", ascending=False).head(3)
    top3_occ = occ_p.sort_values("opp_score", ascending=False).head(3)

    prov_rows.append({
        "GEO": prov,
        "final_score": final_score,
        "industry_score_top3_mean": industry_score,
        "occupation_score_top3_mean": occupation_score,
        "education_best_norm": education_score,
        "supply_score": supply_score,
        "top3_industries": " | ".join(top3_ind[naics_col].tolist()) if not top3_ind.empty else "",
        "top3_occupations": " | ".join(top3_occ[occ_col].tolist()) if not top3_occ.empty else "",
        "PRED_YEAR": PRED_YEAR
    })

prov_rank = pd.DataFrame(prov_rows).sort_values("final_score", ascending=False)
prov_rank["rank"] = np.arange(1, len(prov_rank) + 1)
prov_rank.to_csv(TASK_DIR / "TASK_F_final_province_ranking_top3_ind_top3_occ.csv", index=False)

print(" Wrote TASK_F final province ranking.")
print(" All TASK outputs are in:", TASK_DIR)


cls_rows = []

res = backtest_confusion_and_metrics(ind_feat, ["GEO", naics_col], ind_features, "demand_mean",
                                     FEATURE_YEAR, name="Industry", mode="direction")
if res: cls_rows += list(res)
res = backtest_confusion_and_metrics(ind_feat, ["GEO", naics_col], ind_features, "demand_mean",
                                     FEATURE_YEAR, name="Industry", mode="high_vs_low", quantile=0.67)
if res: cls_rows += list(res)

res = backtest_confusion_and_metrics(edu_wide, ["GEO", "Educational attainment"], edu_features, "employment_rate",
                                     FEATURE_YEAR, name="Education", mode="direction")
if res: cls_rows += list(res)
res = backtest_confusion_and_metrics(edu_wide, ["GEO", "Educational attainment"], edu_features, "employment_rate",
                                     FEATURE_YEAR, name="Education", mode="high_vs_low", quantile=0.67)
if res: cls_rows += list(res)

res = backtest_confusion_and_metrics(neet, ["GEO"], neet_features, "neet_prop",
                                     FEATURE_YEAR, name="Supply(NEET)", mode="high_vs_low", quantile=0.67)
if res: cls_rows += list(res)

if cls_rows:
    pd.DataFrame(cls_rows).to_csv(TASK_DIR / "EVAL_backtest_confusion_metrics.csv", index=False)
    print(f"Saved confusion matrices + classification metrics to: {TASK_DIR / 'EVAL_backtest_confusion_metrics.csv'}")

qual_rows = []

for m in ["wage_mean", "vacancies_mean", "demand_mean"]:
    pred_col = f"{m}_pred"
    if pred_col in ind_pred_out.columns and m in ind_feat.columns:
        obs = ind_feat[ind_feat["YEAR"] <= FEATURE_YEAR].copy()
        p_below, p_above = qualitative_sanity_ranges(ind_pred_out, obs, pred_col, m)
        qual_rows.append({"metric": f"Industry {pred_col}", "pct_below_min": p_below, "pct_above_max": p_above})

for m in ["employment_rate", "unemployment_rate"]:
    pred_col = f"{m}_pred"
    if pred_col in edu_pred_out.columns and m in edu_wide.columns:
        obs = edu_wide[edu_wide["YEAR"] <= FEATURE_YEAR].copy()
        p_below, p_above = qualitative_sanity_ranges(edu_pred_out, obs, pred_col, m)
        qual_rows.append({"metric": f"Education {pred_col}", "pct_below_min": p_below, "pct_above_max": p_above})

if "neet_prop_pred" in neet_pred_out.columns and "neet_prop" in neet.columns:
    obs = neet[neet["YEAR"] <= FEATURE_YEAR].copy()
    p_below, p_above = qualitative_sanity_ranges(neet_pred_out, obs, "neet_prop_pred", "neet_prop")
    qual_rows.append({"metric": "NEET neet_prop_pred", "pct_below_min": p_below, "pct_above_max": p_above})

if qual_rows:
    qual_df = pd.DataFrame(qual_rows)
    qual_df.to_csv(TASK_DIR / "QUAL_sanity_prediction_ranges.csv", index=False)
    print(f"\nQualitative sanity report saved: {TASK_DIR / 'QUAL_sanity_prediction_ranges.csv'}")
    print(qual_df.to_string(index=False))

comp = prov_rank[["GEO","industry_score_top3_mean","occupation_score_top3_mean","education_best_norm","supply_score"]].copy()
base_w = {
    "industry_score_top3_mean": 0.35,
    "occupation_score_top3_mean": 0.25,
    "education_best_norm": 0.20,
    "supply_score": 0.20,
}
ws = weight_sensitivity(comp, base_w, n_sims=500, noise=0.05, seed=0)
ws.to_csv(TASK_DIR / "QUAL_weight_sensitivity.csv", index=False)
print(f"\n Weight sensitivity saved: {TASK_DIR / 'QUAL_weight_sensitivity.csv'}")
print(ws.to_string(index=False))

def _choose_numbered_list(options, title, allow_multi=True, allow_all=True, default_all=True):
    """
    options: list of strings
    returns: list of selected strings OR None if ALL
    """
    if not options:
        print(f"\n[{title}] No options available.")
        return None

    print(f"\n--- {title} ---")
    print(f"Total options: {len(options)}")

    if allow_all:
        print("Type:")
        print("  - Enter  -> " + ("ALL" if default_all else "skip"))
        print("  - all    -> ALL")
    if allow_multi:
        print("  - 1,3,5  -> choose multiple")
    else:
        print("  - 4      -> choose one")

    ans = input("Your selection: ").strip().lower()

    if ans == "" and default_all and allow_all:
        return None
    if ans == "" and not default_all:
        return []
    if allow_all and ans == "all":
        return None

    nums = re.findall(r"\d+", ans)
    if not nums:
        print("No valid selection detected. Using ALL.")
        return None

    idxs = [int(x) for x in nums]
    idxs = [i for i in idxs if 1 <= i <= len(options)]
    if not idxs:
        print("Selection out of range. Using ALL.")
        return None

    if not allow_multi:
        idxs = [idxs[0]]

    return [options[i-1] for i in idxs]

def _rank_labels_overall(df, label_col, score_col="opp_score"):
    tmp = (df.groupby(label_col, as_index=False)[score_col]
             .mean()
             .rename(columns={score_col: "mean_opp"}))
    return tmp.sort_values("mean_opp", ascending=False)

def _rank_labels_for_province(df, label_col, prov, score_col="opp_score"):
    d = df[df["GEO"] == prov].copy()
    if d.empty:
        return d.assign(mean_opp=np.nan)
    tmp = (d.groupby(label_col, as_index=False)[score_col]
             .mean()
             .rename(columns={score_col: "mean_opp"}))
    return tmp.sort_values("mean_opp", ascending=False)

def _display_top_ranked(rank_df, label_col, n=30, width=95):
    show = rank_df.head(n).copy()
    if show.empty:
        return []
    labels = show[label_col].astype(str).tolist()
    for i, (lab, sc) in enumerate(zip(labels, show["mean_opp"].tolist()), start=1):
        print(f"{i:>3}. {short_label(lab, width=width)}  (mean_opp={sc:.4f})")
    return labels

def interactive_report_menu():
    print("\n===============================")
    print(" BEST PLACE FOR ME (CLEAN MENU)")
    print("===============================")
    print("Tip: you will SELECT from lists (no fragile keyword matching).\n")

    edu_vals = sorted(edu_pred_out["Educational attainment"].dropna().astype(str).unique().tolist())

    print("\n--- Education choices ---")
    for i, lab in enumerate(edu_vals, start=1):
        print(f"{i:>3}. {lab}")

    edu_choice = _choose_numbered_list(
        edu_vals,
        "Education selection (choose ONE or ALL)",
        allow_multi=False,
        allow_all=True,
        default_all=True
    )
    selected_edu_label = edu_choice[0] if isinstance(edu_choice, list) and len(edu_choice) == 1 else None

    if selected_edu_label:
        edu_sel = edu_pred_out[edu_pred_out["Educational attainment"].astype(str) == selected_edu_label].copy()
        edu_sel["edu_norm_user"] = norm01(pd.to_numeric(edu_sel["employment_rate_pred"], errors="coerce"))
        edu_best_user = edu_sel.sort_values(["GEO", "employment_rate_pred"], ascending=[True, False]).groupby("GEO").head(1)
        used_edu_mode = f"Selected education: {selected_edu_label}"
    else:
        edu_best_user = best_edu.copy()
        used_edu_mode = "Education: BEST available per province (no filter)"

    print("\n--- Industry selection ---")
    print("Choose how to list industries:")
    print("  1) Top industries overall (mean opportunity across provinces)")
    print("  2) Top industries for a specific province")
    mode_ind = input("Pick 1 or 2 [default 1]: ").strip()
    mode_ind = "2" if mode_ind == "2" else "1"

    prov_for_ind = None
    if mode_ind == "2":
        print("\nPick a province for industry listing:")
        for i, p in enumerate(COMMON_PROVINCES, start=1):
            print(f"{i:>2}. {p}")
        p_ans = input("Province number [default 1]: ").strip()
        try:
            prov_for_ind = COMMON_PROVINCES[max(0, int(p_ans)-1)] if p_ans else COMMON_PROVINCES[0]
        except:
            prov_for_ind = COMMON_PROVINCES[0]
        ind_rank = _rank_labels_for_province(ind_scored, naics_col, prov_for_ind)
        print(f"\nTop industries for: {prov_for_ind}")
    else:
        ind_rank = _rank_labels_overall(ind_scored, naics_col)
        print("\nTop industries overall")

    search_ind = input("Optional search term to filter displayed industries [Enter to skip]: ").strip().lower()
    if search_ind:
        ind_rank = ind_rank[ind_rank[naics_col].astype(str).str.lower().str.contains(re.escape(search_ind), na=False)].copy()

    show_n = input("How many industries to display? [default 30]: ").strip()
    try:
        show_n = int(show_n) if show_n else 30
        show_n = max(5, min(show_n, 200))
    except:
        show_n = 30

    ind_display_labels = _display_top_ranked(ind_rank, naics_col, n=show_n)
    ind_choice = _choose_numbered_list(ind_display_labels, "Select industries (ONE/MANY/ALL from the displayed list)", allow_multi=True, allow_all=True, default_all=True)

    if ind_choice is None:
        ind_rows = ind_scored.copy()
        used_ind_mode = "Industries: ALL"
    else:
        ind_rows = ind_scored[ind_scored[naics_col].isin(ind_choice)].copy()
        if ind_rows.empty:
            print("\nNo rows after industry selection. Falling back to ALL industries.")
            ind_rows = ind_scored.copy()
            used_ind_mode = "Industries: ALL (fallback)"
        else:
            used_ind_mode = f"Industries: {len(ind_choice)} selected"

    print("\n--- Occupation selection ---")
    print("Choose how to list occupations:")
    print("  1) Top occupations overall (mean opportunity across provinces)")
    print("  2) Top occupations for a specific province")
    mode_occ = input("Pick 1 or 2 [default 1]: ").strip()
    mode_occ = "2" if mode_occ == "2" else "1"

    prov_for_occ = None
    if mode_occ == "2":
        print("\nPick a province for occupation listing:")
        for i, p in enumerate(COMMON_PROVINCES, start=1):
            print(f"{i:>2}. {p}")
        p_ans = input("Province number [default 1]: ").strip()
        try:
            prov_for_occ = COMMON_PROVINCES[max(0, int(p_ans)-1)] if p_ans else COMMON_PROVINCES[0]
        except:
            prov_for_occ = COMMON_PROVINCES[0]
        occ_rank = _rank_labels_for_province(occ_scored, occ_col, prov_for_occ)
        print(f"\nTop occupations for: {prov_for_occ}")
    else:
        occ_rank = _rank_labels_overall(occ_scored, occ_col)
        print("\nTop occupations overall")

    search_occ = input("Optional search term to filter displayed occupations [Enter to skip]: ").strip().lower()
    if search_occ:
        occ_rank = occ_rank[occ_rank[occ_col].astype(str).str.lower().str.contains(re.escape(search_occ), na=False)].copy()

    show_n = input("How many occupations to display? [default 30]: ").strip()
    try:
        show_n = int(show_n) if show_n else 30
        show_n = max(5, min(show_n, 200))
    except:
        show_n = 30

    occ_display_labels = _display_top_ranked(occ_rank, occ_col, n=show_n)
    occ_choice = _choose_numbered_list(occ_display_labels, "Select occupations (ONE/MANY/ALL from the displayed list)", allow_multi=True, allow_all=True, default_all=True)

    if occ_choice is None:
        occ_rows = occ_scored.copy()
        used_occ_mode = "Occupations: ALL (filtered totals removed)"
    else:
        occ_rows = occ_scored[occ_scored[occ_col].isin(occ_choice)].copy()
        if occ_rows.empty:
            print("\nNo rows after occupation selection. Falling back to ALL occupations (filtered).")
            occ_rows = occ_scored.copy()
            used_occ_mode = "Occupations: ALL (fallback, filtered)"
        else:
            used_occ_mode = f"Occupations: {len(occ_choice)} selected"

    print("\n========================")
    print(" USER SETTINGS USED")
    print("========================")
    print(used_edu_mode)
    print(used_ind_mode)
    print(used_occ_mode)

    rows = []
    details = {}
    for prov in COMMON_PROVINCES:
        ind_p = ind_rows[ind_rows["GEO"] == prov]
        occ_p = occ_rows[occ_rows["GEO"] == prov]
        edu_p = edu_best_user[edu_best_user["GEO"] == prov]
        sup_p = neet_sc[neet_sc["GEO"] == prov]

        industry_score = safe_mean_topn(ind_p, "opp_score", n=3)
        occupation_score = safe_mean_topn(occ_p, "opp_score", n=3)

        if edu_p.empty:
            edu_fallback = best_edu[best_edu["GEO"] == prov]
            education_score = float(edu_fallback["edu_norm"].iloc[0]) if not edu_fallback.empty else np.nan
            edu_label_used = (edu_fallback["Educational attainment"].iloc[0] if not edu_fallback.empty else "")
        else:
            if "edu_norm_user" in edu_p.columns:
                education_score = float(edu_p["edu_norm_user"].iloc[0])
            else:
                education_score = float(edu_p["edu_norm"].iloc[0]) if "edu_norm" in edu_p.columns else np.nan
            edu_label_used = edu_p["Educational attainment"].iloc[0]

        supply_score = float(sup_p["supply_score"].iloc[0]) if not sup_p.empty else np.nan
        final_score = (PROVINCE_SCORE_WEIGHTS["industry"]*industry_score + PROVINCE_SCORE_WEIGHTS["occupation"]*occupation_score + PROVINCE_SCORE_WEIGHTS["education"]*education_score + PROVINCE_SCORE_WEIGHTS["supply"]*supply_score)

        top3_ind = ind_p.sort_values("opp_score", ascending=False).head(3)
        top3_occ = occ_p.sort_values("opp_score", ascending=False).head(3)

        details[prov] = {
            "top3_industries": top3_ind[naics_col].astype(str).tolist() if not top3_ind.empty else [],
            "top3_occupations": top3_occ[occ_col].astype(str).tolist() if not top3_occ.empty else [],
            "edu_label_used": str(edu_label_used),
        }

        rows.append({"GEO": prov, "final_score": final_score})

    user_rank = pd.DataFrame(rows).sort_values("final_score", ascending=False)
    user_rank["rank"] = np.arange(1, len(user_rank) + 1)

    print("\n========================")
    print(" TOP 10 PROVINCES (USER)")
    print("========================")
    print(user_rank.head(10).to_string(index=False))

    print("\n========================")
    print(" TOP 3 DETAILS")
    print("========================")
    for prov in user_rank.head(3)["GEO"].tolist():
        print(f"\n{prov}")
        print(f"  Education used: {details[prov]['edu_label_used']}")
        inds = details[prov]["top3_industries"]
        occs = details[prov]["top3_occupations"]

        print("  Top industries (from your selection):")
        if inds:
            for x in inds:
                print("   -", clean_wrap(x, width=88))
        else:
            print("   - (none)")

        print("  Top occupations (from your selection):")
        occs = [o for o in occs if not is_total_all_occupations(o)]
        occs = [o for o in occs if extract_bracket_code(o) not in ("0", "00")]
        if occs:
            for x in occs:
                print("   -", clean_wrap(x, width=88))
        else:
            print("   - (none after filtering)")

try:
    interactive_report_menu()
except Exception as e:
    print("\nInteractive report skipped (non-interactive environment). Error:", e)