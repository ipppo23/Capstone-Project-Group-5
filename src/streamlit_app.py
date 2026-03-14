
from __future__ import annotations

from pathlib import Path
import re
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import streamlit as st

try:
    import plotly.express as px
except Exception:
    px = None

from project_paths import build_project_paths, resolve_optional_task_file, resolve_task_file

PATHS = build_project_paths(Path(__file__))


def get_required_task_files() -> Dict[str, Path]:
    return {
        "ind": resolve_task_file("TASK_B_industry_predictions.csv", PATHS),
        "occ": resolve_task_file("TASK_B2_occupation_predictions.csv", PATHS),
        "edu": resolve_task_file("TASK_C_education_predictions.csv", PATHS),
        "neet": resolve_task_file("TASK_C2_supply_neet_predictions.csv", PATHS),
    }


def get_optional_task_files() -> Dict[str, Optional[Path]]:
    return {
        "overall": resolve_optional_task_file([
            "TASK_F_final_province_ranking_top3_ind_top3_occ.csv",
            "TASK_F_final_province_ranking.csv",
            "TASK_F_province_ranking.csv",
        ], PATHS),
        "ind_metrics": resolve_optional_task_file([
            "EVAL_backtest_regression_industry.csv",
            "EVAL_industry_metrics.csv",
        ], PATHS),
        "occ_metrics": resolve_optional_task_file([
            "EVAL_backtest_regression_occupation.csv",
            "EVAL_occupation_metrics.csv",
        ], PATHS),
        "cls_metrics": resolve_optional_task_file([
            "EVAL_backtest_confusion_metrics.csv",
        ], PATHS),
        "sanity": resolve_optional_task_file([
            "QUAL_sanity_prediction_ranges.csv",
            "QUAL_prediction_range_sanity.csv",
        ], PATHS),
        "weights": resolve_optional_task_file([
            "QUAL_weight_sensitivity.csv",
            "QUAL_prescriptive_weight_sensitivity.csv",
        ], PATHS),
    }


W_DEMAND = 0.35
W_WAGE = 0.35
W_EMPLOYMENT = 0.20
W_SUPPLY = 0.10


def inject_css() -> None:
    st.markdown(
        """
        <style>
          .stApp { background: #f6f8fc; }
          section[data-testid="stSidebar"] { background: #eef3fb; }
          .block-container { padding-top: 1.6rem; padding-bottom: 2.0rem; }
          html, body, [class*="css"]  { font-size: 17px; }
          h1 { font-size: 2.10rem !important; }
          h2 { font-size: 1.75rem !important; }
          h3 { font-size: 1.38rem !important; }
          .small-muted { color: #6b7280; font-size: 0.98rem; }
          .metric-card {
            background: #ffffff;
            padding: 1.2rem 1.3rem;
            border-radius: 16px;
            box-shadow: 0 1px 2px rgba(0,0,0,0.04);
            border: 1px solid #e5e7eb;
          }
          .metric-card .big { font-size: 2.0rem; font-weight: 700; margin: 0.2rem 0 0.8rem; }
          .metric-card .line { margin: 0.30rem 0; font-size: 1.05rem; }
          .tab-note {
            background: #ffffff;
            padding: 0.9rem 1.0rem;
            border: 1px solid #e5e7eb;
            border-radius: 12px;
          }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner=False)
def load_data() -> Dict[str, pd.DataFrame]:
    task_files = get_required_task_files()
    ind = pd.read_csv(task_files['ind'])
    occ = pd.read_csv(task_files['occ'])
    edu = pd.read_csv(task_files['edu'])
    neet = pd.read_csv(task_files['neet'])

    ind = ind.rename(columns={
        "North American Industry Classification System (NAICS)": "industry",
        "PRED_YEAR": "pred_year",
        "GEO": "province",
    })
    occ = occ.rename(columns={
        "National Occupational Classification": "occupation",
        "PRED_YEAR": "pred_year",
        "GEO": "province",
    })
    edu = edu.rename(columns={
        "Educational attainment": "education",
        "PRED_YEAR": "pred_year",
        "GEO": "province",
    })
    neet = neet.rename(columns={
        "PRED_YEAR": "pred_year",
        "GEO": "province",
    })

    for df in (ind, occ, edu, neet):
        for c in df.columns:
            if df[c].dtype == object:
                df[c] = df[c].astype(str).str.strip()

    ind["industry_group"] = ind["industry"].map(classify_industry)
    occ["occupation_group"] = occ["occupation"].map(classify_occupation)

    return {"ind": ind, "occ": occ, "edu": edu, "neet": neet}


def _read_optional_csv(path: Optional[Path]) -> Optional[pd.DataFrame]:
    if path is None:
        return None
    try:
        df = pd.read_csv(path)
        if df.empty or len(df.columns) == 0:
            return None
        return df
    except Exception:
        return None


@st.cache_data(show_spinner=False)
def load_optional_data() -> Dict[str, Optional[pd.DataFrame]]:
    optional_files = get_optional_task_files()
    return {
        "overall": _read_optional_csv(optional_files['overall']),
        "ind_metrics": _read_optional_csv(optional_files['ind_metrics']),
        "occ_metrics": _read_optional_csv(optional_files['occ_metrics']),
        "cls_metrics": _read_optional_csv(optional_files['cls_metrics']),
        "sanity": _read_optional_csv(optional_files['sanity']),
        "weights": _read_optional_csv(optional_files['weights']),
    }


def strip_codes(label: Optional[str]) -> str:
    if label is None or (isinstance(label, float) and pd.isna(label)):
        return "—"
    s = str(label).strip()
    s = re.sub(r"\s*\[[^\]]+\]", "", s)
    return re.sub(r"\s+", " ", s).strip()


def fmt_money(v: Optional[float]) -> str:
    if v is None or pd.isna(v):
        return "—"
    return f"${float(v):,.2f}"


def fmt_num(v: Optional[float], digits: int = 1) -> str:
    if v is None or pd.isna(v):
        return "—"
    return f"{float(v):,.{digits}f}"


def classify_industry(label: str) -> str:
    s = label.lower()
    rules = [
        (r"hospital|ambulatory|nursing|health care|social assistance", "health"),
        (r"educational|school|university|college", "education"),
        (r"construction", "construction"),
        (r"manufacturing|factory|wood product|primary metal|fabricated metal|machinery|plastics|textile|apparel|paper|furniture|chemical|beverage|food manufacturing", "manufacturing"),
        (r"finance|insurance|funds|securities|credit|bank|depository", "finance"),
        (r"professional, scientific|computer|software|data processing|information|publishing|telecommunications|broadcasting|computing infrastructure", "tech_professional"),
        (r"public administration|justice|police|military", "public_service"),
        (r"retail", "retail"),
        (r"wholesale", "wholesale"),
        (r"accommodation|food services|restaurant|traveller|lodging", "hospitality"),
        (r"transport|warehousing|courier|postal|truck|transit|air|rail|water transportation", "transport"),
        (r"mining|oil and gas|forestry|fishing|hunting|agriculture|crop|animal production", "resources"),
        (r"real estate|rental|leasing", "real_estate"),
        (r"arts|entertainment|recreation", "arts"),
        (r"administrative and support|waste management|remediation", "admin_support"),
        (r"utilities", "utilities"),
    ]
    for pattern, group in rules:
        if re.search(pattern, s):
            return group
    return "other"


def classify_occupation(label: str) -> str:
    s = label.lower()
    rules = [
        (r"health|nurse|medical|dental|therapy|pharmac|paramedic|hospital|support of health", "health"),
        (r"teaching|education|library|training|professor|instructor", "education"),
        (r"trade|transport|equipment operators|construction|mechanic|repair|installer", "construction"),
        (r"manufacturing|utilities|processing", "manufacturing"),
        (r"business|finance|administrative|financial|supply chain logistics|office", "finance_business"),
        (r"natural and applied sciences|engineer|computer|data|tech", "tech_professional"),
        (r"legislative|government|public protection|social|community|legal", "public_service"),
        (r"sales|service|cashier|customer|food counter|cleaning|travel", "hospitality_retail"),
        (r"management", "management"),
        (r"art|culture|recreation|sport", "arts"),
        (r"natural resources|agriculture|production", "resources"),
    ]
    for pattern, group in rules:
        if re.search(pattern, s):
            return group
    return "other"


INDUSTRY_TO_OCC_GROUPS: Dict[str, List[str]] = {
    "health": ["health", "public_service", "management"],
    "education": ["education", "public_service", "management"],
    "construction": ["construction", "management"],
    "manufacturing": ["manufacturing", "construction", "management", "finance_business"],
    "finance": ["finance_business", "management", "tech_professional"],
    "tech_professional": ["tech_professional", "finance_business", "management"],
    "public_service": ["public_service", "management", "finance_business"],
    "retail": ["hospitality_retail", "management", "finance_business"],
    "wholesale": ["hospitality_retail", "finance_business", "management", "construction"],
    "hospitality": ["hospitality_retail", "management"],
    "transport": ["construction", "hospitality_retail", "management", "finance_business"],
    "resources": ["resources", "construction", "management"],
    "real_estate": ["finance_business", "management", "hospitality_retail"],
    "arts": ["arts", "management", "hospitality_retail"],
    "admin_support": ["finance_business", "hospitality_retail", "management"],
    "utilities": ["construction", "management", "tech_professional"],
    "other": ["management", "finance_business", "other"],
}


def minmax(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    if s.notna().sum() == 0:
        return pd.Series(np.nan, index=s.index)
    mn, mx = s.min(), s.max()
    if pd.isna(mn) or pd.isna(mx) or mx == mn:
        return pd.Series(1.0, index=s.index)
    return (s - mn) / (mx - mn)


def build_province_scores(ind_df: pd.DataFrame, edu_df: pd.DataFrame, neet_df: pd.DataFrame, selected_education: str, selected_industry: str, selected_year: int) -> pd.DataFrame:
    prov_ind = ind_df[(ind_df["pred_year"] == selected_year) & (ind_df["industry"] == selected_industry)].copy()
    prov_edu = edu_df[(edu_df["pred_year"] == selected_year) & (edu_df["education"] == selected_education)].copy()
    prov_neet = neet_df[neet_df["pred_year"] == selected_year].copy()

    merged = prov_ind.merge(
        prov_edu[["province", "employment_rate_pred", "unemployment_rate_pred"]],
        on="province",
        how="left",
    ).merge(
        prov_neet[["province", "neet_prop_pred"]],
        on="province",
        how="left",
    )

    merged["demand_n"] = minmax(merged["demand_mean_pred"])
    merged["wage_n"] = minmax(merged["wage_mean_pred"])
    merged["employment_n"] = minmax(merged["employment_rate_pred"])
    merged["supply_n"] = minmax(-merged["neet_prop_pred"])

    merged["opportunity_score_01"] = (
        W_DEMAND * merged["demand_n"].fillna(0)
        + W_WAGE * merged["wage_n"].fillna(0)
        + W_EMPLOYMENT * merged["employment_n"].fillna(0)
        + W_SUPPLY * merged["supply_n"].fillna(0)
    )
    merged["opportunity_score"] = (100 * merged["opportunity_score_01"]).round(1)
    merged["rank"] = merged["opportunity_score"].rank(method="dense", ascending=False).astype(int)
    return merged.sort_values(["opportunity_score", "province"], ascending=[False, True])


def related_occupations_for_industry(occ_df: pd.DataFrame, selected_industry: str, province: str, selected_year: int) -> pd.DataFrame:
    ind_group = classify_industry(selected_industry)
    allowed = set(INDUSTRY_TO_OCC_GROUPS.get(ind_group, ["other"]))
    occ = occ_df[(occ_df["pred_year"] == selected_year) & (occ_df["province"] == province)].copy()
    occ = occ[occ["occupation_group"].isin(allowed)].copy()
    if occ.empty:
        occ = occ_df[(occ_df["pred_year"] == selected_year) & (occ_df["province"] == province)].copy()

    occ["demand_n"] = minmax(occ["demand_mean_pred"])
    occ["wage_n"] = minmax(occ["wage_mean_pred"])
    occ["occupation_score"] = (100 * (0.5 * occ["demand_n"].fillna(0) + 0.5 * occ["wage_n"].fillna(0))).round(1)
    occ["rank_in_province"] = occ["occupation_score"].rank(method="dense", ascending=False).astype(int)
    return occ.sort_values(["occupation_score", "occupation"], ascending=[False, True])


def province_chart(df: pd.DataFrame):
    if px is None or df.empty:
        return None
    show = df.copy()
    show["Province"] = show["province"]
    fig = px.bar(
        show,
        x="opportunity_score",
        y="Province",
        orientation="h",
        hover_data={
            "wage_mean_pred": ':.2f',
            "employment_rate_pred": ':.1f',
            "province": False,
            "opportunity_score": ':.1f',
        },
    )
    fig.update_layout(
        height=480,
        xaxis_title="Opportunity score (0–100)",
        yaxis_title="Province",
        margin=dict(l=10, r=10, t=10, b=10),
        font=dict(size=15),
        showlegend=False,
    )
    fig.update_yaxes(categoryorder="array", categoryarray=show["Province"].tolist()[::-1])
    return fig


def occupation_chart(df: pd.DataFrame):
    if px is None or df.empty:
        return None
    show = df.copy().head(15)
    show["Occupation"] = show["occupation"].map(strip_codes)
    fig = px.bar(
        show,
        x="wage_mean_pred",
        y="Occupation",
        orientation="h",
        hover_data={
            "wage_mean_pred": ':.2f',
            "occupation": False,
        },
    )
    fig.update_layout(
        height=520,
        xaxis_title="Predicted hourly wage ($)",
        yaxis_title="Occupation",
        margin=dict(l=10, r=10, t=10, b=10),
        font=dict(size=15),
        showlegend=False,
    )
    fig.update_yaxes(categoryorder="array", categoryarray=show["Occupation"].tolist()[::-1])
    return fig


def standardize_overall_province_df(df: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    if df is None or df.empty:
        return None
    out = df.copy()
    if "GEO" in out.columns and "province" not in out.columns:
        out = out.rename(columns={"GEO": "province"})
    if "final_score" in out.columns and "score" not in out.columns:
        vals = pd.to_numeric(out["final_score"], errors="coerce")
        out["score"] = np.where(vals.max() <= 1.01, vals * 100, vals)
    elif "score" in out.columns:
        vals = pd.to_numeric(out["score"], errors="coerce")
        out["score"] = np.where(vals.max() <= 1.01, vals * 100, vals)
    else:
        return None
    if "rank" not in out.columns:
        out["rank"] = out["score"].rank(method="dense", ascending=False).astype(int)
    return out.sort_values(["rank", "province"]).reset_index(drop=True)


def overall_province_chart(df: pd.DataFrame):
    if px is None or df.empty:
        return None
    show = df.copy()
    show["Province"] = show["province"]
    fig = px.bar(show, x="score", y="Province", orientation="h")
    fig.update_layout(
        height=520,
        xaxis_title="final_score (0–100)",
        yaxis_title="Province",
        margin=dict(l=10, r=10, t=10, b=10),
        font=dict(size=15),
        showlegend=False,
    )
    fig.update_yaxes(categoryorder="array", categoryarray=show["Province"].tolist()[::-1])
    return fig


def metric_table(df: Optional[pd.DataFrame], title: str):
    st.markdown(f"### {title}")
    if df is None or df.empty:
        st.info(f"{title} file not found or empty.")
    else:
        st.dataframe(df, use_container_width=True, hide_index=True)


def render_prescriptive_tab(ind, occ, edu, neet):
    st.subheader("Conditional Path Recommendations")
    st.caption("New prescriptive logic: Education → Industry → Province → Occupation")

    available_years = sorted(set(ind["pred_year"]).intersection(edu["pred_year"]).intersection(neet["pred_year"]))
    default_year = max(available_years)

    with st.sidebar:
        st.header("Inputs")
        selected_year = st.selectbox("Prediction year", available_years, index=available_years.index(default_year), key="pred_year")
        education_options = sorted(edu.loc[edu["pred_year"] == selected_year, "education"].dropna().unique().tolist())
        selected_education = st.selectbox("Education", education_options, key="education")
        industry_options = sorted(ind.loc[ind["pred_year"] == selected_year, "industry"].dropna().unique().tolist())
        selected_industry = st.selectbox("Target industry", industry_options, format_func=strip_codes, key="industry")

    province_scores = build_province_scores(ind, edu, neet, selected_education, selected_industry, selected_year)
    if province_scores.empty:
        st.error("No province-level candidates were found for this education + industry path.")
        return

    best_province = province_scores.iloc[0]["province"]
    current_row = province_scores.loc[province_scores["province"] == best_province].iloc[0]
    related_occ = related_occupations_for_industry(occ, selected_industry, best_province, selected_year)

    c1, c2 = st.columns([1.2, 1])
    with c1:
        st.subheader("Your Profile")
        st.markdown(
            f"**Education:** {selected_education}  \n"
            f"**Industry:** {strip_codes(selected_industry)}  \n"
            f"**Best province (auto-selected):** {best_province}"
        )
    with c2:
        st.subheader("Best Result for your profile")
        st.markdown(
            f"<div class='metric-card'>"
            f"<div class='small-muted'>Opportunity score</div>"
            f"<div class='big'>{fmt_num(current_row['opportunity_score'])} / 100</div>"
            f"<div class='line'><b>Predicted hourly wage:</b> {fmt_money(current_row['wage_mean_pred'])}</div>"
            f"<div class='line'><b>Predicted employment:</b> {fmt_num(current_row['employment_rate_pred'])}%</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.subheader("Provinces ranking for your selected education & industry")
    st.caption("Hover over the bars to see the province details.")
    fig1 = province_chart(province_scores)
    if fig1 is not None:
        st.plotly_chart(fig1, use_container_width=True)

    province_view = province_scores[[
        "rank", "province", "opportunity_score", "wage_mean_pred", "employment_rate_pred"
    ]].rename(columns={
        "rank": "Rank",
        "province": "Province",
        "opportunity_score": "Opportunity score (0–100)",
        "wage_mean_pred": "Predicted hourly wage",
        "employment_rate_pred": "Predicted employment (%)",
    }).copy()
    province_view["Predicted hourly wage"] = province_view["Predicted hourly wage"].map(fmt_money)
    st.dataframe(province_view, use_container_width=True, hide_index=True)

    st.subheader("Top occupations for your selected education & industry")
    st.caption("Hover over the bars to see the hourly wage details.")
    if related_occ.empty:
        st.info("No related occupations were found for this industry in the selected province.")
    else:
        fig2 = occupation_chart(related_occ)
        if fig2 is not None:
            st.plotly_chart(fig2, use_container_width=True)

        occ_view = related_occ[[
            "rank_in_province", "occupation", "wage_mean_pred"
        ]].rename(columns={
            "rank_in_province": "Rank",
            "occupation": "Occupation",
            "wage_mean_pred": "Predicted hourly wage",
        }).copy()
        occ_view["Occupation"] = occ_view["Occupation"].map(strip_codes)
        occ_view["Predicted hourly wage"] = occ_view["Predicted hourly wage"].map(fmt_money)
        st.dataframe(occ_view.head(15), use_container_width=True, hide_index=True)

    with st.expander("How this app is working"):
        st.markdown(
            "- The app first filters candidates using your **education** and **target industry**.\n"
            "- It then scores provinces using the same opportunity-score idea, but only within that path.\n"
            "- After provinces are ranked, it attaches **related occupations** for the best province and selected industry.\n"
            "- This changes the **prescriptive flow**, not the ML predictions themselves."
        )
        st.code(
            "Opportunity Score = 0.35 × demand signal + 0.35 × predicted wage + 0.20 × predicted employment + 0.10 × supply score"
        )


def render_province_outlook_tab(optional_data, ind, occ):
    st.subheader("Province Outlook (overall, from your pipeline outputs)")
    st.caption("Overall province opportunity score from the overall ranking file.")

    overall_df = standardize_overall_province_df(optional_data.get("overall"))
    if overall_df is None or overall_df.empty:
        st.info("Overall province ranking file was not found. Add a TASK_F province ranking file to show this tab.")
        return

    left, right = st.columns([1.1, 0.9])
    with left:
        show = overall_df[["province", "score"]].rename(columns={
            "province": "GEO",
            "score": "final_score (0–100)",
        })
        st.dataframe(show, use_container_width=True, hide_index=True)
    with right:
        fig = overall_province_chart(overall_df)
        if fig is not None:
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("Province Details")
    selected_province = st.selectbox("Select a province", overall_df["province"].tolist(), key="province_outlook_select")
    score_row = overall_df.loc[overall_df["province"] == selected_province].iloc[0]

    top_ind = ind[ind["province"] == selected_province].copy()
    if not top_ind.empty:
        top_ind = top_ind.sort_values("wage_mean_pred", ascending=False).head(3)
    top_occ = occ[occ["province"] == selected_province].copy()
    if not top_occ.empty:
        top_occ = top_occ.sort_values("wage_mean_pred", ascending=False).head(3)

    st.markdown(
        f"<div class='metric-card'>"
        f"<div class='big'>{selected_province} – Summary</div>"
        f"<div class='small-muted'>Overall score (0–100)</div>"
        f"<div class='big'>{fmt_num(score_row['score'])}</div>"
        f"<div class='line'><b>Top industries</b></div>"
        f"<div class='line'>{' | '.join(top_ind['industry'].map(strip_codes).tolist()) if not top_ind.empty else '—'}</div>"
        f"<div class='line' style='margin-top:0.8rem;'><b>Top occupations</b></div>"
        f"<div class='line'>{' | '.join(top_occ['occupation'].map(strip_codes).tolist()) if not top_occ.empty else '—'}</div>"
        f"</div>",
        unsafe_allow_html=True,
    )


def render_confusion_matrix(df: Optional[pd.DataFrame]):
    st.subheader("Confusion Matrix (Backtest)")
    if df is None or df.empty:
        st.info("Confusion-metrics file was not found or is empty.")
        return

    # normalize some text fields
    work = df.copy()
    for col in ["dataset", "target", "mode", "model"]:
        if col in work.columns:
            work[col] = work[col].astype(str)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        ds = st.selectbox("Dataset", sorted(work["dataset"].dropna().unique().tolist()), key="cm_ds")
    temp = work[work["dataset"] == ds]
    with c2:
        tgt = st.selectbox("Target", sorted(temp["target"].dropna().unique().tolist()), key="cm_tgt")
    temp = temp[temp["target"] == tgt]
    with c3:
        mode = st.selectbox("Mode", sorted(temp["mode"].dropna().unique().tolist()), key="cm_mode")
    temp = temp[temp["mode"] == mode]
    with c4:
        model = st.selectbox("Model", sorted(temp["model"].dropna().unique().tolist()), key="cm_model")

    row = temp[temp["model"] == model]
    if row.empty:
        st.info("No row available for that combination.")
        return
    row = row.iloc[0]

    left, right = st.columns([0.42, 0.58])

    with left:
        matrix_df = pd.DataFrame(
            [[row.get("tn", 0), row.get("fp", 0)], [row.get("fn", 0), row.get("tp", 0)]],
            index=["Actual 0", "Actual 1"],
            columns=["Pred 0", "Pred 1"],
        )
        st.dataframe(matrix_df, use_container_width=True)

    with right:
        metrics_df = pd.DataFrame({
            "Value": [
                row.get("accuracy"),
                row.get("precision"),
                row.get("recall"),
                row.get("f1"),
                row.get("roc_auc"),
                row.get("n_test"),
                f"{int(row['backtest_input_year'])}→{int(row['backtest_pred_year'])}" if pd.notna(row.get("backtest_input_year")) and pd.notna(row.get("backtest_pred_year")) else row.get("backtest"),
            ]
        }, index=["Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "n_test", "Backtest"])
        st.dataframe(metrics_df, use_container_width=True)

    st.markdown("---")
    st.subheader("Full Confusion Metrics Table")
    st.dataframe(work, use_container_width=True, hide_index=True)


def render_regression_backtest(ind_df: Optional[pd.DataFrame], occ_df: Optional[pd.DataFrame]):
    st.subheader("Regression Backtest (Naive vs Ridge)")
    st.markdown("### Industry regression backtest")
    if ind_df is None or ind_df.empty:
        st.info("Industry regression backtest file was not found or is empty.")
    else:
        st.dataframe(ind_df, use_container_width=True, hide_index=True)

    st.markdown("### Occupation regression backtest")
    if occ_df is None or occ_df.empty:
        st.info("Occupation regression backtest file was not found or is empty.")
    else:
        st.dataframe(occ_df, use_container_width=True, hide_index=True)


def render_sanity_sensitivity(sanity_df: Optional[pd.DataFrame], weight_df: Optional[pd.DataFrame]):
    st.subheader("Prediction Range Sanity Check")
    if sanity_df is None or sanity_df.empty:
        st.info("Sanity-check file was not found or is empty.")
    else:
        st.dataframe(sanity_df, use_container_width=True, hide_index=True)

    st.markdown("### Weight Sensitivity (Prescriptive)")
    if weight_df is None or weight_df.empty:
        st.info("Weight-sensitivity file was not found or is empty.")
    else:
        st.dataframe(weight_df, use_container_width=True, hide_index=True)


def render_model_metrics_tab(optional_data):
    subtabs = st.tabs(["Confusion Matrix", "Regression Backtest", "Sanity & Sensitivity"])
    with subtabs[0]:
        render_confusion_matrix(optional_data.get("cls_metrics"))
    with subtabs[1]:
        render_regression_backtest(optional_data.get("ind_metrics"), optional_data.get("occ_metrics"))
    with subtabs[2]:
        render_sanity_sensitivity(optional_data.get("sanity"), optional_data.get("weights"))


def render_model_explanation_tab():
    st.subheader("Model Explanation")
    st.markdown(
        """
        <div class='tab-note'>
        <b>What the app is doing</b><br><br>
        The app follows a conditional recommendation path:<br>
        <b>Education → Industry → Province → Occupation</b>.<br><br>
        It first filters candidates using the selected education and target industry, then scores provinces within that path, and finally surfaces related occupations for the best province.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### ML layer")
    st.markdown(
        "- The ML outputs provide the predicted wage and employment values used by the app.\n"
        "- The prescriptive layer does not retrain the model; it uses the pipeline outputs to rank relevant paths."
    )

    st.markdown("### Prescriptive layer")
    st.markdown(
        "- Demand signal is taken from labour-demand strength for the selected path.\n"
        "- Wage and employment are normalized and combined with supply context.\n"
        "- Provinces are ranked only within the selected education + industry path."
    )

    st.code(
        "Opportunity Score = 0.35 × demand signal + 0.35 × predicted wage + 0.20 × predicted employment + 0.10 × supply score"
    )


def main() -> None:
    st.set_page_config(page_title="Conditional Path Recommendations", layout="wide")
    inject_css()
    st.title("Conditional Path Recommendations")
    st.caption(f"Project root: {PATHS.project_root}")

    data = load_data()
    ind, occ, edu, neet = data["ind"], data["occ"], data["edu"], data["neet"]
    optional_data = load_optional_data()

    tabs = st.tabs([
        "Prescriptive Recommendations",
        "Province Outlook Forecast",
        "Model Metrics",
        "Model Explanation",
    ])

    with tabs[0]:
        render_prescriptive_tab(ind, occ, edu, neet)
    with tabs[1]:
        render_province_outlook_tab(optional_data, ind, occ)
    with tabs[2]:
        render_model_metrics_tab(optional_data)
    with tabs[3]:
        render_model_explanation_tab()


if __name__ == "__main__":
    main()
