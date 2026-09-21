import os
import sys
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Force UTF-8 encoding on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src.data_engine.loader import DataLoader
from src.data_engine.profiler import DataProfiler
from src.data_engine.cleaner import DataCleaner
from src.data_engine.validator import DataValidator
from src.data_engine.eda import EDAEngine
from src.data_engine.statistical_analyzer import StatisticalAnalyzer
from src.data_engine.sql_engine import SQLEngine
from src.data_engine.viz_engine import VisualizationEngine
from src.data_engine.kpi_engine import KPIEngine
from src.data_engine.ml_engine import MLEngine
from src.data_engine.forecasting_engine import ForecastingEngine
from src.data_engine.duckdb_accelerator import DuckDBAccelerator
from src.agent_core.text_to_sql import TextToSQLAssistant
from src.agent_core.insight_generator import InsightGenerator
from src.agent_core.recommendation_engine import RecommendationEngine
from src.agent_core.report_builder import ReportBuilder
from src.exports.exporter import DataExporter
from src.agent_core.rag_chatbot import build_rag_chatbot_cached
from src.agent_core.domain_mapper import DomainMapper
from src.exports.chat_widget import build_floating_widget_html
import streamlit.components.v1 as components


st.set_page_config(
    page_title="AI Data Analyst Platform 11.0",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Executive Theme
st.markdown("""
<style>
    .stApp { background-color: #0f172a; color: #f8fafc; }
    .kpi-card {
        background: rgba(30, 41, 59, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 20px;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.3);
    }
    .kpi-title { font-size: 0.75rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; color: #94a3b8; }
    .kpi-value { font-size: 1.85rem; font-weight: 800; color: #ffffff; margin-top: 6px; }
    .kpi-subtitle { font-size: 0.75rem; color: #38bdf8; margin-top: 4px; }
    .clean-badge {
        background-color: rgba(16, 185, 129, 0.15);
        color: #10b981;
        border: 1px solid rgba(16, 185, 129, 0.3);
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

if "custom_recommendations" not in st.session_state:
    st.session_state["custom_recommendations"] = []

# RAG Chatbot session state
if "rag_messages" not in st.session_state:
    st.session_state["rag_messages"] = []
if "rag_chatbot" not in st.session_state:
    st.session_state["rag_chatbot"] = None
if "rag_dataset_hash" not in st.session_state:
    st.session_state["rag_dataset_hash"] = None

# Floating widget pending query (from JS postMessage bridge)
if "widget_pending_query" not in st.session_state:
    st.session_state["widget_pending_query"] = None
if "domain_info" not in st.session_state:
    st.session_state["domain_info"] = None


# -------------------------------------------------------------
# SIDEBAR — DATASET INGESTION & SLICERS
# -------------------------------------------------------------
st.sidebar.title("📊 Enterprise BI Platform")
uploaded_file = st.sidebar.file_uploader(
    "Upload Dataset (CSV, XLSX, Parquet, JSON)",
    type=["csv", "xlsx", "xls", "parquet", "json"]
)

sample_choice = st.sidebar.selectbox(
    "Or Select Sample Dataset:",
    ["-- Select Sample --", "Sample E-Commerce Sales"]
)

df_raw = None
dataset_name = "dataset"

if uploaded_file is not None:
    dataset_name = os.path.splitext(uploaded_file.name)[0]
    df_raw, metadata = DataLoader.load_file(uploaded_file, filename=uploaded_file.name)
elif sample_choice == "Sample E-Commerce Sales":
    sample_path = "sample_data/sample_ecommerce.csv"
    if os.path.exists(sample_path):
        dataset_name = "sample_ecommerce"
        df_raw, metadata = DataLoader.load_file(sample_path)

if df_raw is None:
    st.info("👋 **Welcome to the Enterprise AI Data Analyst Platform!** Upload a dataset or select a sample dataset in the sidebar to launch automated end-to-end analysis.")
    st.stop()

# -------------------------------------------------------------
# DATA GOVERNANCE & CLEANING
# -------------------------------------------------------------
st.sidebar.markdown("---")
st.sidebar.header("🧹 Data Governance Settings")
remove_dups = st.sidebar.checkbox("Deduplicate Rows", value=True)
trim_ws = st.sidebar.checkbox("Trim Text Whitespace", value=True)
std_nulls = st.sidebar.checkbox("Standardize String Nulls", value=True)
impute_num_strat = st.sidebar.selectbox("Numerical Imputation Strategy", ["median", "mean", "none"])

cleaner = DataCleaner(df_raw, metadata)
df_clean, trans_log = cleaner.run_full_cleaning_pipeline(
    remove_duplicates=remove_dups,
    trim_whitespace=trim_ws,
    standardize_nulls=std_nulls,
    impute_missing_numerical=impute_num_strat
)

# Profile AFTER cleaning so numeric-string conversions are reflected
profiling = DataProfiler.profile_dataset(df_clean, metadata)
dq_score = profiling["data_quality_score"]
validation = DataValidator.validate_cleaning(df_raw, df_clean, trans_log)
eda_res = EDAEngine.run_eda(df_clean, metadata)
kpi_res = KPIEngine.calculate_kpis(df_clean, metadata)
stat_res = StatisticalAnalyzer.run_tests(df_clean, metadata)

# Domain Intelligence (cached per dataset)
if st.session_state["domain_info"] is None or st.session_state.get("_last_dataset") != dataset_name:
    st.session_state["domain_info"] = DomainMapper.infer_domain(metadata)
    st.session_state["_last_dataset"] = dataset_name
domain_info = st.session_state["domain_info"]

cat_cols = metadata["column_buckets"]["categorical_columns"]
num_cols = metadata["column_buckets"]["numerical_columns"]
dt_cols = metadata["column_buckets"]["datetime_columns"]

# Global Slicers
st.sidebar.markdown("---")
st.sidebar.header("🎛️ Dashboard Slicers")
df_filtered = df_clean.copy()
if cat_cols:
    filter_cat = cat_cols[0]
    unique_cats = list(df_clean[filter_cat].dropna().unique())
    selected_cats = st.sidebar.multiselect(
        f"Filter by {filter_cat.replace('_', ' ').title()}:",
        options=unique_cats,
        default=unique_cats
    )
    if selected_cats:
        df_filtered = df_filtered[df_filtered[filter_cat].isin(selected_cats)]



# Header
st.markdown(f"""
<div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px;">
    <div>
        <h1 style="margin: 0; font-weight: 800; font-size: 2.2rem; color: #ffffff;">Enterprise AI Data Analyst Platform 11.0</h1>
        <p style="margin: 4px 0 0 0; color: #94a3b8; font-size: 0.9rem;">
            Dataset: <b style="color: #38bdf8;">{dataset_name}</b> &nbsp;|&nbsp;
            Filtered Rows: <b style="color: #34d399;">{len(df_filtered):,}</b> of {len(df_clean):,} &nbsp;|&nbsp;
            Domain: <b style="color: #f472b6;">{domain_info['icon']} {domain_info['domain']}</b>
            <span style="font-size:0.72rem;color:#64748b;margin-left:4px;">({domain_info['confidence']} confidence)</span>
        </p>
    </div>
    <div style="display:flex;gap:10px;align-items:center;">
        <span class="clean-badge">⭐ Quality Score: {dq_score}/100</span>
    </div>
</div>
""", unsafe_allow_html=True)

# -------------------------------------------------------------
# WORKSPACE MULTI-TAB NAVIGATION
# -------------------------------------------------------------
tab_dash, tab_custom_viz, tab_ai_sql, tab_recs, tab_ml_forecast, tab_export, tab_chatbot = st.tabs([
    "🖥️ Executive Dashboard",
    "🎨 Custom Chart Studio",
    "💬 AI Text-to-SQL & Query Lab",
    "🎯 Insights & Custom Recommendations",
    "📉 Machine Learning & Forecasting",
    "📥 Export Center",
    "🤖 AI Data Chatbot (RAG)"
])


# -------------------------------------------------------------
# TAB 1: EXECUTIVE BOARDROOM DASHBOARD
# -------------------------------------------------------------
with tab_dash:
    k1, k2, k3, k4 = st.columns(4)
    if "Sales & Revenue" in kpi_res:
        sales = kpi_res["Sales & Revenue"]
        k1.markdown(f"""<div class="kpi-card"><div class="kpi-title">Total Revenue</div><div class="kpi-value">{sales.get('Total Revenue', 'N/A')}</div><div class="kpi-subtitle">▲ Verified Aggregate Spend</div></div>""", unsafe_allow_html=True)
        k2.markdown(f"""<div class="kpi-card"><div class="kpi-title">Average Order Value</div><div class="kpi-value">{sales.get('Average Order Value (AOV)', 'N/A')}</div><div class="kpi-subtitle">🛒 Mean Spend per Order</div></div>""", unsafe_allow_html=True)
        k3.markdown(f"""<div class="kpi-card"><div class="kpi-title">Transaction Volume</div><div class="kpi-value">{len(df_filtered):,}</div><div class="kpi-subtitle">📦 Filtered Orders</div></div>""", unsafe_allow_html=True)
    else:
        k1.markdown(f"""<div class="kpi-card"><div class="kpi-title">Total Records</div><div class="kpi-value">{len(df_filtered):,}</div><div class="kpi-subtitle">📦 Row Count</div></div>""", unsafe_allow_html=True)
        k2.markdown(f"""<div class="kpi-card"><div class="kpi-title">Total Columns</div><div class="kpi-value">{len(df_filtered.columns)}</div><div class="kpi-subtitle">📊 Feature Count</div></div>""", unsafe_allow_html=True)
        k3.markdown(f"""<div class="kpi-card"><div class="kpi-title">Data Quality</div><div class="kpi-value">{dq_score}/100</div><div class="kpi-subtitle">⭐ Quality Index</div></div>""", unsafe_allow_html=True)

    if "Operational Quality" in kpi_res:
        ops = kpi_res["Operational Quality"]
        k4.markdown(f"""<div class="kpi-card"><div class="kpi-title">Return / Exception Rate</div><div class="kpi-value">{ops.get('Return / Exception Rate', '0%')}</div><div class="kpi-subtitle">⚠️ {ops.get('Return / Exception Count', 0)} Exceptions</div></div>""", unsafe_allow_html=True)
    else:
        k4.markdown(f"""<div class="kpi-card"><div class="kpi-title">Status</div><div class="kpi-value">Cleaned</div><div class="kpi-subtitle">✅ Audit Passed</div></div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    c_left, c_right = st.columns(2)
    with c_left:
        st.subheader("📈 Temporal Trend Analysis")
        if dt_cols and num_cols:
            dt_col = dt_cols[0]
            num_col = num_cols[0]
            temp = df_filtered.copy()
            temp[dt_col] = pd.to_datetime(temp[dt_col], errors="coerce")
            trend_df = temp.dropna(subset=[dt_col]).groupby(dt_col)[num_col].sum().reset_index().sort_values(dt_col)
            fig_trend = px.area(trend_df, x=dt_col, y=num_col, template="plotly_dark", color_discrete_sequence=["#38bdf8"])
            fig_trend.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_trend, use_container_width=True)
        else:
            st.info("No datetime column available.")

        st.subheader("📊 Segment Spend Ranking")
        if cat_cols and num_cols:
            cat_df = df_filtered.groupby(cat_cols[0])[num_cols[0]].sum().reset_index().sort_values(num_cols[0], ascending=True)
            fig_bar = px.bar(cat_df, x=num_cols[0], y=cat_cols[0], orientation="h", color=num_cols[0], color_continuous_scale="Blues", template="plotly_dark")
            fig_bar.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_bar, use_container_width=True)

    with c_right:
        st.subheader("🍩 Category Distribution Share")
        if cat_cols:
            counts = df_filtered[cat_cols[0]].value_counts().reset_index()
            counts.columns = [cat_cols[0], "count"]
            fig_pie = px.pie(counts, names=cat_cols[0], values="count", hole=0.45, template="plotly_dark")
            fig_pie.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_pie, use_container_width=True)

        st.subheader("🎯 Measure Scatter Analysis")
        if len(num_cols) >= 2:
            fig_scatter = px.scatter(df_filtered, x=num_cols[0], y=num_cols[1], color=cat_cols[0] if cat_cols else None, template="plotly_dark")
            fig_scatter.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_scatter, use_container_width=True)

    st.subheader("📋 Dataset Preview Table")
    st.dataframe(df_filtered.head(10), use_container_width=True)

    # ── Domain Intelligence Panel ────────────────────────────────────────────
    st.markdown("---")
    st.subheader(f"{domain_info['icon']} Domain Intelligence — {domain_info['domain']}")

    _d1, _d2 = st.columns([1, 2])
    with _d1:
        st.markdown(f"""
<div class="kpi-card">
  <div class="kpi-title">Inferred Business Domain</div>
  <div class="kpi-value" style="font-size:1.4rem;">{domain_info['icon']} {domain_info['domain']}</div>
  <div class="kpi-subtitle">Confidence: {domain_info['confidence']} &nbsp;·&nbsp; Signals: {', '.join(domain_info['domain_signals_matched'][:5]) or 'General'}</div>
</div>
""", unsafe_allow_html=True)
        if domain_info.get("use_cases"):
            st.markdown("**Business Use Cases:**")
            for uc in domain_info["use_cases"][:4]:
                st.markdown(f"  ✅ {uc}")

    with _d2:
        st.markdown("**🎯 Recommended ML Goals & Analytical Objectives:**")
        _goal_cols = st.columns(min(2, len(domain_info["ml_goals"])))
        for _gi, _goal in enumerate(domain_info["ml_goals"][:4]):
            _pri_colors = {"HIGH": "#f59e0b", "CRITICAL": "#ef4444", "MEDIUM": "#38bdf8", "LOW": "#64748b"}
            _pri_col = _pri_colors.get(_goal.get("priority", "MEDIUM"), "#64748b")
            _goal_cols[_gi % 2].markdown(f"""
<div class="kpi-card" style="margin-bottom:10px;">
  <div style="display:flex;justify-content:space-between;margin-bottom:4px;">
    <span style="font-size:0.72rem;font-weight:700;color:{_pri_col};">● {_goal.get('priority','MEDIUM')}</span>
    <span style="font-size:0.68rem;color:#64748b;">{_goal.get('model_type','')}</span>
  </div>
  <div style="font-weight:700;color:#f8fafc;font-size:0.85rem;margin-bottom:4px;">{_goal.get('title','')}</div>
  <div style="color:#94a3b8;font-size:0.72rem;margin-bottom:6px;">{_goal.get('description','')}</div>
  <div style="color:#22c55e;font-size:0.68rem;">💡 {_goal.get('business_value','')}</div>
</div>
""", unsafe_allow_html=True)

    with st.expander("📊 Recommended Analyses for this Domain", expanded=False):
        for _ra in domain_info.get("recommended_analyses", []):
            st.markdown(f"  📌 {_ra}")

    # ── Data Quality & Outlier Report ────────────────────────────────────────
    st.markdown("---")
    with st.expander("🔬 Data Quality Audit & Outlier Detection Report", expanded=False):
        _audit_col1, _audit_col2 = st.columns(2)
        with _audit_col1:
            st.markdown("**Cleaning Transformations Applied:**")
            if trans_log:
                for _t in trans_log:
                    _icon = "✅" if _t["records_affected"] > 0 else "⬜"
                    st.markdown(f"{_icon} **Step {_t['step']}:** {_t['issue']} — *{_t['records_affected']} records affected*")
                    if _t.get("details"):
                        st.caption(f"Details: {_t['details'][:200]}")
            else:
                st.success("✅ No cleaning actions required — dataset is already clean.")
        with _audit_col2:
            st.markdown("**Column Type Resolution:**")
            _type_data = []
            for col, info in metadata.get("column_analysis", {}).items():
                _type_data.append({
                    "Column": col,
                    "Type": info.get("inferred_type", "?"),
                    "Missing %": f"{info.get('missing_pct', 0)}%",
                    "Unique": info.get("unique_count", 0),
                })
            import pandas as _pd2
            st.dataframe(_pd2.DataFrame(_type_data), use_container_width=True, height=200)



# -------------------------------------------------------------
# TAB 2: CUSTOM CHART STUDIO
# -------------------------------------------------------------
with tab_custom_viz:
    st.subheader("🎨 Custom Plotly Graph Studio")
    st.caption("Select your own columns, aggregations, chart types, and color groupings to build custom figures.")

    v1, v2, v3, v4 = st.columns(4)
    chart_type = v1.selectbox("Chart Type:", ["Bar Chart", "Line Chart", "Scatter Plot", "Box Plot", "Histogram", "Donut Chart", "Heatmap"])
    x_axis = v2.selectbox("Select X-Axis / Category Column:", options=list(df_filtered.columns))
    y_axis = v3.selectbox("Select Y-Axis / Metric Column:", options=["-- None / Count --"] + num_cols)
    color_by = v4.selectbox("Color / Group By:", options=["-- None --"] + cat_cols)
    agg_func = st.selectbox("Aggregation Function:", ["SUM", "AVG / MEAN", "COUNT", "MAX", "MIN"])

    try:
        fig_custom = None
        if chart_type == "Bar Chart":
            if y_axis != "-- None / Count --":
                grp = df_filtered.groupby(x_axis)[y_axis].sum().reset_index() if agg_func == "SUM" else df_filtered.groupby(x_axis)[y_axis].mean().reset_index()
                fig_custom = px.bar(grp, x=x_axis, y=y_axis, color=x_axis, template="plotly_dark", title=f"{agg_func} of {y_axis} by {x_axis}")
            else:
                counts = df_filtered[x_axis].value_counts().reset_index()
                counts.columns = [x_axis, "count"]
                fig_custom = px.bar(counts, x=x_axis, y="count", color=x_axis, template="plotly_dark", title=f"Record Count by {x_axis}")

        elif chart_type == "Line Chart" and y_axis != "-- None / Count --":
            grp = df_filtered.groupby(x_axis)[y_axis].mean().reset_index()
            fig_custom = px.line(grp, x=x_axis, y=y_axis, markers=True, template="plotly_dark", title=f"{y_axis} over {x_axis}")

        elif chart_type == "Scatter Plot" and y_axis != "-- None / Count --":
            fig_custom = px.scatter(df_filtered, x=x_axis, y=y_axis, color=color_by if color_by != "-- None --" else None, template="plotly_dark", title=f"Scatter: {x_axis} vs {y_axis}")

        elif chart_type == "Box Plot" and y_axis != "-- None / Count --":
            fig_custom = px.box(df_filtered, x=x_axis, y=y_axis, color=color_by if color_by != "-- None --" else None, template="plotly_dark", title=f"Box Plot: {y_axis} across {x_axis}")

        elif chart_type == "Histogram":
            fig_custom = px.histogram(df_filtered, x=x_axis, color=color_by if color_by != "-- None --" else None, template="plotly_dark", title=f"Histogram: {x_axis}")

        elif chart_type == "Donut Chart":
            counts = df_filtered[x_axis].value_counts().reset_index()
            counts.columns = [x_axis, "count"]
            fig_custom = px.pie(counts, names=x_axis, values="count", hole=0.5, template="plotly_dark", title=f"Share of {x_axis}")

        elif chart_type == "Heatmap" and len(num_cols) >= 2:
            fig_custom = px.imshow(df_filtered[num_cols].corr().round(2), text_auto=True, color_continuous_scale="RdBu_r", template="plotly_dark", title="Pearson Correlation Heatmap")

        if fig_custom:
            fig_custom.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_custom, use_container_width=True)
    except Exception as e:
        st.error(f"Error rendering chart: {e}")

# -------------------------------------------------------------
# TAB 3: CONVERSATIONAL TEXT-TO-SQL & SQL LAB
# -------------------------------------------------------------
with tab_ai_sql:
    st.subheader("💬 Natural Language Text-to-SQL Assistant")
    st.caption("Ask questions in plain English (e.g. 'Show top 5 product categories by spend') to automatically generate & execute SQL queries.")

    nl_question = st.text_input("Ask a question about the dataset:", "Show top 5 product categories by total spend")
    if st.button("🤖 Ask AI Assistant & Run SQL", type="primary"):
        res_df, q_meta = TextToSQLAssistant.ask_data(nl_question, df_filtered, metadata, table_name=dataset_name)
        st.markdown(f"**Generated SQL Query**:")
        st.code(q_meta["generated_sql"], language="sql")
        
        if res_df is not None and len(res_df) > 0:
            st.success(f"Query returned {len(res_df)} rows.")
            st.dataframe(res_df, use_container_width=True)
        else:
            st.warning("No rows returned or error executing generated query.")

    st.markdown("---")
    st.subheader("💾 Manual SQL Console (Read-Only Guardrails Active)")
    sql_engine = SQLEngine(df_clean, table_name=dataset_name)
    rec_queries = sql_engine.generate_recommended_queries(metadata)

    selected_q = st.selectbox("Recommended SQL Queries:", ["-- Custom --"] + list(rec_queries.keys()))
    default_q = rec_queries[selected_q] if selected_q != "-- Custom --" else f"SELECT * FROM {dataset_name} LIMIT 10;"

    custom_sql = st.text_area("SQL Editor:", value=default_q, height=130)
    if st.button("▶️ Execute SQL Code"):
        res_df, q_meta = sql_engine.execute_query(custom_sql)
        if q_meta["success"]:
            st.success(f"Execution successful ({q_meta['rows_returned']} rows).")
            st.dataframe(res_df, use_container_width=True)
        else:
            st.error(q_meta["error"])
    sql_engine.close()

# -------------------------------------------------------------
# TAB 4: INSIGHTS & CUSTOM RECOMMENDATION BUILDER
# -------------------------------------------------------------
with tab_recs:
    st.subheader("🎯 Add Custom Business Recommendations")
    with st.form("custom_rec_form"):
        rec_title = st.text_input("Problem Subject:", "Expand Marketing Reach to Tier-2 Cities")
        rec_prob = st.text_area("What is the Problem?", "Describe problem...")
        rec_solve = st.text_area("What to Solve?", "Target benchmark...")
        rec_how = st.text_area("How to Solve?", "Resolution strategy...")
        rec_impact = st.text_input("Expected Impact:", "15% growth in Q3")
        rec_priority = st.selectbox("Priority:", ["HIGH PRIORITY", "MEDIUM PRIORITY", "LOW PRIORITY"])
        rec_next = st.text_input("Immediate Next Step:", "Launch pilot campaign")
        
        submitted = st.form_submit_button("➕ Add Custom Recommendation")
        if submitted and rec_title:
            st.session_state["custom_recommendations"].append({
                "priority": rec_priority,
                "problem": rec_title,
                "what_is_the_problem": rec_prob,
                "what_to_solve": rec_solve,
                "how_to_solve": rec_how,
                "expected_impact": rec_impact,
                "next_step": rec_next
            })
            st.success(f"Added Custom Recommendation: '{rec_title}'!")

    st.markdown("---")
    st.subheader("📋 Complete Strategic Recommendations Matrix (AI + Custom)")
    ai_recs = RecommendationEngine.generate_recommendations(profiling, validation, eda_res, stat_res, kpi_res)
    all_recs = ai_recs + st.session_state["custom_recommendations"]

    for r in all_recs:
        st.markdown(f"#### [{r['priority']}] {r['problem']}")
        st.write(f"**🚨 WHAT IS THE PROBLEM**: {r.get('what_is_the_problem')}")
        st.write(f"**🎯 WHAT TO SOLVE**: {r.get('what_to_solve')}")
        st.write(f"**🛠️ HOW TO SOLVE**:\n{r.get('how_to_solve')}")
        st.write(f"**💡 EXPECTED IMPACT**: {r.get('expected_impact')}")
        st.write(f"**➡️ IMMEDIATE NEXT STEP**: `{r.get('next_step')}`")
        st.markdown("---")

# -------------------------------------------------------------
# TAB 5: MACHINE LEARNING & FORECASTING
# -------------------------------------------------------------
with tab_ml_forecast:
    st.subheader("📉 Time-Series Trend Forecasting")
    if dt_cols and num_cols:
        f_date = st.selectbox("Select Date Column:", dt_cols)
        f_val = st.selectbox("Select Target Metric to Forecast:", num_cols)
        f_periods = st.slider("Periods to Forecast Ahead:", min_value=3, max_value=12, value=6)

        forecast_res = ForecastingEngine.forecast_trend(df_filtered, date_col=f_date, value_col=f_val, periods_ahead=f_periods)
        if "forecast_data" in forecast_res:
            st.info(f"**Forecast Trajectory**: {forecast_res['interpretation']}")
            
            hist_df = pd.DataFrame(forecast_res["historical_data"])
            fut_df = pd.DataFrame(forecast_res["forecast_data"])

            fig_f = go.Figure()
            fig_f.add_trace(go.Scatter(x=hist_df["date"], y=hist_df["actual"], mode="lines+markers", name="Historical Actual", line=dict(color="#38bdf8", width=3)))
            fig_f.add_trace(go.Scatter(x=fut_df["date"], y=fut_df["forecast"], mode="lines+markers", name="Forecast Projection", line=dict(color="#f472b6", width=3, dash="dash")))
            fig_f.add_trace(go.Scatter(x=fut_df["date"], y=fut_df["ci_upper"], mode="lines", name="95% Upper Bound", line=dict(width=0), showlegend=False))
            fig_f.add_trace(go.Scatter(x=fut_df["date"], y=fut_df["ci_lower"], mode="lines", name="95% Lower Bound", fill="tonexty", fillcolor="rgba(244, 114, 182, 0.15)", line=dict(width=0), showlegend=False))
            fig_f.update_layout(template="plotly_dark", title=f"Trend Projection ({forecast_res['granularity']})", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_f, use_container_width=True)

    st.markdown("---")
    st.subheader("🤖 AutoML & SHAP Explainable AI (XAI)")
    selected_target = st.selectbox("Select ML Target Feature:", num_cols + cat_cols)
    if selected_target:
        ml_res = MLEngine.train_baseline_model(df_clean, target_col=selected_target)
        if "metrics" in ml_res:
            st.write(f"**Model Performance**: {ml_res['interpretation']}")
            m_cols = st.columns(len(ml_res["metrics"]))
            for idx, (mk, mv) in enumerate(ml_res["metrics"].items()):
                m_cols[idx].metric(mk.upper(), str(mv))

            # Explainable AI (SHAP-style)
            attributions = ForecastingEngine.explain_feature_attribution(ml_res)
            if attributions:
                st.markdown("#### 🔬 SHAP Feature Attribution & Model Decision Boundary")
                st.dataframe(pd.DataFrame(attributions), use_container_width=True)

# -------------------------------------------------------------
# TAB 6: EXPORT CENTER
# -------------------------------------------------------------
with tab_export:
    st.subheader("📥 Download Cleaned Datasets & Executive Briefings")

    insights = InsightGenerator.generate_insights(metadata, profiling, validation, eda_res, stat_res, kpi_res, None)
    report_md = ReportBuilder.build_markdown_report(
        dataset_name=dataset_name,
        metadata=metadata,
        profiling=profiling,
        validation=validation,
        transformation_log=trans_log,
        eda_results=eda_res,
        stat_results=stat_res,
        kpi_results=kpi_res,
        chart_recs=[],
        insights=insights,
        recommendations=all_recs
    )

    c1, c2, c3 = st.columns(3)
    c1.download_button("📄 Cleaned CSV Dataset", df_clean.to_csv(index=False).encode("utf-8"), f"{dataset_name}_cleaned.csv", "text/csv")
    c2.download_button("📝 Executive Markdown Report", report_md.encode("utf-8"), f"{dataset_name}_report.md", "text/markdown")

    # Printable HTML/PDF Briefing
    printable_path = f"output_report/printable_{dataset_name}.html"
    os.makedirs("output_report", exist_ok=True)
    from src.exports.pdf_exporter import PrintablePDFExporter
    PrintablePDFExporter.generate_printable_report(dataset_name, metadata, profiling, report_md, printable_path)

    with open(printable_path, "r", encoding="utf-8") as f:
        html_bytes = f.read().encode("utf-8")
    c3.download_button("🖨️ Printable Executive PDF Briefing (HTML)", html_bytes, f"{dataset_name}_printable_briefing.html", "text/html")

# -------------------------------------------------------------
# TAB 7: 🤖 AI DATA CHATBOT (RAG)
# Retrieval-Augmented Generation chatbot grounded in dataset analysis
# -------------------------------------------------------------
with tab_chatbot:
    import hashlib as _hashlib

    # ── Header ──────────────────────────────────────────────────────────────
    st.markdown("""
<div style="display:flex;align-items:center;gap:14px;margin-bottom:8px;">
    <div style="font-size:3rem;">🤖</div>
    <div>
        <h2 style="margin:0;font-weight:800;color:#ffffff;">AI Data Analyst Chatbot</h2>
        <p style="margin:0;color:#94a3b8;font-size:0.88rem;">
            Powered by RAG (Retrieval-Augmented Generation) — answers grounded in <b style="color:#38bdf8;">your actual dataset analysis</b>
        </p>
    </div>
</div>
""", unsafe_allow_html=True)

    # ── Sidebar: Gemini API Key (for this tab) ───────────────────────────────
    st.sidebar.markdown("---")
    st.sidebar.header("🤖 AI Chatbot Settings")
    gemini_key_input = st.sidebar.text_input(
        "Gemini API Key (for AI Chatbot):",
        value=os.environ.get("GEMINI_API_KEY", os.environ.get("GOOGLE_API_KEY", "")),
        type="password",
        help="Add your Gemini API key for AI-powered answers. Get one free at makersuite.google.com"
    )
    chatbot_model_note = "🟢 Gemini AI Active" if gemini_key_input else "🟡 Keyword Fallback Mode (No API Key)"
    st.sidebar.caption(chatbot_model_note)

    # ── Status Banner ────────────────────────────────────────────────────────
    col_status, col_clear = st.columns([4, 1])
    with col_status:
        if gemini_key_input:
            st.success("🟢 **Gemini AI Active** — Full RAG-powered natural language answers enabled.")
        else:
            st.warning(
                "🟡 **Keyword Fallback Mode** — Add your Gemini API Key in the sidebar for AI-powered answers. "
                "Get a free key at [Google AI Studio](https://makersuite.google.com/app/apikey)."
            )
    with col_clear:
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state["rag_messages"] = []
            st.session_state["rag_chatbot"] = None
            st.session_state["rag_dataset_hash"] = None
            st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Build / Cache RAG Knowledge Base ────────────────────────────────────
    # Compute a hash to detect dataset changes and rebuild vector store only when needed
    _dataset_hash = _hashlib.md5(
        f"{dataset_name}_{len(df_clean)}_{gemini_key_input[:8] if gemini_key_input else 'nk'}".encode()
    ).hexdigest()

    if st.session_state["rag_dataset_hash"] != _dataset_hash or st.session_state["rag_chatbot"] is None:
        with st.spinner("🔧 Building RAG Knowledge Base from your dataset analysis... (one-time per dataset)"):
            # Generate full analysis artifacts for RAG
            _rag_insights = InsightGenerator.generate_insights(
                metadata, profiling, validation, eda_res, stat_res, kpi_res, None
            )
            _rag_recs = RecommendationEngine.generate_recommendations(
                profiling, validation, eda_res, stat_res, kpi_res
            )
            _rag_recs_all = _rag_recs + st.session_state.get("custom_recommendations", [])
            _rag_report_md = ReportBuilder.build_markdown_report(
                dataset_name=dataset_name,
                metadata=metadata,
                profiling=profiling,
                validation=validation,
                transformation_log=trans_log,
                eda_results=eda_res,
                stat_results=stat_res,
                kpi_results=kpi_res,
                chart_recs=[],
                insights=_rag_insights,
                recommendations=_rag_recs_all
            )
            _df_sample_csv = df_clean.head(20).to_csv(index=False)

            st.session_state["rag_chatbot"] = build_rag_chatbot_cached(
                dataset_name=dataset_name,
                metadata=metadata,
                profiling=profiling,
                validation=validation,
                eda_res=eda_res,
                stat_res=stat_res,
                kpi_res=kpi_res,
                insights=_rag_insights,
                recommendations=_rag_recs_all,
                report_md=_rag_report_md,
                df_sample_csv=_df_sample_csv,
                gemini_api_key=gemini_key_input if gemini_key_input else None,
            )
            st.session_state["rag_dataset_hash"] = _dataset_hash
        st.success(f"✅ RAG Knowledge Base ready — indexed **{dataset_name}** analysis.")

    _chatbot = st.session_state["rag_chatbot"]

    # ── Suggested Questions ──────────────────────────────────────────────────
    _suggested = _chatbot.get_suggested_questions(metadata)
    with st.expander("💡 **Suggested Questions** — Click to ask", expanded=True):
        _cols = st.columns(2)
        for _i, _q in enumerate(_suggested):
            if _cols[_i % 2].button(f"❓ {_q}", key=f"suggest_{_i}", use_container_width=True):
                st.session_state["rag_messages"].append({"role": "user", "content": _q})
                with st.spinner("🧠 Analyzing..."):
                    _ans, _srcs = _chatbot.chat(_q)
                st.session_state["rag_messages"].append({
                    "role": "assistant",
                    "content": _ans,
                    "sources": _srcs
                })
                st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Chat History Display ─────────────────────────────────────────────────
    _chat_container = st.container()
    with _chat_container:
        if not st.session_state["rag_messages"]:
            st.markdown("""
<div style="text-align:center;padding:40px;color:#64748b;">
    <div style="font-size:3rem;margin-bottom:12px;">💬</div>
    <p style="font-size:1.1rem;font-weight:600;color:#94a3b8;">Ask anything about your dataset</p>
    <p style="font-size:0.85rem;">Questions like "What are the main quality issues?", "Which products drive revenue?", "What are the top recommendations?"</p>
</div>
""", unsafe_allow_html=True)
        else:
            for _msg in st.session_state["rag_messages"]:
                if _msg["role"] == "user":
                    with st.chat_message("user", avatar="👤"):
                        st.markdown(_msg["content"])
                else:
                    with st.chat_message("assistant", avatar="🤖"):
                        st.markdown(_msg["content"])
                        # Source attribution
                        _sources = _msg.get("sources", [])
                        if _sources:
                            _source_labels = {
                                "dataset_overview": "📋 Dataset Overview",
                                "data_quality": "🔍 Data Quality Profile",
                                "cleaning_validation": "🧹 Cleaning & Validation",
                                "eda": "📊 EDA Analysis",
                                "statistical_analysis": "📐 Statistical Tests",
                                "kpis": "📈 KPI Metrics",
                                "insights": "💡 AI Insights",
                                "recommendations": "🎯 Recommendations",
                                "full_report": "📄 Full Report",
                                "data_sample": "🗂️ Data Sample",
                                "dataset_analysis": "📊 Dataset Analysis",
                            }
                            _readable_srcs = [_source_labels.get(s, s) for s in _sources]
                            st.caption(f"📚 Sources: {' · '.join(_readable_srcs)}")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Chat Input ───────────────────────────────────────────────────────────
    _user_input = st.chat_input(
        f"Ask anything about '{dataset_name}'... (e.g. 'What are the top KPIs?', 'Why is data quality low?')"
    )

    if _user_input:
        st.session_state["rag_messages"].append({"role": "user", "content": _user_input})
        with st.spinner("🧠 Retrieving context and generating answer..."):
            _answer, _sources_list = _chatbot.chat(_user_input)
        st.session_state["rag_messages"].append({
            "role": "assistant",
            "content": _answer,
            "sources": _sources_list
        })
        st.rerun()

    # ── RAG Architecture Info ────────────────────────────────────────────────
    with st.expander("ℹ️ How the RAG Chatbot Works", expanded=False):
        st.markdown("""
**Retrieval-Augmented Generation (RAG) Architecture:**

```
Your Question
     │
     ▼
┌─────────────────────────────┐
│  Query Embedding             │  ← Google Generative AI Embeddings
│  (text-embedding-004)        │
└────────────┬────────────────┘
             │
             ▼
┌─────────────────────────────┐
│  ChromaDB Vector Search      │  ← In-memory vector store
│  MMR Retrieval (Top 5 docs) │     (rebuilt per dataset)
└────────────┬────────────────┘
             │ Relevant chunks from:
             │  • Data Quality Report   • EDA Results
             │  • KPI Analysis          • Statistical Tests
             │  • AI Insights           • Recommendations
             │  • Full 23-section Report
             ▼
┌─────────────────────────────┐
│  Gemini 2.5-Flash LLM        │  ← Answer generation
│  + Conversation Memory (8t) │     grounded in retrieved context
└─────────────────────────────┘
             │
             ▼
        Your Answer  (with source attribution)
```

**Fallback Chain:** Gemini chain → Direct Gemini API → BM25 Keyword Retriever → Deterministic text search

**Privacy:** Your data never leaves your machine unless you add a Gemini API key. 
All embedding and retrieval happens in-memory.
""")


# =============================================================================
# FLOATING RAG CHAT WIDGET — Persistent across ALL tabs
# =============================================================================
# This runs OUTSIDE all tab contexts so the bubble appears on every tab.
# The widget communicates via st.session_state + component value.

import hashlib as _hl

_widget_gemini_key = os.environ.get("GEMINI_API_KEY", os.environ.get("GOOGLE_API_KEY", ""))
_widget_hash = _hl.md5(
    f"{dataset_name}_{len(df_clean)}_{_widget_gemini_key[:8] if _widget_gemini_key else 'nk'}".encode()
).hexdigest()

# Build/cache the chatbot for the floating widget (reuses same cache key as Tab 7)
if st.session_state["rag_dataset_hash"] != _widget_hash or st.session_state["rag_chatbot"] is None:
    with st.spinner("🤖 Initializing AI Chatbot Knowledge Base…"):
        _w_insights = InsightGenerator.generate_insights(
            metadata, profiling, validation, eda_res, stat_res, kpi_res, None
        )
        _w_recs = RecommendationEngine.generate_recommendations(
            profiling, validation, eda_res, stat_res, kpi_res
        )
        _w_recs_all = _w_recs + st.session_state.get("custom_recommendations", [])
        _w_report_md = ReportBuilder.build_markdown_report(
            dataset_name=dataset_name,
            metadata=metadata,
            profiling=profiling,
            validation=validation,
            transformation_log=trans_log,
            eda_results=eda_res,
            stat_results=stat_res,
            kpi_results=kpi_res,
            chart_recs=[],
            insights=_w_insights,
            recommendations=_w_recs_all
        )
        st.session_state["rag_chatbot"] = build_rag_chatbot_cached(
            dataset_name=dataset_name,
            metadata=metadata,
            profiling=profiling,
            validation=validation,
            eda_res=eda_res,
            stat_res=stat_res,
            kpi_res=kpi_res,
            insights=_w_insights,
            recommendations=_w_recs_all,
            report_md=_w_report_md,
            df_sample_csv=df_clean.head(20).to_csv(index=False),
            gemini_api_key=_widget_gemini_key if _widget_gemini_key else None,
        )
        st.session_state["rag_dataset_hash"] = _widget_hash

_widget_chatbot = st.session_state["rag_chatbot"]

# Handle any pending widget query (submitted by previous render)
if st.session_state.get("widget_pending_query"):
    _pq = st.session_state["widget_pending_query"]
    st.session_state["widget_pending_query"] = None
    _w_ans, _w_srcs = _widget_chatbot.chat(_pq)
    st.session_state["rag_messages"].append({"role": "user", "content": _pq})
    st.session_state["rag_messages"].append({
        "role": "assistant",
        "content": _w_ans,
        "sources": _w_srcs
    })

# Render the floating widget HTML component
_suggested_qs = _widget_chatbot.get_suggested_questions(metadata)
_widget_html = build_floating_widget_html(
    messages=st.session_state["rag_messages"],
    suggested_questions=_suggested_qs,
    dataset_name=dataset_name,
    is_ai_active=bool(_widget_gemini_key),
    height=680,
)

# Inject widget via component; capture any postMessage value
_widget_value = components.html(_widget_html, height=680, scrolling=False)

# If user typed a query in the widget (postMessage value received by Streamlit)
if _widget_value and isinstance(_widget_value, dict) and _widget_value.get("action") == "chat":
    _incoming_query = _widget_value.get("query", "").strip()
    if _incoming_query:
        st.session_state["widget_pending_query"] = _incoming_query
        st.rerun()


