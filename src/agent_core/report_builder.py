import pandas as pd
from typing import Dict, Any, List, Optional

class ReportBuilder:
    """
    Standardized report compiler formatting complete 23-section executive Markdown & HTML reports.
    """

    @classmethod
    def build_markdown_report(
        cls,
        dataset_name: str,
        metadata: Dict[str, Any],
        profiling: Dict[str, Any],
        validation: Dict[str, Any],
        transformation_log: List[Dict[str, Any]],
        eda_results: Dict[str, Any],
        stat_results: List[Dict[str, Any]],
        kpi_results: Dict[str, Any],
        chart_recs: List[Dict[str, Any]],
        insights: List[Dict[str, Any]],
        recommendations: List[Dict[str, Any]],
        ml_results: Optional[Dict[str, Any]] = None
    ) -> str:
        
        dq_score = profiling.get("data_quality_score", 100.0)
        shape = metadata.get("shape", (0, 0))
        
        md = []

        # Header
        md.append(f"# Automated Comprehensive Data Analysis Report: {dataset_name}\n")
        md.append("> **Prepared by**: Expert AI Data Analyst Agent (Evidence-Driven System)")
        md.append(f"> **Data Quality Score**: `{dq_score} / 100` | **Rows**: `{shape[0]}` | **Columns**: `{shape[1]}`\n")
        md.append("---\n")

        # 1. Executive Summary
        md.append("## 1. Executive Summary\n")
        md.append(f"This report presents an end-to-end analytical evaluation of the **{dataset_name}** dataset. ")
        md.append(f"The dataset was automatically ingested, profiled, cleaned, validated, and analyzed across exploratory, statistical, and domain KPI dimensions. ")
        md.append(f"The initial Data Quality Score of **{dq_score}/100** was assigned based on measurable missingness, duplicate frequency, and structural formatting.\n")

        # 2. Dataset Overview
        md.append("## 2. Dataset Overview\n")
        md.append(f"- **File Type**: `{metadata.get('file_type', 'N/A')}`")
        md.append(f"- **Total Rows**: `{shape[0]}`")
        md.append(f"- **Total Columns**: `{shape[1]}`")
        md.append(f"- **Primary Key Candidates**: `{metadata['column_buckets'].get('primary_key_candidates', ['None'])}`")
        md.append(f"- **Numerical Columns**: `{metadata['column_buckets'].get('numerical_columns', [])}`")
        md.append(f"- **Categorical Columns**: `{metadata['column_buckets'].get('categorical_columns', [])}`")
        md.append(f"- **Datetime Columns**: `{metadata['column_buckets'].get('datetime_columns', [])}`")
        if metadata.get("pii_detected"):
            md.append(f"- **PII Detected & Masked**: `{metadata.get('pii_detected')}`")
        md.append("\n")

        # 3. Data Quality & Cleaning Validation
        md.append("## 3. Data Quality & Audit Trail\n")
        md.append("### Baseline vs Validated Dataset Metrics\n")
        
        summary_tbl = validation.get("summary_table", {})
        if summary_tbl:
            md.append("| Metric | Before Cleaning | After Cleaning | Delta Change |")
            md.append("| :--- | :--- | :--- | :--- |")
            for i in range(len(summary_tbl.get("metric", []))):
                m = summary_tbl["metric"][i]
                b = summary_tbl["before_cleaning"][i]
                a = summary_tbl["after_cleaning"][i]
                d = summary_tbl["delta_change"][i]
                md.append(f"| **{m}** | {b} | {a} | {d} |")
            md.append("\n")

        if transformation_log:
            md.append("### Transformation Log\n")
            for log in transformation_log:
                md.append(f"- **Step {log.get('step')}**: `{log.get('issue')}`")
                md.append(f"  - *Rationale*: {log.get('rationale')}")
                md.append(f"  - *Action*: {log.get('action')} (Records affected: `{log.get('records_affected')}`)\n")

        # 4. Business KPI Analysis
        md.append("## 4. Key Performance Indicators (KPIs)\n")
        if kpi_results:
            for category, kpi_dict in kpi_results.items():
                md.append(f"### {category}\n")
                for k, v in kpi_dict.items():
                    md.append(f"- **{k}**: `{v}`")
                md.append("\n")
        else:
            md.append("No domain KPIs could be inferred from column names.\n")

        # 5. Exploratory Data Analysis
        md.append("## 5. Exploratory Data Analysis (EDA)\n")
        corrs = eda_results.get("correlations", {}).get("top_correlated_pairs", [])
        if corrs:
            md.append("### Top Pairwise Correlations\n")
            md.append("| Variable 1 | Variable 2 | Pearson Correlation | Spearman Correlation | Direction | Strength |")
            md.append("| :--- | :--- | :--- | :--- | :--- | :--- |")
            for c in corrs[:5]:
                md.append(f"| `{c['var1']}` | `{c['var2']}` | `{c['pearson']}` | `{c['spearman']}` | {c['direction']} | {c['strength']} |")
            md.append("\n> **Important Note**: *Correlation does NOT equal Causation.*\n")

        # 6. Statistical Hypothesis Testing
        md.append("## 6. Statistical Analysis & Hypothesis Testing\n")
        if stat_results:
            for s in stat_results:
                sig_label = "✅ Statistically Significant" if s.get("statistically_significant") else "❌ Not Significant"
                md.append(f"### {s.get('test_name')}: `{s.get('grouping_variable')}` vs `{s.get('target_variable')}` ({sig_label})\n")
                md.append(f"- **Null Hypothesis ($H_0$)**: {s.get('null_hypothesis')}")
                md.append(f"- **Alternative Hypothesis ($H_1$)**: {s.get('alt_hypothesis')}")
                md.append(f"- **Test Statistic**: `{s.get('statistic')}` | **p-value**: `{s.get('p_value_formatted')}` (alpha = 0.05)")
                md.append(f"- **Business Interpretation**: {s.get('business_interpretation')}\n")
        else:
            md.append("No binary or categorical pairings met criteria for formal statistical testing.\n")

        # 7. Machine Learning Baseline Extension
        if ml_results and "metrics" in ml_results:
            md.append("## 7. Data Science ML Extension\n")
            md.append(f"### Model: {ml_results.get('model_name')} ({ml_results.get('problem_type')})\n")
            md.append(f"- **Target Variable**: `{ml_results.get('target_column')}`")
            md.append(f"- **Train / Test Sample**: `{ml_results.get('sample_size_train')} / {ml_results.get('sample_size_test')}`")
            md.append("- **Evaluation Metrics**:")
            for mk, mv in ml_results["metrics"].items():
                md.append(f"  - **{mk.upper()}**: `{mv}`")
            if ml_results.get("top_feature_importance"):
                md.append("- **Top Predictor Features**:")
                for fk, fv in ml_results["top_feature_importance"].items():
                    md.append(f"  - `{fk}`: {round(fv*100, 2)}%")
            md.append(f"\n> *Interpretation*: {ml_results.get('interpretation')}\n")

        # 8. Visualizations Summary
        md.append("## 8. Automated Visualization Studio Recommendations\n")
        if chart_recs:
            for ch in chart_recs:
                md.append(f"### {ch.get('chart_type')}: {ch.get('question')}\n")
                md.append(f"- **Data Required**: `{ch.get('data_required')}`")
                md.append(f"- **Aggregation**: `{ch.get('aggregation')}`")
                md.append(f"- **Analytical Insight**: {ch.get('insight')}\n")

        # 9. Key Evidence-Based Insights
        md.append("## 9. Key Evidence-Based Insights\n")
        for ins in insights:
            md.append(f"### [{ins.get('level')}] {ins.get('title')}\n")
            md.append(f"- **Finding**: {ins.get('finding')}")
            md.append(f"- **Evidence**: `{ins.get('evidence')}`")
            md.append(f"- **Business Meaning**: {ins.get('business_meaning')}\n")

        # 10. Prioritized Recommendations
        md.append("## 10. Strategic Problem-Solving & Actionable Recommendations\n")
        for rec in recommendations:
            md.append(f"### [{rec.get('priority')}] {rec.get('problem')}\n")
            md.append(f"- **WHAT IS THE PROBLEM**: {rec.get('what_is_the_problem')}")
            md.append(f"- **WHAT TO SOLVE**: {rec.get('what_to_solve')}")
            md.append(f"- **HOW TO SOLVE**:\n{rec.get('how_to_solve')}")
            md.append(f"- **EXPECTED IMPACT**: {rec.get('expected_impact')}")
            md.append(f"- **IMMEDIATE NEXT STEP**: `{rec.get('next_step')}`\n")

        # 11. Analytical Limitations & Guardrails
        md.append("## 11. Analytical Limitations & Disclosure\n")
        md.append("1. **Causal Limitations**: All observed relationships represent mathematical association/correlation and do NOT establish causality.")
        md.append("2. **Sampling Scope**: Findings reflect the specific timeframe and observation grain present in the uploaded dataset.")
        md.append("3. **Imputation Boundary**: Missing values were imputed using standard statistical heuristics (median/mode); extreme tail distributions should be evaluated carefully.\n")

        # 12. Methodology & Data Dictionary
        md.append("## 12. Methodology & Data Dictionary\n")
        md.append("### Pipeline Methodology\n")
        md.append("`INGESTION` -> `PROFILING` -> `CLEANING` -> `VALIDATION` -> `EDA` -> `STATISTICS` -> `VISUALIZATION` -> `REPORTING`\n")
        
        md.append("### Data Dictionary\n")
        md.append("| Column | Inferred Type | Missing % | Unique Count | Role |")
        md.append("| :--- | :--- | :--- | :--- | :--- |")
        for col, info in metadata.get("column_analysis", {}).items():
            md.append(f"| `{col}` | `{info['inferred_type']}` | `{info['missing_pct']}%` | `{info['unique_count']}` | {'Primary Key' if info['is_primary_key_candidate'] else 'Feature'} |")

        return "\n".join(md)
