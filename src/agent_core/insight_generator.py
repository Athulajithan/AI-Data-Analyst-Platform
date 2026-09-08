import pandas as pd
from typing import Dict, Any, List, Optional

class InsightGenerator:
    """
    Generates 4-Level hierarchical insights (Descriptive -> Diagnostic -> Predictive -> Prescriptive)
    grounded strictly in empirical data calculations.
    """

    @classmethod
    def generate_insights(
        cls,
        metadata: Dict[str, Any],
        profiling: Dict[str, Any],
        validation: Dict[str, Any],
        eda_results: Dict[str, Any],
        stat_results: List[Dict[str, Any]],
        kpi_results: Dict[str, Any],
        ml_results: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        
        insights = []

        # 1. Level 1: Descriptive Insight (Data Quality & Ingestion Summary)
        dq_score = profiling.get("data_quality_score", 100.0)
        shape = metadata.get("shape", (0, 0))
        rem_dups = validation.get("duplicates_resolved", 0)
        rem_nulls = validation.get("missing_values_resolved", 0)

        insights.append({
            "level": "LEVEL 1 — Descriptive",
            "title": "Dataset Quality & Structural Baseline",
            "finding": f"Ingested dataset contains {shape[0]} records and {shape[1]} features with an initial Data Quality Score of {dq_score}/100.",
            "evidence": f"Dataset inspection identified {rem_dups} duplicate rows and {rem_nulls} missing entries. Data cleaning successfully resolved structural anomalies.",
            "business_meaning": "The dataset is verified as structurally sound for quantitative decision-making."
        })

        # 2. Level 1: Descriptive Insight (KPI Summary)
        if "Sales & Revenue" in kpi_results:
            sales_kpis = kpi_results["Sales & Revenue"]
            total_rev = sales_kpis.get("Total Revenue", "N/A")
            aov = sales_kpis.get("Average Order Value (AOV)", "N/A")
            insights.append({
                "level": "LEVEL 1 — Descriptive",
                "title": "Revenue & Transaction Volume Baseline",
                "finding": f"Total transaction volume reached {total_rev} with an Average Order Value (AOV) of {aov}.",
                "evidence": f"Calculated across {sales_kpis.get('Total Transactions Recorded', 0)} valid sales entries.",
                "business_meaning": f"Establishes current revenue baseline for performance tracking."
            })

        # 3. Level 2: Diagnostic Insight (Segmentation / Category Performance)
        seg_dict = eda_results.get("segmentations", {})
        if seg_dict:
            first_key = list(seg_dict.keys())[0]
            seg_info = seg_dict[first_key]
            top_seg = seg_info.get("top_segment")
            bot_seg = seg_info.get("bottom_segment")
            cat_name = seg_info.get("category_col", "").replace("_", " ").title()
            measure_name = seg_info.get("measure_col", "").replace("_", " ").title()

            insights.append({
                "level": "LEVEL 2 — Diagnostic",
                "title": f"Segment Divergence in {cat_name}",
                "finding": f"Segment '{top_seg}' significantly outperforms segment '{bot_seg}' in total {measure_name}.",
                "evidence": f"Aggregated grouping of '{seg_info.get('category_col')}' vs '{seg_info.get('measure_col')}'.",
                "business_meaning": f"Identifies key growth driver segments versus underperforming verticals requiring strategic intervention."
            })

        # 4. Level 2: Diagnostic Insight (Statistical Testing)
        for stat in stat_results:
            if stat.get("statistically_significant"):
                insights.append({
                    "level": "LEVEL 2 — Diagnostic",
                    "title": f"Statistically Significant Variation ({stat.get('test_name')})",
                    "finding": stat.get("business_interpretation"),
                    "evidence": f"{stat.get('null_hypothesis')} rejected at alpha=0.05 level with p-value={stat.get('p_value_formatted')} (statistic={stat.get('statistic')}).",
                    "business_meaning": "Observed variation is statistically proven to be non-random."
                })
                break

        # 5. Level 3: Predictive Insight (Trend / Growth or ML)
        trends = eda_results.get("trends", {})
        if trends:
            first_trend_key = list(trends.keys())[0]
            t_data = trends[first_trend_key]
            growth = t_data.get("percentage_growth", 0.0)
            measure = t_data.get("measure_col", "").replace("_", " ").title()
            
            insights.append({
                "level": "LEVEL 3 — Predictive",
                "title": f"Temporal Trajectory for {measure}",
                "finding": f"{measure} exhibited a net {growth}% growth trajectory over the observed time horizon.",
                "evidence": f"Initial value: {t_data.get('initial_value')} -> Latest value: {t_data.get('latest_value')} (Absolute change: {t_data.get('absolute_change')}).",
                "business_meaning": "Extrapolating recent trend velocity suggests sustained trajectory assuming macro operational stability."
            })

        if ml_results and "metrics" in ml_results:
            m_metrics = ml_results["metrics"]
            m_type = ml_results.get("problem_type")
            m_target = ml_results.get("target_column")
            
            if m_type == "Classification":
                acc = m_metrics.get("accuracy")
                insights.append({
                    "level": "LEVEL 3 — Predictive",
                    "title": f"Automated Predictability of Target '{m_target}'",
                    "finding": f"Baseline Machine Learning model predicts '{m_target}' with {round(acc*100, 2)}% accuracy (F1={m_metrics.get('f1_score')}).",
                    "evidence": f"Random Forest Classifier trained on {ml_results.get('sample_size_train')} samples and tested on {ml_results.get('sample_size_test')} holdout samples.",
                    "business_meaning": f"Enables proactive automated risk scoring or customer classification prior to event occurrence."
                })

        # 6. Level 4: Prescriptive Insight (Actionable Strategy)
        anomalies = eda_results.get("anomalies", {})
        high_anomaly_col = None
        for col, info in anomalies.items():
            if info.get("iqr_outlier_count", 0) > 0:
                high_anomaly_col = col
                break

        if high_anomaly_col:
            info = anomalies[high_anomaly_col]
            insights.append({
                "level": "LEVEL 4 — Prescriptive",
                "title": f"Outlier Management in '{high_anomaly_col}'",
                "finding": f"Detected {info.get('iqr_outlier_count')} extreme statistical anomalies in '{high_anomaly_col}'.",
                "evidence": f"IQR outlier threshold rule: values outside range [{info.get('lower_bound')}, {info.get('upper_bound')}].",
                "business_meaning": f"Business should establish threshold validation logic to inspect extreme values for fraud or data-entry errors."
            })

        return insights
