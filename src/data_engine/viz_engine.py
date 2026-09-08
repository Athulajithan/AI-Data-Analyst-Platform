import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, Any, List, Optional, Tuple

class VisualizationEngine:
    """
    Automated Chart Selection Engine and Plotly visualizer complying strictly
    with Section 10 & Section 11 chart selection rules.
    """

    @classmethod
    def recommend_charts(cls, df: pd.DataFrame, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        recommendations = []

        num_cols = metadata["column_buckets"]["numerical_columns"]
        cat_cols = metadata["column_buckets"]["categorical_columns"]
        dt_cols = metadata["column_buckets"]["datetime_columns"]

        # 1. Line Chart for Time Series
        if dt_cols and num_cols:
            recommendations.append({
                "question": f"How does '{num_cols[0]}' trend over time across '{dt_cols[0]}'?",
                "data_required": [dt_cols[0], num_cols[0]],
                "aggregation": f"SUM({num_cols[0]}) grouped by Date",
                "chart_type": "LINE CHART",
                "chart_id": "time_series_trend",
                "insight": f"Identifies temporal growth, seasonality, and trend velocity."
            })

        # 2. Horizontal Bar Chart for Categorical Breakdown
        if cat_cols and num_cols:
            recommendations.append({
                "question": f"Which categories in '{cat_cols[0]}' generate the highest aggregate '{num_cols[0]}'?",
                "data_required": [cat_cols[0], num_cols[0]],
                "aggregation": f"SUM({num_cols[0]}) grouped by '{cat_cols[0]}', sorted descending",
                "chart_type": "BAR CHART",
                "chart_id": "category_bar_chart",
                "insight": f"Ranks leading vs trailing segments."
            })

        # 3. Scatter Plot for Numerical Relationships
        if len(num_cols) >= 2:
            recommendations.append({
                "question": f"What is the correlation and scatter relationship between '{num_cols[0]}' and '{num_cols[1]}'?",
                "data_required": [num_cols[0], num_cols[1]],
                "aggregation": "Row-level scatter points",
                "chart_type": "SCATTER PLOT",
                "chart_id": "numerical_scatter",
                "insight": f"Displays slope, cluster patterns, and potential non-linear dependencies."
            })

        # 4. Box Plot for Outlier & Distribution Comparison
        if cat_cols and num_cols:
            recommendations.append({
                "question": f"How is '{num_cols[0]}' distributed across groups of '{cat_cols[0]}', and where are the outliers?",
                "data_required": [cat_cols[0], num_cols[0]],
                "aggregation": "Quartiles, median, and IQR bounds per group",
                "chart_type": "BOX PLOT",
                "chart_id": "distribution_box",
                "insight": f"Reveals variance, skewness, and extreme statistical outliers."
            })

        # 5. Correlation Heatmap
        if len(num_cols) >= 3:
            recommendations.append({
                "question": "What are the pairwise Pearson correlation strength scores across all numerical variables?",
                "data_required": num_cols,
                "aggregation": "Pearson Correlation Matrix",
                "chart_type": "HEATMAP",
                "chart_id": "correlation_heatmap",
                "insight": f"Highlights high co-linearity and variable dependencies."
            })

        # 6. Donut / Pie Chart for small category proportions
        if cat_cols and df[cat_cols[0]].nunique() <= 6:
            recommendations.append({
                "question": f"What is the proportional breakdown of total records across '{cat_cols[0]}'?",
                "data_required": [cat_cols[0]],
                "aggregation": f"COUNT(*) proportion by '{cat_cols[0]}'",
                "chart_type": "DONUT CHART",
                "chart_id": "category_donut",
                "insight": f"Visualizes market share and percentage breakdown."
            })

        return recommendations

    @classmethod
    def generate_chart(cls, chart_id: str, df: pd.DataFrame, metadata: Dict[str, Any]) -> Optional[go.Figure]:
        num_cols = metadata["column_buckets"]["numerical_columns"]
        cat_cols = metadata["column_buckets"]["categorical_columns"]
        dt_cols = metadata["column_buckets"]["datetime_columns"]

        try:
            if chart_id == "time_series_trend" and dt_cols and num_cols:
                temp_df = df.copy()
                temp_df[dt_cols[0]] = pd.to_datetime(temp_df[dt_cols[0]], errors="coerce")
                grouped = temp_df.dropna(subset=[dt_cols[0]]).groupby(dt_cols[0])[num_cols[0]].sum().reset_index()
                fig = px.line(
                    grouped,
                    x=dt_cols[0],
                    y=num_cols[0],
                    title=f"Time Series Trend: {num_cols[0]} over {dt_cols[0]}",
                    labels={dt_cols[0]: "Date", num_cols[0]: num_cols[0].replace("_", " ").title()},
                    template="plotly_white"
                )
                fig.update_traces(line=dict(width=3, color="#1f77b4"))
                return fig

            elif chart_id == "category_bar_chart" and cat_cols and num_cols:
                grouped = df.groupby(cat_cols[0])[num_cols[0]].sum().reset_index()
                grouped = grouped.sort_values(by=num_cols[0], ascending=True)
                fig = px.bar(
                    grouped,
                    x=num_cols[0],
                    y=cat_cols[0],
                    orientation="h",
                    title=f"{num_cols[0].replace('_', ' ').title()} by {cat_cols[0].replace('_', ' ').title()}",
                    labels={num_cols[0]: num_cols[0].replace("_", " ").title(), cat_cols[0]: cat_cols[0].replace("_", " ").title()},
                    color=num_cols[0],
                    color_continuous_scale="Blues",
                    template="plotly_white"
                )
                return fig

            elif chart_id == "numerical_scatter" and len(num_cols) >= 2:
                fig = px.scatter(
                    df,
                    x=num_cols[0],
                    y=num_cols[1],
                    color=cat_cols[0] if cat_cols else None,
                    title=f"Scatter Analysis: {num_cols[0].title()} vs {num_cols[1].title()}",
                    labels={num_cols[0]: num_cols[0].title(), num_cols[1]: num_cols[1].title()},
                    trendline="ols" if len(df) > 10 else None,
                    template="plotly_white"
                )
                return fig

            elif chart_id == "distribution_box" and cat_cols and num_cols:
                fig = px.box(
                    df,
                    x=cat_cols[0],
                    y=num_cols[0],
                    points="outliers",
                    title=f"Distribution Box Plot: {num_cols[0].title()} across {cat_cols[0].title()}",
                    color=cat_cols[0],
                    template="plotly_white"
                )
                return fig

            elif chart_id == "correlation_heatmap" and len(num_cols) >= 2:
                corr_matrix = df[num_cols].corr(method="pearson").round(2)
                fig = px.imshow(
                    corr_matrix,
                    text_auto=True,
                    color_continuous_scale="RdBu_r",
                    zmin=-1,
                    zmax=1,
                    title="Pearson Correlation Heatmap Matrix",
                    template="plotly_white"
                )
                return fig

            elif chart_id == "category_donut" and cat_cols:
                counts = df[cat_cols[0]].value_counts().reset_index()
                counts.columns = [cat_cols[0], "count"]
                fig = px.pie(
                    counts,
                    names=cat_cols[0],
                    values="count",
                    hole=0.4,
                    title=f"Proportional Breakdown of {cat_cols[0].title()}",
                    template="plotly_white"
                )
                return fig

        except Exception as e:
            print(f"Chart generation error for {chart_id}: {e}")

        return None
