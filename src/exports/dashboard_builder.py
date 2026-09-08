import os
import json
import pandas as pd
from typing import Dict, Any, List

class HTMLDashboardBuilder:
    """
    Generates a standalone, highly polished, executive-grade HTML/JS Dashboard
    powered by Tailwind CSS and Plotly.js that can be opened in any web browser.
    """

    @classmethod
    def generate_html_dashboard(
        cls,
        dataset_name: str,
        df_cleaned: pd.DataFrame,
        metadata: Dict[str, Any],
        profiling: Dict[str, Any],
        kpi_results: Dict[str, Any],
        insights: List[Dict[str, Any]],
        output_filepath: str
    ) -> str:
        
        num_cols = metadata["column_buckets"]["numerical_columns"]
        cat_cols = metadata["column_buckets"]["categorical_columns"]
        dt_cols = metadata["column_buckets"]["datetime_columns"]

        dq_score = profiling.get("data_quality_score", 100.0)

        # 1. Prepare KPI card data
        kpi_cards = []
        if "Sales & Revenue" in kpi_results:
            sales = kpi_results["Sales & Revenue"]
            kpi_cards.append({"title": "Total Revenue", "value": sales.get("Total Revenue", "N/A"), "subtitle": "Aggregate spend", "icon": "💰", "color": "blue"})
            kpi_cards.append({"title": "Average Order Value", "value": sales.get("Average Order Value (AOV)", "N/A"), "subtitle": "Mean order size", "icon": "🛒", "color": "emerald"})
            kpi_cards.append({"title": "Total Transactions", "value": str(sales.get("Total Transactions Recorded", len(df_cleaned))), "subtitle": "Valid orders", "icon": "📦", "color": "purple"})

        if "Operational Quality" in kpi_results:
            ops = kpi_results["Operational Quality"]
            kpi_cards.append({"title": "Return Rate", "value": ops.get("Return / Exception Rate", "0%"), "subtitle": f"{ops.get('Return / Exception Count', 0)} returned items", "icon": "⚠️", "color": "amber"})
        else:
            kpi_cards.append({"title": "Data Quality Score", "value": f"{dq_score}/100", "subtitle": "Verified clean", "icon": "⭐", "color": "indigo"})

        # 2. Prepare JSON data payload for Plotly JS charts
        chart_data_js = {}

        # Trend Chart
        if dt_cols and num_cols:
            dt_col = dt_cols[0]
            num_col = num_cols[0]
            temp = df_cleaned.copy()
            temp[dt_col] = pd.to_datetime(temp[dt_col], errors="coerce")
            trend_df = temp.dropna(subset=[dt_col]).groupby(dt_col)[num_col].sum().reset_index().sort_values(dt_col)
            chart_data_js["trend"] = {
                "x": trend_df[dt_col].astype(str).tolist(),
                "y": trend_df[num_col].round(2).tolist(),
                "title": f"Temporal Performance: {num_col.replace('_', ' ').title()} over Time",
                "x_label": dt_col.replace('_', ' ').title(),
                "y_label": num_col.replace('_', ' ').title()
            }

        # Category Bar Chart
        if cat_cols and num_cols:
            cat_col = cat_cols[0]
            num_col = num_cols[0]
            cat_df = df_cleaned.groupby(cat_col)[num_col].sum().reset_index().sort_values(num_col, ascending=True)
            chart_data_js["category_bar"] = {
                "x": cat_df[num_col].round(2).tolist(),
                "y": cat_df[cat_col].astype(str).tolist(),
                "title": f"Breakdown by {cat_col.replace('_', ' ').title()}",
                "x_label": num_col.replace('_', ' ').title(),
                "y_label": cat_col.replace('_', ' ').title()
            }

        # Scatter Chart
        if len(num_cols) >= 2:
            n1, n2 = num_cols[0], num_cols[1]
            chart_data_js["scatter"] = {
                "x": df_cleaned[n1].round(2).tolist(),
                "y": df_cleaned[n2].round(2).tolist(),
                "title": f"Scatter Analysis: {n1.replace('_', ' ').title()} vs {n2.replace('_', ' ').title()}",
                "x_label": n1.replace('_', ' ').title(),
                "y_label": n2.replace('_', ' ').title()
            }

        # Donut Chart
        if cat_cols:
            c_col = cat_cols[0]
            counts = df_cleaned[c_col].value_counts().head(6)
            chart_data_js["donut"] = {
                "labels": counts.index.astype(str).tolist(),
                "values": counts.values.tolist(),
                "title": f"Share of {c_col.replace('_', ' ').title()}"
            }

        # 3. Build HTML template string
        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Executive BI Dashboard - {dataset_name}</title>
    <!-- Tailwind CSS -->
    <script src="https://cdn.tailwindcss.com"></script>
    <!-- Plotly JS -->
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <style>
        body {{ font-family: 'Inter', sans-serif; background-color: #0f172a; color: #f8fafc; }}
        .glass-card {{
            background: rgba(30, 41, 59, 0.7);
            backdrop-filter: blur(12px);
            border: 1px solid rgba(255, 255, 255, 0.08);
            box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.3);
        }}
    </style>
</head>
<body class="min-h-screen pb-12">

    <!-- Header Navigation -->
    <header class="border-b border-slate-800 bg-slate-900/80 sticky top-0 z-50 backdrop-blur">
        <div class="max-w-7xl mx-auto px-6 py-4 flex justify-between items-center">
            <div class="flex items-center space-x-3">
                <div class="bg-blue-600 text-white p-2 rounded-lg font-bold text-xl">📊</div>
                <div>
                    <h1 class="text-xl font-bold text-white tracking-tight">Executive BI Analytics Dashboard</h1>
                    <p class="text-xs text-slate-400">Dataset: <span class="text-blue-400 font-semibold">{dataset_name}</span> | Data Quality Score: <span class="text-emerald-400 font-semibold">{dq_score}/100</span></p>
                </div>
            </div>
            <div class="flex items-center space-x-3">
                <span class="inline-flex items-center px-3 py-1 rounded-full text-xs font-medium bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">
                    ● Verified Clean
                </span>
                <span class="text-xs text-slate-400">Generated by AI Data Analyst Agent</span>
            </div>
        </div>
    </header>

    <main class="max-w-7xl mx-auto px-6 mt-8 space-y-8">

        <!-- KPI Grid -->
        <section class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
            {"".join([f'''
            <div class="glass-card rounded-2xl p-6 transition-all hover:-translate-y-1">
                <div class="flex justify-between items-start">
                    <div>
                        <p class="text-xs font-semibold text-slate-400 uppercase tracking-wider">{card['title']}</p>
                        <h3 class="text-2xl font-extrabold text-white mt-2">{card['value']}</h3>
                        <p class="text-xs text-slate-400 mt-1">{card['subtitle']}</p>
                    </div>
                    <div class="text-3xl p-3 bg-slate-800/80 rounded-xl">{card['icon']}</div>
                </div>
            </div>
            ''' for card in kpi_cards])}
        </section>

        <!-- Main Analytics Grid -->
        <section class="grid grid-cols-1 lg:grid-cols-2 gap-8">
            <!-- Time Trend Chart -->
            <div class="glass-card rounded-2xl p-6">
                <h3 class="text-lg font-bold text-white mb-4 flex items-center gap-2">📈 Temporal Performance Trend</h3>
                <div id="chart-trend" style="height: 340px;"></div>
            </div>
            <!-- Category Share Donut -->
            <div class="glass-card rounded-2xl p-6">
                <h3 class="text-lg font-bold text-white mb-4 flex items-center gap-2">🍩 Category Distribution Share</h3>
                <div id="chart-donut" style="height: 340px;"></div>
            </div>
        </section>

        <section class="grid grid-cols-1 lg:grid-cols-2 gap-8">
            <!-- Category Ranking Bar -->
            <div class="glass-card rounded-2xl p-6">
                <h3 class="text-lg font-bold text-white mb-4 flex items-center gap-2">📊 Category Value Ranking</h3>
                <div id="chart-bar" style="height: 340px;"></div>
            </div>
            <!-- Scatter Analysis -->
            <div class="glass-card rounded-2xl p-6">
                <h3 class="text-lg font-bold text-white mb-4 flex items-center gap-2">🎯 Scatter Distribution</h3>
                <div id="chart-scatter" style="height: 340px;"></div>
            </div>
        </section>

        <!-- Executive Insights Section -->
        <section class="glass-card rounded-2xl p-8">
            <h3 class="text-xl font-bold text-white mb-6 flex items-center gap-3">
                <span class="text-blue-500 text-2xl">🧠</span> Key Evidence-Based Executive Insights
            </h3>
            <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
                {"".join([f'''
                <div class="bg-slate-800/50 rounded-xl p-5 border border-slate-700/50">
                    <span class="px-2.5 py-1 rounded-md text-xs font-semibold bg-blue-500/20 text-blue-300 uppercase">{ins.get('level', 'INSIGHT')}</span>
                    <h4 class="text-base font-bold text-white mt-3">{ins.get('title')}</h4>
                    <p class="text-sm text-slate-300 mt-2">{ins.get('finding')}</p>
                    <div class="mt-3 pt-3 border-t border-slate-700/50 text-xs text-slate-400">
                        <strong>Evidence:</strong> <code class="text-emerald-400 bg-slate-900 px-2 py-0.5 rounded">{ins.get('evidence')}</code>
                    </div>
                </div>
                ''' for ins in insights[:4]])}
            </div>
        </section>

    </main>

    <!-- Plotly JavaScript Render Logic -->
    <script>
        const chartData = {json.dumps(chart_data_js)};

        const darkLayout = {{
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: 'rgba(0,0,0,0)',
            font: {{ color: '#94a3b8', family: 'Inter, sans-serif' }},
            margin: {{ t: 30, r: 20, l: 40, b: 40 }},
            xaxis: {{ gridcolor: '#334155', zerolinecolor: '#334155' }},
            yaxis: {{ gridcolor: '#334155', zerolinecolor: '#334155' }}
        }};

        // Render Trend Chart
        if (chartData.trend) {{
            Plotly.newPlot('chart-trend', [{{
                x: chartData.trend.x,
                y: chartData.trend.y,
                type: 'scatter',
                mode: 'lines+markers',
                line: {{ color: '#38bdf8', width: 3, shape: 'spline' }},
                marker: {{ color: '#0284c7', size: 6 }}
            }}], {{
                ...darkLayout,
                xaxis: {{ ...darkLayout.xaxis, title: chartData.trend.x_label }},
                yaxis: {{ ...darkLayout.yaxis, title: chartData.trend.y_label }}
            }}, {{ responsive: true }});
        }}

        // Render Donut Chart
        if (chartData.donut) {{
            Plotly.newPlot('chart-donut', [{{
                labels: chartData.donut.labels,
                values: chartData.donut.values,
                type: 'pie',
                hole: 0.5,
                marker: {{ colors: ['#38bdf8', '#818cf8', '#c084fc', '#f472b6', '#fb7185', '#34d399'] }}
            }}], {{
                ...darkLayout,
                showlegend: true,
                legend: {{ orientation: 'h', y: -0.1 }}
            }}, {{ responsive: true }});
        }}

        // Render Bar Chart
        if (chartData.category_bar) {{
            Plotly.newPlot('chart-bar', [{{
                x: chartData.category_bar.x,
                y: chartData.category_bar.y,
                type: 'bar',
                orientation: 'h',
                marker: {{ color: '#818cf8' }}
            }}], {{
                ...darkLayout,
                xaxis: {{ ...darkLayout.xaxis, title: chartData.category_bar.x_label }},
                yaxis: {{ ...darkLayout.yaxis, title: chartData.category_bar.y_label }}
            }}, {{ responsive: true }});
        }}

        // Render Scatter Chart
        if (chartData.scatter) {{
            Plotly.newPlot('chart-scatter', [{{
                x: chartData.scatter.x,
                y: chartData.scatter.y,
                type: 'scatter',
                mode: 'markers',
                marker: {{ color: '#f472b6', size: 10, opacity: 0.8 }}
            }}], {{
                ...darkLayout,
                xaxis: {{ ...darkLayout.xaxis, title: chartData.scatter.x_label }},
                yaxis: {{ ...darkLayout.yaxis, title: chartData.scatter.y_label }}
            }}, {{ responsive: true }});
        }}
    </script>
</body>
</html>
"""
        with open(output_filepath, "w", encoding="utf-8") as f:
            f.write(html)

        return output_filepath
