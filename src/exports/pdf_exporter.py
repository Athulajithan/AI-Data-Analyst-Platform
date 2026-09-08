import os
import pandas as pd
from typing import Dict, Any, List

class PrintablePDFExporter:
    """
    Generates a print-ready Executive PDF/HTML document styled with print CSS (@media print),
    page breaks, executive KPI cards, data tables, and formal signatures.
    """

    @classmethod
    def generate_printable_report(
        cls,
        dataset_name: str,
        metadata: Dict[str, Any],
        profiling: Dict[str, Any],
        report_md: str,
        output_filepath: str
    ) -> str:
        
        dq_score = profiling.get("data_quality_score", 100.0)
        shape = metadata.get("shape", (0, 0))

        # Convert basic Markdown report elements to clean HTML print layout
        html_body = report_md.replace("\n# ", "\n<h1>").replace("\n## ", "\n<h2>").replace("\n### ", "\n<h3>").replace("```", "")
        
        printable_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Executive Print Report - {dataset_name}</title>
    <style>
        @page {{ size: A4; margin: 20mm; }}
        body {{
            font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
            color: #1e293b;
            line-height: 1.6;
            background-color: #ffffff;
            margin: 0;
            padding: 20px;
        }}
        .header-box {{
            border-bottom: 3px solid #0284c7;
            padding-bottom: 15px;
            margin-bottom: 25px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        .title {{ font-size: 24px; font-weight: bold; color: #0f172a; margin: 0; }}
        .subtitle {{ font-size: 13px; color: #64748b; margin-top: 4px; }}
        .badge {{
            background-color: #e0f2fe;
            color: #0369a1;
            padding: 6px 12px;
            border-radius: 20px;
            font-size: 12px;
            font-weight: bold;
        }}
        h1, h2, h3 {{ color: #0f172a; margin-top: 25px; page-break-after: avoid; }}
        h2 {{ border-bottom: 1px solid #e2e8f0; padding-bottom: 6px; font-size: 18px; }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 15px 0;
            font-size: 12px;
        }}
        th, td {{ border: 1px solid #cbd5e1; padding: 8px 12px; text-align: left; }}
        th {{ background-color: #f1f5f9; color: #334155; font-weight: bold; }}
        .page-break {{ page-break-before: always; }}
        @media print {{
            body {{ padding: 0; }}
            .no-print {{ display: none; }}
        }}
    </style>
</head>
<body>

    <div class="no-print" style="margin-bottom: 20px; text-align: right;">
        <button onclick="window.print()" style="background-color: #0284c7; color: white; border: none; padding: 10px 20px; border-radius: 6px; font-weight: bold; cursor: pointer;">
            🖨️ Print / Save as PDF
        </button>
    </div>

    <div class="header-box">
        <div>
            <h1 class="title">Executive Data Analytics Report</h1>
            <p class="subtitle">Dataset: {dataset_name} | Verified Rows: {shape[0]} | Features: {shape[1]}</p>
        </div>
        <div>
            <span class="badge">Data Quality Score: {dq_score}/100</span>
        </div>
    </div>

    <div class="content">
        {html_body}
    </div>

    <div style="margin-top: 40px; border-top: 1px solid #cbd5e1; padding-top: 15px; font-size: 11px; color: #94a3b8; text-align: center;">
        <p>Prepared by Expert AI Data Analyst Agent • Confidential Business Artifact</p>
    </div>

</body>
</html>
"""
        with open(output_filepath, "w", encoding="utf-8") as f:
            f.write(printable_html)

        return output_filepath
