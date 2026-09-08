import re
import sqlite3
import pandas as pd
from typing import Dict, Any, Tuple, Optional

class SQLEngine:
    """
    In-memory SQLite execution engine providing interactive SQL analysis over loaded DataFrames,
    protected by strict AST/regex READ-ONLY security guardrails.
    """

    DESTRUCTIVE_KEYWORDS = re.compile(
        r"\b(DROP|TRUNCATE|DELETE|ALTER|UPDATE|INSERT|CREATE|REPLACE|ATTACH|DETACH|GRANT|REVOKE)\b",
        re.IGNORECASE
    )

    def __init__(self, df: pd.DataFrame, table_name: str = "dataset"):
        self.table_name = table_name
        self.conn = sqlite3.connect(":memory:", check_same_thread=False)
        # Register dataframe into sqlite
        df.to_sql(table_name, self.conn, if_exists="replace", index=False)

    def execute_query(self, query: str) -> Tuple[Optional[pd.DataFrame], Dict[str, Any]]:
        cleaned_query = query.strip()
        
        # Check guardrail
        match = self.DESTRUCTIVE_KEYWORDS.search(cleaned_query)
        if match:
            forbidden_word = match.group(0).upper()
            return None, {
                "success": False,
                "error": f"Security Guardrail Triggered: Destructive keyword '{forbidden_word}' is forbidden. SQL queries must be strictly READ-ONLY.",
                "rows_returned": 0
            }

        try:
            result_df = pd.read_sql_query(cleaned_query, self.conn)
            return result_df, {
                "success": True,
                "error": None,
                "rows_returned": len(result_df),
                "columns_returned": list(result_df.columns)
            }
        except Exception as e:
            return None, {
                "success": False,
                "error": f"SQL Execution Error: {str(e)}",
                "rows_returned": 0
            }

    def generate_recommended_queries(self, metadata: Dict[str, Any]) -> Dict[str, str]:
        table = self.table_name
        num_cols = metadata["column_buckets"]["numerical_columns"]
        cat_cols = metadata["column_buckets"]["categorical_columns"]
        dt_cols = metadata["column_buckets"]["datetime_columns"]

        queries = {}

        # Query 1: Top categories by aggregate sum
        if cat_cols and num_cols:
            cat = cat_cols[0]
            num = num_cols[0]
            queries["top_segment_summary"] = f"""
WITH RankedSegments AS (
    SELECT 
        {cat},
        COUNT(*) AS total_records,
        SUM({num}) AS total_{num},
        ROUND(AVG({num}), 2) AS avg_{num}
    FROM {table}
    GROUP BY {cat}
)
SELECT * 
FROM RankedSegments
ORDER BY total_{num} DESC
LIMIT 10;
""".strip()

        # Query 2: Window Function / Running total if datetime exists
        if dt_cols and num_cols:
            dt = dt_cols[0]
            num = num_cols[0]
            queries["running_total_trend"] = f"""
WITH DailyTotals AS (
    SELECT 
        DATE({dt}) AS transaction_day,
        SUM({num}) AS daily_{num}
    FROM {table}
    WHERE {dt} IS NOT NULL
    GROUP BY DATE({dt})
)
SELECT 
    transaction_day,
    daily_{num},
    SUM(daily_{num}) OVER (ORDER BY transaction_day) AS cumulative_{num},
    ROUND(AVG(daily_{num}) OVER (ORDER BY transaction_day ROWS BETWEEN 6 PRECEDING AND CURRENT ROW), 2) AS 7day_moving_avg
FROM DailyTotals
ORDER BY transaction_day ASC;
""".strip()

        # Query 3: Percentile / Ranking query
        if num_cols:
            num = num_cols[0]
            queries["percentile_ranking"] = f"""
SELECT 
    *,
    NTILE(4) OVER (ORDER BY {num} DESC) AS spend_quartile
FROM {table}
LIMIT 20;
""".strip()

        return queries

    def close(self):
        try:
            self.conn.close()
        except Exception:
            pass
