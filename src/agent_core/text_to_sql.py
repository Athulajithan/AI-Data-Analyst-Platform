import re
import pandas as pd
from typing import Dict, Any, Tuple, Optional
from src.data_engine.sql_engine import SQLEngine

class TextToSQLAssistant:
    """
    Conversational AI Assistant that translates natural language user queries
    into safe, read-only SQL queries and executes them against the dataset.
    """

    @classmethod
    def ask_data(
        cls,
        user_question: str,
        df: pd.DataFrame,
        metadata: Dict[str, Any],
        table_name: str = "dataset"
    ) -> Tuple[Optional[pd.DataFrame], Dict[str, Any]]:
        
        q_lower = user_question.lower().strip()
        num_cols = metadata["column_buckets"]["numerical_columns"]
        cat_cols = metadata["column_buckets"]["categorical_columns"]
        dt_cols = metadata["column_buckets"]["datetime_columns"]

        num_target = num_cols[0] if num_cols else "*"
        cat_target = cat_cols[0] if cat_cols else None

        # Rule-Based NLP SQL Synthesizer
        sql_query = None

        # Rule 1: Top N breakdown (e.g. "top 5 categories by spend")
        if "top" in q_lower or "highest" in q_lower or "best" in q_lower:
            limit_match = re.search(r"\b\d+\b", q_lower)
            limit_val = limit_match.group(0) if limit_match else "5"
            
            # Find referenced category or default to cat_cols[0]
            matched_cat = next((c for c in cat_cols if c.lower() in q_lower), cat_target)
            matched_num = next((n for n in num_cols if n.lower() in q_lower), num_target)

            if matched_cat and matched_num != "*":
                sql_query = f"""SELECT {matched_cat}, SUM({matched_num}) AS total_{matched_num}, ROUND(AVG({matched_num}), 2) AS avg_{matched_num}, COUNT(*) AS total_count FROM {table_name} GROUP BY {matched_cat} ORDER BY total_{matched_num} DESC LIMIT {limit_val};"""
            elif matched_cat:
                sql_query = f"""SELECT {matched_cat}, COUNT(*) AS total_count FROM {table_name} GROUP BY {matched_cat} ORDER BY total_count DESC LIMIT {limit_val};"""

        # Rule 2: Count / Total breakdown
        elif "count" in q_lower or "how many" in q_lower or "breakdown" in q_lower:
            matched_cat = next((c for c in cat_cols if c.lower() in q_lower), cat_target)
            if matched_cat:
                sql_query = f"""SELECT {matched_cat}, COUNT(*) AS total_records FROM {table_name} GROUP BY {matched_cat} ORDER BY total_records DESC;"""

        # Rule 3: Average / Mean metric query
        elif "average" in q_lower or "mean" in q_lower or "aov" in q_lower:
            matched_num = next((n for n in num_cols if n.lower() in q_lower), num_target)
            matched_cat = next((c for c in cat_cols if c.lower() in q_lower), cat_target)
            if matched_cat and matched_num != "*":
                sql_query = f"""SELECT {matched_cat}, ROUND(AVG({matched_num}), 2) AS average_{matched_num} FROM {table_name} GROUP BY {matched_cat} ORDER BY average_{matched_num} DESC;"""

        # Fallback default query
        if not sql_query:
            if cat_target and num_target != "*":
                sql_query = f"""SELECT {cat_target}, SUM({num_target}) AS total_{num_target} FROM {table_name} GROUP BY {cat_target} ORDER BY total_{num_target} DESC LIMIT 10;"""
            else:
                sql_query = f"""SELECT * FROM {table_name} LIMIT 10;"""

        # Execute Query via SQLEngine
        sql_eng = SQLEngine(df, table_name=table_name)
        res_df, q_meta = sql_eng.execute_query(sql_query)
        sql_eng.close()

        q_meta["generated_sql"] = sql_query
        q_meta["user_question"] = user_question

        return res_df, q_meta
