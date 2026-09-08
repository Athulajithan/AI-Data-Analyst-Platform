import sqlite3
import pandas as pd
from typing import Dict, Any, Tuple, Optional

class DuckDBAccelerator:
    """
    High-performance analytical acceleration engine utilizing DuckDB (or in-memory SQLite fallback)
    for high-speed aggregation and query processing on large datasets.
    """

    def __init__(self, df: pd.DataFrame, table_name: str = "dataset"):
        self.table_name = table_name
        self.df = df
        self.duckdb_available = False

        try:
            import duckdb
            self.con = duckdb.connect(database=":memory:")
            self.con.register(table_name, df)
            self.duckdb_available = True
        except ImportError:
            self.con = sqlite3.connect(":memory:", check_same_thread=False)
            df.to_sql(table_name, self.con, if_exists="replace", index=False)
            self.duckdb_available = False

    def query(self, sql_query: str) -> Tuple[Optional[pd.DataFrame], Dict[str, Any]]:
        from .sql_engine import SQLEngine
        if SQLEngine.DESTRUCTIVE_KEYWORDS.search(sql_query):
            return None, {"success": False, "error": "Security Guardrail: Query must be READ-ONLY."}

        try:
            if self.duckdb_available:
                res_df = self.con.execute(sql_query).df()
            else:
                res_df = pd.read_sql_query(sql_query, self.con)
            return res_df, {"success": True, "engine": "DuckDB" if self.duckdb_available else "SQLite", "rows": len(res_df)}
        except Exception as e:
            return None, {"success": False, "error": str(e)}

    def close(self):
        try:
            self.con.close()
        except Exception:
            pass
