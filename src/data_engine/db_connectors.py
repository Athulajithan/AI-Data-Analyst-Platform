import sqlite3
import pandas as pd
from typing import Dict, Any, Tuple, Optional

class DatabaseConnector:
    """
    Manages connections and queries across relational databases (SQLite, PostgreSQL, MySQL, DuckDB)
    with strict READ-ONLY execution guardrails.
    """

    @classmethod
    def connect_and_query(cls, connection_string: str, query: str) -> Tuple[Optional[pd.DataFrame], Dict[str, Any]]:
        cleaned_query = query.strip()
        
        # Guardrail check
        from .sql_engine import SQLEngine
        match = SQLEngine.DESTRUCTIVE_KEYWORDS.search(cleaned_query)
        if match:
            return None, {
                "success": False,
                "error": f"Security Guardrail: Destructive keyword '{match.group(0).upper()}' is forbidden. Queries must be READ-ONLY.",
                "rows_returned": 0
            }

        try:
            # Check connection type
            if connection_string.startswith("sqlite:///") or connection_string.endswith(".db") or connection_string.endswith(".sqlite"):
                db_path = connection_string.replace("sqlite:///", "")
                conn = sqlite3.connect(db_path)
                df = pd.read_sql_query(cleaned_query, conn)
                conn.close()
                return df, {"success": True, "rows_returned": len(df), "columns": list(df.columns)}

            else:
                # Use pandas read_sql with generic connection string via SQLAlchemy
                import sqlalchemy
                engine = sqlalchemy.create_engine(connection_string)
                df = pd.read_sql_query(cleaned_query, engine)
                return df, {"success": True, "rows_returned": len(df), "columns": list(df.columns)}

        except Exception as e:
            return None, {"success": False, "error": f"Database Query Failed: {str(e)}", "rows_returned": 0}
