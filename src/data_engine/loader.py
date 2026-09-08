import os
import io
import re
import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, Optional, List

class DataLoader:
    """
    Handles dataset ingestion from multiple sources, inferring column roles,
    detecting data types, duplicates, missingness, PII, and quality anomalies
    while strictly preserving the original dataset.
    """

    PII_PATTERNS = {
        "email": re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"),
        "phone": re.compile(r"(\+\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}"),
        "ssn": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
        "credit_card": re.compile(r"\b(?:\d[ -]*?){13,16}\b"),
    }

    NULL_LIKE_STRINGS = {"", "null", "none", "na", "n/a", "nan", "missing", "unknown", "?", "-", "."}

    @classmethod
    def load_file(cls, source: Any, filename: Optional[str] = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Loads dataset from filepath, bytes/buffer, or existing DataFrame.
        Returns (df_original, metadata)
        """
        df = None
        file_type = "DataFrame"

        if isinstance(source, pd.DataFrame):
            df = source.copy()
            file_type = "DataFrame"
        elif isinstance(source, (str, os.PathLike)):
            filepath = str(source)
            ext = os.path.splitext(filepath)[1].lower()
            file_type = ext.lstrip(".")
            if ext == ".csv":
                df = pd.read_csv(filepath)
            elif ext in [".xlsx", ".xls"]:
                df = pd.read_excel(filepath)
            elif ext == ".json":
                df = pd.read_json(filepath)
            elif ext == ".parquet":
                df = pd.read_parquet(filepath)
            else:
                # Try CSV as default fallback
                df = pd.read_csv(filepath)
        elif hasattr(source, "read"):
            name = filename or getattr(source, "name", "file.csv")
            ext = os.path.splitext(name)[1].lower()
            file_type = ext.lstrip(".")
            if ext in [".xlsx", ".xls"]:
                df = pd.read_excel(source)
            elif ext == ".parquet":
                df = pd.read_parquet(source)
            elif ext == ".json":
                df = pd.read_json(source)
            else:
                df = pd.read_csv(source)
        else:
            raise ValueError(f"Unsupported data source type: {type(source)}")

        metadata = cls.inspect_dataset(df, file_type=file_type)
        return df, metadata

    @classmethod
    def inspect_dataset(cls, df: pd.DataFrame, file_type: str = "Unknown") -> Dict[str, Any]:
        """
        Performs non-destructive initial inspection and metadata extraction.
        """
        num_rows, num_cols = df.shape
        columns = list(df.columns)
        
        column_analysis = {}
        pii_detected = {}
        
        id_columns = []
        datetime_columns = []
        numerical_columns = []
        categorical_columns = []
        text_columns = []
        boolean_columns = []
        constant_columns = []
        high_cardinality_cols = []
        primary_key_candidates = []
        target_candidates = []

        for col in columns:
            series = df[col]
            non_null_series = series.dropna()
            total_count = len(series)
            missing_count = series.isna().sum()
            
            # Check for null-like string representations
            if series.dtype == "object" or isinstance(series.dtype, pd.CategoricalDtype):
                str_series = series.astype(str).str.strip().str.lower()
                null_like_matches = str_series.isin(cls.NULL_LIKE_STRINGS).sum()
            else:
                null_like_matches = 0

            total_missing_effective = missing_count + null_like_matches
            missing_pct = (total_missing_effective / total_count * 100) if total_count > 0 else 0.0

            unique_count = series.nunique(dropna=True)
            unique_pct = (unique_count / total_count * 100) if total_count > 0 else 0.0

            # Inferred Type Detection
            inferred_type = cls._infer_column_type(series)

            # High cardinality / Constant check
            if unique_count <= 1:
                constant_columns.append(col)
            elif unique_pct > 80 and inferred_type in ["categorical", "text"]:
                high_cardinality_cols.append(col)

            # Primary Key check
            if unique_count == total_count and missing_count == 0 and total_count > 0:
                primary_key_candidates.append(col)

            # Categorization into buckets
            if col.lower().endswith(("_id", "id", "code", "key", "num", "no")) or (unique_count == total_count and inferred_type != "numerical"):
                id_columns.append(col)
            elif inferred_type == "datetime":
                datetime_columns.append(col)
            elif inferred_type == "numerical":
                numerical_columns.append(col)
                if col.lower() in ["target", "label", "outcome", "churn", "is_churn", "fraud", "sales", "revenue", "spend", "spend_amount", "spend_total"]:
                    target_candidates.append(col)
            elif inferred_type == "boolean":
                boolean_columns.append(col)
                if col.lower().startswith(("is_", "has_", "flag_")):
                    target_candidates.append(col)
            elif inferred_type == "categorical":
                categorical_columns.append(col)
            else:
                text_columns.append(col)

            # PII Check
            pii_type = cls._detect_pii(non_null_series)
            if pii_type:
                pii_detected[col] = pii_type

            column_analysis[col] = {
                "inferred_type": inferred_type,
                "raw_dtype": str(series.dtype),
                "total_count": total_count,
                "missing_count": missing_count,
                "null_like_count": null_like_matches,
                "missing_pct": round(missing_pct, 2),
                "unique_count": unique_count,
                "unique_pct": round(unique_pct, 2),
                "is_primary_key_candidate": col in primary_key_candidates,
                "is_constant": col in constant_columns,
                "is_pii": pii_detected.get(col, None)
            }

        # Check duplicate rows
        duplicate_rows_count = int(df.duplicated().sum())

        return {
            "file_type": file_type,
            "shape": (num_rows, num_cols),
            "num_rows": num_rows,
            "num_cols": num_cols,
            "columns": columns,
            "column_analysis": column_analysis,
            "column_buckets": {
                "id_columns": id_columns,
                "datetime_columns": datetime_columns,
                "numerical_columns": numerical_columns,
                "categorical_columns": categorical_columns,
                "text_columns": text_columns,
                "boolean_columns": boolean_columns,
                "constant_columns": constant_columns,
                "high_cardinality_columns": high_cardinality_cols,
                "primary_key_candidates": primary_key_candidates,
                "target_candidates": target_candidates,
            },
            "duplicate_rows_count": duplicate_rows_count,
            "duplicate_rows_pct": round((duplicate_rows_count / num_rows * 100), 2) if num_rows > 0 else 0.0,
            "pii_detected": pii_detected,
        }

    @classmethod
    def _infer_column_type(cls, series: pd.Series) -> str:
        non_null = series.dropna()
        if len(non_null) == 0:
            return "empty"

        # Check boolean
        if pd.api.types.is_bool_dtype(series):
            return "boolean"
        if set(non_null.astype(str).str.lower().unique()).issubset({"true", "false", "1", "0", "yes", "no", "y", "n", "t", "f"}):
            if len(non_null.unique()) <= 2:
                return "boolean"

        # Check datetime
        if pd.api.types.is_datetime64_any_dtype(series):
            return "datetime"
        if series.dtype == "object":
            sample = non_null.head(20).astype(str)
            try:
                converted = pd.to_datetime(sample, errors="coerce")
                if converted.notna().mean() > 0.8:
                    return "datetime"
            except Exception:
                pass

        # Check numerical
        if pd.api.types.is_numeric_dtype(series):
            return "numerical"

        # Try converting string numbers
        if series.dtype == "object":
            cleaned_sample = non_null.head(50).astype(str).str.replace(",", "").str.replace("$", "").str.replace("%", "").str.strip()
            converted = pd.to_numeric(cleaned_sample, errors="coerce")
            if converted.notna().mean() > 0.85:
                return "numerical"

        # Check categorical vs text based on unique ratio
        unique_ratio = series.nunique() / len(series)
        if unique_ratio < 0.3 or series.nunique() <= 50:
            return "categorical"

        return "text"

    @classmethod
    def _detect_pii(cls, series: pd.Series) -> Optional[str]:
        if series.dtype != "object":
            return None
        sample = series.astype(str).head(50)
        for name, pattern in cls.PII_PATTERNS.items():
            matches = sample.apply(lambda x: bool(pattern.search(x)))
            if matches.mean() > 0.4:
                return name
        return None
