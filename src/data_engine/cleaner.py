import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, List, Optional

class DataCleaner:
    """
    Cleans datasets non-destructively, preserving `df_original` and generating a
    detailed `TransformationLog` for every cleaning operation.
    """

    def __init__(self, df_original: pd.DataFrame, metadata: Dict[str, Any]):
        self.df_original = df_original
        self.df_cleaned = df_original.copy()
        self.metadata = metadata
        self.transformation_log: List[Dict[str, Any]] = []

    def run_full_cleaning_pipeline(
        self,
        remove_duplicates: bool = True,
        trim_whitespace: bool = True,
        standardize_nulls: bool = True,
        standardize_categories: bool = True,
        impute_missing_numerical: str = "median",  # "median", "mean", or "none"
        impute_missing_categorical: str = "mode",  # "mode", "unknown", or "none"
        cast_datetime: bool = True,
    ) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
        """
        Executes an end-to-end automated cleaning pipeline.
        """
        if trim_whitespace:
            self._trim_string_whitespace()
        if standardize_nulls:
            self._standardize_null_strings()
        if remove_duplicates:
            self._remove_duplicate_rows()
        if standardize_categories:
            self._standardize_categorical_values()
        if cast_datetime:
            self._cast_datetime_columns()
        if impute_missing_numerical != "none":
            self._impute_numerical(strategy=impute_missing_numerical)
        if impute_missing_categorical != "none":
            self._impute_categorical(strategy=impute_missing_categorical)

        return self.df_cleaned, self.transformation_log

    def _log_transformation(self, issue: str, rationale: str, action: str, records_affected: int, details: Optional[str] = None):
        self.transformation_log.append({
            "step": len(self.transformation_log) + 1,
            "issue": issue,
            "rationale": rationale,
            "action": action,
            "records_affected": records_affected,
            "details": details or ""
        })

    def _trim_string_whitespace(self):
        affected_count = 0
        obj_cols = self.df_cleaned.select_dtypes(include=["object"]).columns
        for col in obj_cols:
            before = self.df_cleaned[col].dropna().astype(str)
            after = before.str.strip()
            diff_mask = (before != after)
            count = diff_mask.sum()
            if count > 0:
                self.df_cleaned[col] = self.df_cleaned[col].apply(lambda x: x.strip() if isinstance(x, str) else x)
                affected_count += count
        if affected_count > 0:
            self._log_transformation(
                issue="Leading/trailing whitespace in text fields",
                rationale="Unstripped whitespace creates accidental unique categories and breaks exact matching.",
                action="Applied string striping across object columns.",
                records_affected=int(affected_count)
            )

    def _standardize_null_strings(self):
        from .loader import DataLoader
        affected_count = 0
        obj_cols = self.df_cleaned.select_dtypes(include=["object"]).columns
        for col in obj_cols:
            mask = self.df_cleaned[col].astype(str).str.strip().str.lower().isin(DataLoader.NULL_LIKE_STRINGS)
            count = mask.sum()
            if count > 0:
                self.df_cleaned.loc[mask, col] = np.nan
                affected_count += count
        if affected_count > 0:
            self._log_transformation(
                issue="Null-like text placeholders ('N/A', 'null', 'unknown')",
                rationale="Text representations of missing values interfere with pandas null checks.",
                action="Replaced null-like strings with standard NaN values.",
                records_affected=int(affected_count)
            )

    def _remove_duplicate_rows(self):
        before_count = len(self.df_cleaned)
        self.df_cleaned = self.df_cleaned.drop_duplicates().reset_index(drop=True)
        removed = before_count - len(self.df_cleaned)
        if removed > 0:
            self._log_transformation(
                issue=f"{removed} redundant duplicate rows found",
                rationale="Identical rows distort statistical distributions and bias model training.",
                action="Removed duplicate rows, keeping the first occurrence.",
                records_affected=int(removed)
            )

    def _standardize_categorical_values(self):
        # Map common gender variants (e.g. F/M -> Female/Male)
        cat_cols = self.metadata["column_buckets"]["categorical_columns"]
        total_affected = 0

        for col in cat_cols:
            if col not in self.df_cleaned.columns:
                continue
            if "gender" in col.lower() or "sex" in col.lower():
                s = self.df_cleaned[col].astype(str).str.strip().str.upper()
                mapping = {"F": "Female", "FEMALE": "Female", "M": "Male", "MALE": "Male"}
                new_s = s.map(mapping).fillna(self.df_cleaned[col])
                diff_mask = (self.df_cleaned[col] != new_s) & self.df_cleaned[col].notna()
                affected = diff_mask.sum()
                if affected > 0:
                    self.df_cleaned[col] = new_s
                    total_affected += affected

        if total_affected > 0:
            self._log_transformation(
                issue="Inconsistent categorical representations (e.g., 'F' vs 'Female')",
                rationale="Differing string formats split identical categories into separate groups.",
                action="Standardized category labels to canonical forms.",
                records_affected=int(total_affected)
            )

    def _cast_datetime_columns(self):
        dt_cols = self.metadata["column_buckets"]["datetime_columns"]
        total_affected = 0
        for col in dt_cols:
            if col in self.df_cleaned.columns and not pd.api.types.is_datetime64_any_dtype(self.df_cleaned[col]):
                converted = pd.to_datetime(self.df_cleaned[col], errors="coerce")
                valid_count = converted.notna().sum()
                if valid_count > 0:
                    self.df_cleaned[col] = converted
                    total_affected += valid_count
        if total_affected > 0:
            self._log_transformation(
                issue="Datetime columns stored as raw strings or objects",
                rationale="String dates cannot support time-series calculations or seasonal grouping.",
                action="Converted columns to datetime64[ns] data type.",
                records_affected=int(total_affected)
            )

    def _impute_numerical(self, strategy: str = "median"):
        num_cols = self.metadata["column_buckets"]["numerical_columns"]
        total_affected = 0
        for col in num_cols:
            if col not in self.df_cleaned.columns:
                continue
            missing_count = self.df_cleaned[col].isna().sum()
            if missing_count > 0:
                skew = self.df_cleaned[col].skew()
                if strategy == "median" or (abs(skew) > 0.5 and strategy != "mean"):
                    fill_val = float(self.df_cleaned[col].median())
                    chosen_strat = "median"
                else:
                    fill_val = float(self.df_cleaned[col].mean())
                    chosen_strat = "mean"

                self.df_cleaned[col] = self.df_cleaned[col].fillna(fill_val)
                total_affected += missing_count
                self._log_transformation(
                    issue=f"Missing values in numerical column '{col}' ({missing_count} missing)",
                    rationale=f"Missing numerical values prevent full statistical analysis. {chosen_strat.capitalize()} imputation selected due to distribution skewness={round(skew, 2) if pd.notna(skew) else 0}.",
                    action=f"Imputed missing entries with column {chosen_strat} ({round(fill_val, 2)}).",
                    records_affected=int(missing_count)
                )

    def _impute_categorical(self, strategy: str = "mode"):
        cat_cols = self.metadata["column_buckets"]["categorical_columns"]
        for col in cat_cols:
            if col not in self.df_cleaned.columns:
                continue
            missing_count = self.df_cleaned[col].isna().sum()
            if missing_count > 0:
                if strategy == "mode" and self.df_cleaned[col].nunique() > 0:
                    fill_val = self.df_cleaned[col].mode().iloc[0]
                else:
                    fill_val = "Unknown"
                self.df_cleaned[col] = self.df_cleaned[col].fillna(fill_val)
                self._log_transformation(
                    issue=f"Missing values in categorical column '{col}' ({missing_count} missing)",
                    rationale="Categorical missingness breaks grouping and matrix transforms.",
                    action=f"Imputed missing entries with '{fill_val}'.",
                    records_affected=int(missing_count)
                )
