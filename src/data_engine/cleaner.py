import pandas as pd
import numpy as np
import re
from typing import Dict, Any, Tuple, List, Optional

class DataCleaner:
    """
    Cleans datasets non-destructively, preserving `df_original` and generating a
    detailed `TransformationLog` for every cleaning operation.

    Cleaning Pipeline Order:
      0. String-to-Numeric standardization   ← FIRST (prevents TypeError)
      1. Whitespace trim
      2. Null-string standardization
      3. Duplicate removal
      4. Categorical standardization
      5. Datetime casting
      6. Numerical imputation
      7. Categorical imputation
      8. Outlier detection & handling
    """

    # Pattern: optional sign, optional currency, digits with optional commas, optional decimal, optional %
    _NUMERIC_STR_RE = re.compile(
        r"^[+\-]?[\$\£\€\¥\₹]?\s*[\d,]+(\.\d+)?\s*%?$"
    )

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
        standardize_numeric_strings: bool = True,
        detect_outliers: bool = True,
        outlier_strategy: str = "clip",            # "clip" or "report"
    ) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
        """
        Executes an end-to-end automated cleaning pipeline.
        String-to-numeric standardization runs FIRST to prevent TypeError in all subsequent steps.
        """
        # ── Step 0: Convert formatted numeric strings → actual numerics (MUST be first) ──
        if standardize_numeric_strings:
            self._standardize_numeric_strings()
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
        if detect_outliers:
            self._detect_and_handle_outliers(strategy=outlier_strategy)

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

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 0 — String-to-Numeric Standardization
    # Converts '$89,765', '45,000', '23.5%', '1 234' → actual float64 / int64
    # This MUST run before whitespace trim, null standardization, and imputation.
    # ─────────────────────────────────────────────────────────────────────────
    def _standardize_numeric_strings(self):
        """
        Detects object-dtype columns where the majority of non-null values are
        numeric strings with formatting characters (currency symbols, commas,
        percent signs, spaces) and converts them to proper numeric dtype.
        """
        converted_cols = []
        obj_cols = self.df_cleaned.select_dtypes(include=["object"]).columns

        for col in obj_cols:
            non_null = self.df_cleaned[col].dropna().astype(str).str.strip()
            if len(non_null) == 0:
                continue

            # Quick test: strip formatting chars and try numeric conversion on a sample
            sample = non_null.head(100)
            cleaned_sample = (
                sample
                .str.replace(r"[\$£€¥₹,\s%]", "", regex=True)  # strip currency/commas/spaces/percent
                .str.replace(r"\((\d+\.?\d*)\)", r"-\1", regex=True)  # (123) → -123  (accounting negative)
                .str.replace(r"[^\d.\-+]", "", regex=True)  # remove any remaining non-numeric chars
            )
            numeric_sample = pd.to_numeric(cleaned_sample, errors="coerce")
            hit_rate = numeric_sample.notna().mean()

            # Threshold: if ≥ 75% of sample values parse as numeric → convert the whole column
            if hit_rate >= 0.75:
                full_cleaned = (
                    self.df_cleaned[col]
                    .astype(str)
                    .str.strip()
                    .str.replace(r"[\$£€¥₹,\s%]", "", regex=True)
                    .str.replace(r"\((\d+\.?\d*)\)", r"-\1", regex=True)
                    .str.replace(r"[^\d.\-+]", "", regex=True)
                )
                numeric_col = pd.to_numeric(full_cleaned, errors="coerce")
                successfully_converted = numeric_col.notna().sum()

                if successfully_converted > 0:
                    # Preserve NaN for values that were originally NaN
                    original_null_mask = self.df_cleaned[col].isna()
                    self.df_cleaned[col] = numeric_col
                    self.df_cleaned.loc[original_null_mask, col] = np.nan

                    # Downcast to int if no fractional part
                    if (numeric_col.dropna() == numeric_col.dropna().astype(int)).all():
                        try:
                            self.df_cleaned[col] = pd.to_numeric(self.df_cleaned[col], downcast="integer")
                        except Exception:
                            pass

                    converted_cols.append({
                        "column": col,
                        "records_converted": int(successfully_converted),
                    })
                    # Update metadata column buckets to reflect new numeric type
                    num_cols = self.metadata["column_buckets"]["numerical_columns"]
                    cat_cols = self.metadata["column_buckets"]["categorical_columns"]
                    text_cols = self.metadata["column_buckets"]["text_columns"]
                    if col not in num_cols:
                        num_cols.append(col)
                    for remove_list in [cat_cols, text_cols]:
                        if col in remove_list:
                            remove_list.remove(col)

        if converted_cols:
            col_summary = ", ".join(
                [f"'{c['column']}' ({c['records_converted']} values)" for c in converted_cols]
            )
            self._log_transformation(
                issue=f"Formatted numeric strings detected in {len(converted_cols)} column(s)",
                rationale=(
                    "Columns containing values like '$89,765', '45,000', or '23.5%' are stored as "
                    "object dtype and cause TypeError in all statistical computations. "
                    "Stripping currency symbols, commas, and percent signs before casting to float64/int64 "
                    "is required before any analysis can proceed."
                ),
                action=f"Converted to numeric dtype: {col_summary}",
                records_affected=sum(c["records_converted"] for c in converted_cols),
                details=str(converted_cols)
            )

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 8 — Outlier Detection & Handling
    # ─────────────────────────────────────────────────────────────────────────
    def _detect_and_handle_outliers(self, strategy: str = "clip"):
        """
        Detects outliers in numerical columns using IQR (primary) and Z-score (secondary).
        - IQR: values < Q1 - 1.5*IQR or > Q3 + 1.5*IQR are flagged
        - Z-score: |z| > 3.5 flagged for near-normal distributions
        - Strategy 'clip': clips to [lower_fence, upper_fence]
        - Strategy 'report': only logs, does not modify values
        """
        num_cols = self.metadata["column_buckets"]["numerical_columns"]
        total_outliers = 0
        outlier_details = []

        for col in num_cols:
            if col not in self.df_cleaned.columns:
                continue
            series = pd.to_numeric(self.df_cleaned[col], errors="coerce").dropna()
            if len(series) < 10:
                continue

            q1 = series.quantile(0.25)
            q3 = series.quantile(0.75)
            iqr = q3 - q1

            if iqr == 0:
                # Zero IQR means constant column → use Z-score instead
                mean_v = series.mean()
                std_v = series.std()
                if std_v == 0:
                    continue
                z_scores = ((series - mean_v) / std_v).abs()
                outlier_mask = self.df_cleaned[col].notna() & (
                    pd.to_numeric(self.df_cleaned[col], errors="coerce").apply(
                        lambda x: abs((x - mean_v) / std_v) > 3.5 if pd.notna(x) else False
                    )
                )
                lower_fence = mean_v - 3.5 * std_v
                upper_fence = mean_v + 3.5 * std_v
                method = "Z-score (|z|>3.5)"
            else:
                lower_fence = q1 - 1.5 * iqr
                upper_fence = q3 + 1.5 * iqr
                numeric_col = pd.to_numeric(self.df_cleaned[col], errors="coerce")
                outlier_mask = numeric_col.notna() & (
                    (numeric_col < lower_fence) | (numeric_col > upper_fence)
                )
                method = "IQR (±1.5×IQR)"

            outlier_count = int(outlier_mask.sum())
            if outlier_count == 0:
                continue

            total_outliers += outlier_count
            outlier_pct = round(outlier_count / len(series) * 100, 1)
            outlier_details.append(
                f"'{col}': {outlier_count} outliers ({outlier_pct}%) via {method} "
                f"[fence: {round(lower_fence, 2)} – {round(upper_fence, 2)}]"
            )

            if strategy == "clip":
                self.df_cleaned[col] = pd.to_numeric(self.df_cleaned[col], errors="coerce").clip(
                    lower=lower_fence, upper=upper_fence
                )

        if outlier_details:
            action_desc = (
                f"Clipped {total_outliers} extreme values to IQR/Z-score fences."
                if strategy == "clip"
                else f"Detected {total_outliers} outliers (report-only mode, no modification)."
            )
            self._log_transformation(
                issue=f"Statistical outliers detected in {len(outlier_details)} numerical column(s)",
                rationale=(
                    "Extreme values distort means, inflate standard deviations, skew ML model training, "
                    "and produce misleading KPI aggregations. IQR fencing is robust to non-normal distributions."
                ),
                action=action_desc,
                records_affected=total_outliers,
                details=" | ".join(outlier_details)
            )

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
