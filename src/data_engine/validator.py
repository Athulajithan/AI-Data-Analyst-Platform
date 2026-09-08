import pandas as pd
from typing import Dict, Any, List

class DataValidator:
    """
    Validates post-cleaning metrics against original baseline state and generates
    a transparent, evidence-based Validation Summary.
    """

    @classmethod
    def validate_cleaning(
        cls,
        df_original: pd.DataFrame,
        df_cleaned: pd.DataFrame,
        transformation_log: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        
        orig_rows, orig_cols = df_original.shape
        clean_rows, clean_cols = df_cleaned.shape
        
        orig_nulls = int(df_original.isna().sum().sum())
        clean_nulls = int(df_cleaned.isna().sum().sum())
        
        orig_dups = int(df_original.duplicated().sum())
        clean_dups = int(df_cleaned.duplicated().sum())

        records_removed = orig_rows - clean_rows
        missing_resolved = orig_nulls - clean_nulls
        duplicates_resolved = orig_dups - clean_dups
        
        # Count total records modified across non-removed rows
        records_modified = sum(t.get("records_affected", 0) for t in transformation_log if "Removed" not in t.get("action", ""))

        remaining_issues = []
        if clean_nulls > 0:
            remaining_issues.append(f"{clean_nulls} null entries remaining across columns.")
        if clean_dups > 0:
            remaining_issues.append(f"{clean_dups} duplicate rows remaining.")
        if len(remaining_issues) == 0:
            remaining_issues.append("None. All detected structural issues resolved.")

        summary_table = {
            "metric": [
                "Total Rows",
                "Total Columns",
                "Total Missing Values",
                "Duplicate Rows",
                "Data Quality Status"
            ],
            "before_cleaning": [
                orig_rows,
                orig_cols,
                orig_nulls,
                orig_dups,
                "Raw / Unvalidated"
            ],
            "after_cleaning": [
                clean_rows,
                clean_cols,
                clean_nulls,
                clean_dups,
                "Validated & Prepared"
            ],
            "delta_change": [
                f"-{records_removed}" if records_removed > 0 else "0",
                f"{clean_cols - orig_cols}",
                f"-{missing_resolved}",
                f"-{duplicates_resolved}",
                "Cleaned" if len(remaining_issues) == 1 and "None" in remaining_issues[0] else "Partially Cleaned"
            ]
        }

        return {
            "summary_table": summary_table,
            "records_removed": records_removed,
            "records_modified": records_modified,
            "missing_values_resolved": missing_resolved,
            "duplicates_resolved": duplicates_resolved,
            "remaining_nulls": clean_nulls,
            "remaining_duplicates": clean_dups,
            "remaining_issues": remaining_issues,
            "is_fully_clean": clean_nulls == 0 and clean_dups == 0
        }
