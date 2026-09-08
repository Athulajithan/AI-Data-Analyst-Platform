import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple

class DataProfiler:
    """
    Computes comprehensive mathematical profiling metrics across numerical,
    categorical, and datetime features, and produces an empirical Data Quality Score.
    """

    @classmethod
    def profile_dataset(cls, df: pd.DataFrame, metadata: Dict[str, Any]) -> Dict[str, Any]:
        num_profile = cls._profile_numerical(df, metadata["column_buckets"]["numerical_columns"])
        cat_profile = cls._profile_categorical(df, metadata["column_buckets"]["categorical_columns"])
        dt_profile = cls._profile_datetime(df, metadata["column_buckets"]["datetime_columns"])
        quality_score, score_breakdown = cls.calculate_quality_score(df, metadata)

        return {
            "numerical_summary": num_profile,
            "categorical_summary": cat_profile,
            "datetime_summary": dt_profile,
            "data_quality_score": quality_score,
            "score_breakdown": score_breakdown,
        }

    @classmethod
    def _profile_numerical(cls, df: pd.DataFrame, num_cols: List[str]) -> Dict[str, Dict[str, Any]]:
        results = {}
        for col in num_cols:
            if col not in df.columns:
                continue
            series = pd.to_numeric(df[col], errors="coerce")
            valid = series.dropna()
            total = len(series)
            missing = total - len(valid)

            if len(valid) == 0:
                results[col] = {"status": "Empty or non-numeric"}
                continue

            q25 = float(valid.quantile(0.25))
            q50 = float(valid.median())
            q75 = float(valid.quantile(0.75))
            iqr = q75 - q25

            results[col] = {
                "count": int(total),
                "valid_count": int(len(valid)),
                "missing_count": int(missing),
                "missing_pct": round(missing / total * 100, 2) if total > 0 else 0.0,
                "mean": round(float(valid.mean()), 4),
                "median": round(q50, 4),
                "std": round(float(valid.std()), 4) if len(valid) > 1 else 0.0,
                "min": round(float(valid.min()), 4),
                "max": round(float(valid.max()), 4),
                "q25": round(q25, 4),
                "q50": round(q50, 4),
                "q75": round(q75, 4),
                "iqr": round(iqr, 4),
                "skewness": round(float(valid.skew()), 4) if len(valid) > 2 else 0.0,
                "unique_count": int(valid.nunique()),
            }
        return results

    @classmethod
    def _profile_categorical(cls, df: pd.DataFrame, cat_cols: List[str]) -> Dict[str, Dict[str, Any]]:
        results = {}
        for col in cat_cols:
            if col not in df.columns:
                continue
            series = df[col].astype(str)
            total = len(series)
            counts = series.value_counts(dropna=False)
            unique_count = len(counts)

            top_5 = counts.head(5).to_dict()
            top_5_pct = {k: round(v / total * 100, 2) for k, v in top_5.items()}

            # Rare categories (< 1% frequency)
            rare = counts[counts / total < 0.01]
            rare_count = len(rare)

            results[col] = {
                "total_count": total,
                "unique_count": unique_count,
                "top_categories": top_5,
                "top_categories_pct": top_5_pct,
                "most_frequent": str(counts.index[0]) if len(counts) > 0 else None,
                "most_frequent_count": int(counts.iloc[0]) if len(counts) > 0 else 0,
                "rare_categories_count": rare_count,
            }
        return results

    @classmethod
    def _profile_datetime(cls, df: pd.DataFrame, dt_cols: List[str]) -> Dict[str, Dict[str, Any]]:
        results = {}
        for col in dt_cols:
            if col not in df.columns:
                continue
            dt_series = pd.to_datetime(df[col], errors="coerce")
            valid = dt_series.dropna()
            total = len(dt_series)
            missing = total - len(valid)

            if len(valid) == 0:
                results[col] = {"status": "No valid datetime records"}
                continue

            min_date = valid.min()
            max_date = valid.max()
            date_range = (max_date - min_date).days

            results[col] = {
                "total_count": total,
                "valid_count": len(valid),
                "missing_count": missing,
                "missing_pct": round(missing / total * 100, 2) if total > 0 else 0.0,
                "min_date": str(min_date),
                "max_date": str(max_date),
                "date_range_days": date_range,
            }
        return results

    @classmethod
    def calculate_quality_score(cls, df: pd.DataFrame, metadata: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
        """
        Calculates an objective Data Quality Score from 0 to 100 with clear component penalties.
        """
        base_score = 100.0
        penalties = {}

        # 1. Missingness penalty (max -30 points)
        total_cells = df.size
        total_missing = df.isna().sum().sum()
        missing_pct = (total_missing / total_cells * 100) if total_cells > 0 else 0.0
        missing_penalty = min(30.0, missing_pct * 1.5)
        penalties["missing_data"] = {
            "penalty": round(missing_penalty, 2),
            "reason": f"{round(missing_pct, 2)}% of total cell entries are missing."
        }

        # 2. Duplicate rows penalty (max -20 points)
        dup_pct = metadata.get("duplicate_rows_pct", 0.0)
        dup_penalty = min(20.0, dup_pct * 2.0)
        penalties["duplicate_rows"] = {
            "penalty": round(dup_penalty, 2),
            "reason": f"{dup_pct}% of dataset rows are exact duplicates."
        }

        # 3. Constant / Zero-variance columns penalty (max -15 points)
        constant_cols = metadata["column_buckets"].get("constant_columns", [])
        if constant_cols:
            const_penalty = min(15.0, len(constant_cols) * 5.0)
            penalties["constant_columns"] = {
                "penalty": round(const_penalty, 2),
                "reason": f"{len(constant_cols)} columns contain constant/uninformative values: {constant_cols}."
            }

        # 4. PII Exposure penalty (max -10 points)
        pii = metadata.get("pii_detected", {})
        if pii:
            pii_penalty = min(10.0, len(pii) * 5.0)
            penalties["pii_exposure"] = {
                "penalty": round(pii_penalty, 2),
                "reason": f"Potential unmasked PII detected in columns: {list(pii.keys())}."
            }

        # 5. Null-like string penalty (max -10 points)
        null_like_total = sum(v["null_like_count"] for v in metadata["column_analysis"].values())
        if null_like_total > 0:
            null_like_pct = (null_like_total / total_cells * 100) if total_cells > 0 else 0.0
            null_penalty = min(10.0, null_like_pct * 2.0)
            penalties["null_like_strings"] = {
                "penalty": round(null_penalty, 2),
                "reason": f"{null_like_total} unparsed null-like strings (e.g. 'N/A', 'null') found."
            }

        total_penalty = sum(v["penalty"] for v in penalties.values())
        final_score = max(0.0, round(base_score - total_penalty, 2))

        return final_score, penalties
