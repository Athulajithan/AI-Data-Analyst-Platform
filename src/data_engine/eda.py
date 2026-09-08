import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, Any, List, Tuple, Optional

class EDAEngine:
    """
    Executes Exploratory Data Analysis including correlation matrix (Pearson/Spearman),
    segmentation breakdowns, time-series growth metrics, and statistical anomaly detection.
    """

    @classmethod
    def run_eda(cls, df: pd.DataFrame, metadata: Dict[str, Any]) -> Dict[str, Any]:
        num_cols = metadata["column_buckets"]["numerical_columns"]
        cat_cols = metadata["column_buckets"]["categorical_columns"]
        dt_cols = metadata["column_buckets"]["datetime_columns"]

        correlations = cls.compute_correlations(df, num_cols)
        segmentations = cls.compute_segmentations(df, cat_cols, num_cols)
        trends = cls.compute_time_trends(df, dt_cols, num_cols)
        anomalies = cls.detect_anomalies(df, num_cols)

        return {
            "correlations": correlations,
            "segmentations": segmentations,
            "trends": trends,
            "anomalies": anomalies,
        }

    @classmethod
    def compute_correlations(cls, df: pd.DataFrame, num_cols: List[str]) -> Dict[str, Any]:
        if len(num_cols) < 2:
            return {"status": "Fewer than 2 numerical columns available for correlation analysis."}

        clean_num = df[num_cols].dropna()
        if len(clean_num) < 3:
            return {"status": "Insufficient valid rows for correlation calculation."}

        pearson = clean_num.corr(method="pearson").round(4).to_dict()
        spearman = clean_num.corr(method="spearman").round(4).to_dict()

        # Extract top pairs
        top_pairs = []
        cols = list(clean_num.columns)
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                c1, c2 = cols[i], cols[j]
                val_p = pearson[c1][c2]
                val_s = spearman[c1][c2]
                if abs(val_p) >= 0.2:
                    top_pairs.append({
                        "var1": c1,
                        "var2": c2,
                        "pearson": val_p,
                        "spearman": val_s,
                        "direction": "Positive" if val_p > 0 else "Negative",
                        "strength": "Strong" if abs(val_p) > 0.7 else ("Moderate" if abs(val_p) > 0.4 else "Weak")
                    })

        top_pairs.sort(key=lambda x: abs(x["pearson"]), reverse=True)

        return {
            "pearson_matrix": pearson,
            "spearman_matrix": spearman,
            "top_correlated_pairs": top_pairs
        }

    @classmethod
    def compute_segmentations(cls, df: pd.DataFrame, cat_cols: List[str], num_cols: List[str]) -> Dict[str, Any]:
        results = {}
        if not cat_cols or not num_cols:
            return results

        # Limit to top 3 categorical and top 3 numerical cols to avoid bloat
        target_cats = cat_cols[:3]
        target_nums = num_cols[:3]

        for cat in target_cats:
            if cat not in df.columns:
                continue
            for num in target_nums:
                if num not in df.columns:
                    continue
                grp = df.groupby(cat)[num].agg(["count", "sum", "mean", "std", "median"]).reset_index()
                grp["mean"] = grp["mean"].round(2)
                grp["sum"] = grp["sum"].round(2)
                grp["std"] = grp["std"].fillna(0).round(2)
                grp = grp.sort_values(by="sum", ascending=False)
                
                key = f"{cat}__vs__{num}"
                results[key] = {
                    "category_col": cat,
                    "measure_col": num,
                    "breakdown": grp.to_dict(orient="records"),
                    "top_segment": str(grp.iloc[0][cat]) if len(grp) > 0 else None,
                    "bottom_segment": str(grp.iloc[-1][cat]) if len(grp) > 0 else None,
                }
        return results

    @classmethod
    def compute_time_trends(cls, df: pd.DataFrame, dt_cols: List[str], num_cols: List[str]) -> Dict[str, Any]:
        results = {}
        if not dt_cols or not num_cols:
            return results

        dt_col = dt_cols[0]
        if dt_col not in df.columns:
            return results

        temp_df = df.copy()
        temp_df[dt_col] = pd.to_datetime(temp_df[dt_col], errors="coerce")
        temp_df = temp_df.dropna(subset=[dt_col]).sort_values(by=dt_col)

        if len(temp_df) < 2:
            return results

        for num in num_cols[:3]:
            if num not in temp_df.columns:
                continue
            # Monthly or Daily aggregation based on date span
            days_span = (temp_df[dt_col].max() - temp_df[dt_col].min()).days
            freq = "ME" if days_span > 60 else "D"
            
            resampled = temp_df.set_index(dt_col)[num].resample(freq).sum().reset_index()
            resampled[num] = resampled[num].round(2)
            
            if len(resampled) >= 2:
                first_val = float(resampled[num].iloc[0])
                last_val = float(resampled[num].iloc[-1])
                abs_change = round(last_val - first_val, 2)
                pct_change = round((abs_change / first_val * 100), 2) if first_val != 0 else 0.0

                results[f"{dt_col}__vs__{num}"] = {
                    "time_col": dt_col,
                    "measure_col": num,
                    "granularity": "Monthly" if freq == "ME" else "Daily",
                    "initial_value": first_val,
                    "latest_value": last_val,
                    "absolute_change": abs_change,
                    "percentage_growth": pct_change,
                    "trend_data": resampled.astype(str).to_dict(orient="records")
                }

        return results

    @classmethod
    def detect_anomalies(cls, df: pd.DataFrame, num_cols: List[str]) -> Dict[str, Any]:
        results = {}
        for col in num_cols:
            if col not in df.columns:
                continue
            series = pd.to_numeric(df[col], errors="coerce").dropna()
            if len(series) < 5:
                continue

            # IQR Method
            q25, q75 = series.quantile(0.25), series.quantile(0.75)
            iqr = q75 - q25
            lower_bound = q25 - 1.5 * iqr
            upper_bound = q75 + 1.5 * iqr

            outliers_iqr = series[(series < lower_bound) | (series > upper_bound)]

            # Modified Z-Score Method (MAD)
            median = series.median()
            mad = (series - median).abs().median()
            if mad != 0:
                mod_z = 0.6745 * (series - median).abs() / mad
                outliers_mod_z = series[mod_z > 3.5]
            else:
                outliers_mod_z = pd.Series(dtype=float)

            results[col] = {
                "iqr_outlier_count": len(outliers_iqr),
                "iqr_outlier_pct": round(len(outliers_iqr) / len(series) * 100, 2),
                "lower_bound": round(float(lower_bound), 2),
                "upper_bound": round(float(upper_bound), 2),
                "modified_z_outlier_count": len(outliers_mod_z),
                "sample_outlier_values": outliers_iqr.head(5).tolist() if len(outliers_iqr) > 0 else [],
                "impact_summary": f"Detected {len(outliers_iqr)} statistical outliers via IQR method."
            }

        return results
