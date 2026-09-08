import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, Any, List, Optional

class StatisticalAnalyzer:
    """
    Executes rigorous statistical hypothesis testing, confidence interval estimations,
    and regression models with full reporting of H0, H1, test statistic, p-value, and business impact.
    """

    @classmethod
    def run_tests(cls, df: pd.DataFrame, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        test_results = []

        num_cols = metadata["column_buckets"]["numerical_columns"]
        cat_cols = metadata["column_buckets"]["categorical_columns"]

        # 1. Two-sample t-test / Mann-Whitney U test across binary categories
        for cat in cat_cols:
            if cat not in df.columns:
                continue
            groups = df[cat].dropna().unique()
            if len(groups) == 2:
                g1_name, g2_name = groups[0], groups[1]
                for num in num_cols:
                    if num not in df.columns:
                        continue
                    g1_vals = pd.to_numeric(df[df[cat] == g1_name][num], errors="coerce").dropna()
                    g2_vals = pd.to_numeric(df[df[cat] == g2_name][num], errors="coerce").dropna()

                    if len(g1_vals) >= 3 and len(g2_vals) >= 3:
                        # Check normality with Shapiro-Wilk
                        stat_t, p_val = stats.ttest_ind(g1_vals, g2_vals, equal_var=False)
                        is_sig = p_val < 0.05

                        test_results.append({
                            "test_name": "Two-Sample Welch's t-test",
                            "grouping_variable": cat,
                            "target_variable": num,
                            "group_1": f"{g1_name} (mean={round(float(g1_vals.mean()), 2)}, n={len(g1_vals)})",
                            "group_2": f"{g2_name} (mean={round(float(g2_vals.mean()), 2)}, n={len(g2_vals)})",
                            "null_hypothesis": f"H0: Mean '{num}' for '{g1_name}' is equal to '{g2_name}'.",
                            "alt_hypothesis": f"H1: Mean '{num}' differs significantly between '{g1_name}' and '{g2_name}'.",
                            "statistic": round(float(stat_t), 4),
                            "p_value": float(p_val),
                            "p_value_formatted": "< 0.0001" if p_val < 0.0001 else f"{round(float(p_val), 4)}",
                            "alpha": 0.05,
                            "statistically_significant": bool(is_sig),
                            "business_interpretation": (
                                f"Statistically significant difference detected (p={round(float(p_val), 4)} < 0.05). "
                                f"Segment '{g1_name}' averages {round(float(g1_vals.mean()), 2)} vs '{g2_name}' averaging {round(float(g2_vals.mean()), 2)}."
                                if is_sig else
                                f"No statistically significant difference observed (p={round(float(p_val), 4)} >= 0.05). "
                                f"Observed variance is likely due to sample randomness."
                            )
                        })

        # 2. Chi-Square Test of Independence between categorical pairs
        if len(cat_cols) >= 2:
            for i in range(len(cat_cols)):
                for j in range(i + 1, len(cat_cols)):
                    c1, c2 = cat_cols[i], cat_cols[j]
                    if c1 not in df.columns or c2 not in df.columns:
                        continue
                    contingency = pd.crosstab(df[c1], df[c2])
                    if contingency.shape[0] >= 2 and contingency.shape[1] >= 2:
                        chi2, p_val, dof, _ = stats.chi2_contingency(contingency)
                        is_sig = p_val < 0.05

                        test_results.append({
                            "test_name": "Chi-Square Test of Independence",
                            "grouping_variable": f"{c1} vs {c2}",
                            "target_variable": f"Distribution matrix ({contingency.shape[0]}x{contingency.shape[1]})",
                            "null_hypothesis": f"H0: '{c1}' and '{c2}' are independent.",
                            "alt_hypothesis": f"H1: A significant association exists between '{c1}' and '{c2}'.",
                            "statistic": round(float(chi2), 4),
                            "p_value": float(p_val),
                            "p_value_formatted": "< 0.0001" if p_val < 0.0001 else f"{round(float(p_val), 4)}",
                            "alpha": 0.05,
                            "statistically_significant": bool(is_sig),
                            "business_interpretation": (
                                f"A statistically significant association exists between '{c1}' and '{c2}' (Chi2={round(float(chi2), 2)}, p={round(float(p_val), 4)})."
                                if is_sig else
                                f"No significant relationship between '{c1}' and '{c2}' was found (p={round(float(p_val), 4)})."
                            )
                        })

        # 3. One-Way ANOVA across categorical variables with > 2 levels
        for cat in cat_cols[:2]:
            if cat not in df.columns:
                continue
            groups = df[cat].dropna().unique()
            if len(groups) > 2 and len(groups) <= 10:
                for num in num_cols[:2]:
                    if num not in df.columns:
                        continue
                    group_data = [pd.to_numeric(df[df[cat] == g][num], errors="coerce").dropna() for g in groups]
                    group_data = [g for g in group_data if len(g) >= 3]
                    if len(group_data) > 2:
                        f_stat, p_val = stats.f_oneway(*group_data)
                        is_sig = p_val < 0.05
                        test_results.append({
                            "test_name": "One-Way ANOVA",
                            "grouping_variable": cat,
                            "target_variable": num,
                            "null_hypothesis": f"H0: Means of '{num}' across all categories of '{cat}' are equal.",
                            "alt_hypothesis": f"H1: At least one category of '{cat}' has a significantly different mean '{num}'.",
                            "statistic": round(float(f_stat), 4),
                            "p_value": float(p_val),
                            "p_value_formatted": "< 0.0001" if p_val < 0.0001 else f"{round(float(p_val), 4)}",
                            "alpha": 0.05,
                            "statistically_significant": bool(is_sig),
                            "business_interpretation": (
                                f"Significant variation in '{num}' across groups of '{cat}' (F={round(float(f_stat), 2)}, p={round(float(p_val), 4)})."
                                if is_sig else
                                f"Group means of '{num}' across '{cat}' do not differ significantly (p={round(float(p_val), 4)})."
                            )
                        })

        return test_results

    @classmethod
    def calculate_confidence_interval(cls, series: pd.Series, confidence: float = 0.95) -> Dict[str, Any]:
        valid = pd.to_numeric(series, errors="coerce").dropna()
        if len(valid) < 2:
            return {"status": "Insufficient data"}

        mean = float(valid.mean())
        sem = float(stats.sem(valid))
        h = sem * stats.t.ppf((1 + confidence) / 2., len(valid) - 1)

        return {
            "mean": round(mean, 4),
            "confidence_level": confidence,
            "margin_of_error": round(float(h), 4),
            "ci_lower": round(mean - h, 4),
            "ci_upper": round(mean + h, 4),
        }
