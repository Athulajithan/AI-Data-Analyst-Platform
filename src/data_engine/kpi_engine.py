import pandas as pd
import numpy as np
from typing import Dict, Any, List

class KPIEngine:
    """
    Computes industry-standard business KPIs across Sales, Marketing, Customer,
    Product, and Operational domain categories strictly from verifiable data.
    """

    @classmethod
    def calculate_kpis(cls, df: pd.DataFrame, metadata: Dict[str, Any]) -> Dict[str, Any]:
        kpis = {}

        num_cols = [c.lower() for c in metadata["column_buckets"]["numerical_columns"]]
        cat_cols = [c.lower() for c in metadata["column_buckets"]["categorical_columns"]]
        dt_cols = metadata["column_buckets"]["datetime_columns"]

        orig_col_map = {c.lower(): c for c in df.columns}

        # 1. Sales & Revenue KPIs
        revenue_col = next((orig_col_map[c] for c in num_cols if c in ["spend", "total_spend", "revenue", "sales", "price", "amount", "total_amount"]), None)
        units_col = next((orig_col_map[c] for c in num_cols if c in ["units", "units_sold", "quantity", "qty"]), None)
        order_col = next((orig_col_map[c] for c in orig_col_map if c in ["transaction_id", "order_id", "trans_id"]), None)

        if revenue_col:
            rev_series = pd.to_numeric(df[revenue_col], errors="coerce").dropna()
            total_rev = float(rev_series.sum())
            avg_order_val = float(rev_series.mean())
            max_sale = float(rev_series.max())

            kpis["Sales & Revenue"] = {
                "Total Revenue": f"${round(total_rev, 2):,}",
                "Average Order Value (AOV)": f"${round(avg_order_val, 2):,}",
                "Max Transaction Value": f"${round(max_sale, 2):,}",
                "Total Transactions Recorded": len(rev_series),
            }
            if units_col and units_col in df.columns:
                units_series = pd.to_numeric(df[units_col], errors="coerce").dropna()
                kpis["Sales & Revenue"]["Total Units Sold"] = int(units_series.sum())

        # 2. Customer KPIs
        cust_col = next((orig_col_map[c] for c in orig_col_map if c in ["customer_id", "cust_id", "user_id", "client_id"]), None)
        if cust_col:
            unique_custs = df[cust_col].nunique()
            total_orders = len(df)
            orders_per_cust = df.groupby(cust_col).size()
            repeat_customers = (orders_per_cust > 1).sum()
            repeat_rate = round((repeat_customers / unique_custs * 100), 2) if unique_custs > 0 else 0.0

            kpis["Customer Analytics"] = {
                "Unique Customer Count": unique_custs,
                "Repeat Customer Count": int(repeat_customers),
                "Repeat Purchase Rate": f"{repeat_rate}%",
                "Average Orders per Customer": round(float(orders_per_cust.mean()), 2),
            }

        # 3. Product & Operations KPIs
        return_col = next((orig_col_map[c] for c in orig_col_map if c in ["is_returned", "returned", "return_flag", "is_defective", "churn"]), None)
        if return_col:
            ret_series = df[return_col].astype(str).str.lower()
            yes_count = ret_series.isin(["yes", "true", "1"]).sum()
            return_rate = round((yes_count / len(df) * 100), 2) if len(df) > 0 else 0.0

            kpis["Operational Quality"] = {
                "Return / Exception Count": int(yes_count),
                "Return / Exception Rate": f"{return_rate}%",
                "Successful Fulfillments": len(df) - int(yes_count),
            }

        return kpis
