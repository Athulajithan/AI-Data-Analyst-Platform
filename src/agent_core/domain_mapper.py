"""
Domain Mapper — AI Data Analyst Platform 11.0
==============================================
Automatically infers the business domain from dataset column names, data patterns,
and KPI distributions. Maps domains to actionable ML goals, KPI targets, and
recommended analyses.

Supported Domains:
  E-Commerce / Retail    — Revenue, Orders, Products, Returns
  Finance & Banking      — Transactions, Accounts, Credit, Risk
  Healthcare / Clinical  — Patients, Diagnoses, Medications, Outcomes
  Human Resources / HR   — Employees, Salary, Attrition, Performance
  Marketing & CRM        — Leads, Campaigns, Conversion, Engagement
  Operations / Logistics — Shipments, Inventory, Lead Time, Efficiency
  General / Unknown      — Fallback for unmatched datasets
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import re


# ──────────────────────────────────────────────────────────────────────────────
# DOMAIN SIGNAL DEFINITIONS
# Each domain has:
#   - keywords: column name substrings (case-insensitive) that signal this domain
#   - weight: importance weight per keyword match (more specific = higher)
#   - ml_goals: list of {title, description, model_type, priority}
#   - kpi_targets: list of KPI names to compute
#   - recommended_analyses: text list of suggested analyses
#   - use_cases: business use cases that justify the analysis
# ──────────────────────────────────────────────────────────────────────────────

DOMAIN_SIGNALS: Dict[str, Dict[str, Any]] = {
    "E-Commerce & Retail": {
        "keywords": {
            "order": 3, "product": 3, "revenue": 3, "sale": 3, "purchase": 3,
            "cart": 3, "checkout": 3, "sku": 3, "price": 2, "discount": 2,
            "coupon": 2, "return": 2, "refund": 2, "category": 1, "brand": 2,
            "inventory": 2, "stock": 2, "customer": 1, "aov": 3, "ltv": 3,
            "basket": 3, "shipping": 2, "delivery": 2, "retailer": 3,
        },
        "icon": "🛒",
        "ml_goals": [
            {
                "title": "Customer Lifetime Value (LTV) Prediction",
                "description": "Predict future revenue per customer using purchase history, frequency, and recency.",
                "model_type": "Regression (XGBoost / Random Forest)",
                "priority": "HIGH",
                "business_value": "Enables dynamic budget allocation for customer acquisition vs. retention."
            },
            {
                "title": "Churn Prediction",
                "description": "Identify customers likely to stop purchasing within the next 30/60/90 days.",
                "model_type": "Binary Classification (Logistic Regression / XGBoost)",
                "priority": "HIGH",
                "business_value": "Proactive retention campaigns can reduce churn by 15–25%."
            },
            {
                "title": "Market Basket Analysis (MBA)",
                "description": "Discover which products are frequently bought together using association rules.",
                "model_type": "Apriori / FP-Growth",
                "priority": "MEDIUM",
                "business_value": "Powers cross-sell recommendations — typically +8–12% revenue uplift."
            },
            {
                "title": "Demand Forecasting",
                "description": "Forecast future product demand to optimize inventory and reduce stockouts.",
                "model_type": "Time-Series (ARIMA / Prophet / LSTM)",
                "priority": "HIGH",
                "business_value": "Reduces overstock costs and lost-sales events."
            },
            {
                "title": "RFM Customer Segmentation",
                "description": "Segment customers by Recency, Frequency, and Monetary value for targeted marketing.",
                "model_type": "Clustering (K-Means / DBSCAN)",
                "priority": "MEDIUM",
                "business_value": "Enables personalized campaigns with measurably higher conversion rates."
            },
        ],
        "kpi_targets": ["Total Revenue", "AOV", "Return Rate", "Conversion Rate", "Customer Acquisition Cost"],
        "recommended_analyses": [
            "Revenue trend over time (daily/weekly/monthly)",
            "Top 20% products contributing to 80% revenue (Pareto analysis)",
            "Category-wise revenue breakdown",
            "Return/refund rate by product, category, and region",
            "Customer segmentation by purchase frequency",
            "Seasonal demand patterns",
        ],
        "use_cases": [
            "Personalized product recommendations",
            "Dynamic pricing optimization",
            "Inventory replenishment automation",
            "Customer winback campaigns",
        ],
    },

    "Finance & Banking": {
        "keywords": {
            "transaction": 3, "account": 3, "balance": 3, "credit": 3, "debit": 3,
            "loan": 3, "interest": 3, "fraud": 3, "payment": 2, "bank": 3,
            "risk": 2, "default": 3, "portfolio": 3, "asset": 2, "liability": 2,
            "income": 2, "expense": 2, "claim": 2, "premium": 2, "policy": 2,
            "investment": 3, "return_rate": 3, "apr": 3, "mortgage": 3,
        },
        "icon": "💰",
        "ml_goals": [
            {
                "title": "Fraud Detection",
                "description": "Detect anomalous transactions indicating fraudulent activity in real-time.",
                "model_type": "Anomaly Detection (Isolation Forest / AutoEncoder)",
                "priority": "CRITICAL",
                "business_value": "Prevents direct financial losses and regulatory penalties."
            },
            {
                "title": "Credit Default Prediction",
                "description": "Predict probability of loan default for credit risk scoring.",
                "model_type": "Binary Classification (Logistic Regression / XGBoost)",
                "priority": "HIGH",
                "business_value": "Reduces non-performing loans and improves capital allocation."
            },
            {
                "title": "Customer Churn & Attrition",
                "description": "Predict which banking customers are likely to close accounts.",
                "model_type": "Binary Classification",
                "priority": "MEDIUM",
                "business_value": "Enables proactive retention for high-value account holders."
            },
            {
                "title": "Transaction Categorization",
                "description": "Automatically classify transactions into spending categories.",
                "model_type": "Multi-Class Classification (BERT / TF-IDF + SVM)",
                "priority": "MEDIUM",
                "business_value": "Powers personal finance management features and spending insights."
            },
        ],
        "kpi_targets": ["Total Transaction Volume", "Fraud Rate", "Default Rate", "Net Interest Margin"],
        "recommended_analyses": [
            "Transaction volume trends (daily/monthly)",
            "Fraud pattern analysis by amount, time, merchant category",
            "Customer segment risk profiling",
            "Portfolio concentration analysis",
        ],
        "use_cases": [
            "Real-time fraud alert system",
            "Automated credit scoring pipeline",
            "Regulatory compliance reporting",
            "Customer financial health scoring",
        ],
    },

    "Healthcare & Clinical": {
        "keywords": {
            "patient": 3, "diagnosis": 3, "medication": 3, "treatment": 3,
            "hospital": 3, "clinical": 3, "disease": 3, "symptom": 3,
            "icd": 3, "cpt": 3, "readmission": 3, "mortality": 3,
            "age": 1, "gender": 1, "bmi": 3, "blood": 2, "pressure": 2,
            "glucose": 2, "cholesterol": 2, "lab": 2, "test": 1,
            "outcome": 2, "survival": 3, "comorbidity": 3,
        },
        "icon": "🏥",
        "ml_goals": [
            {
                "title": "Readmission Risk Prediction",
                "description": "Predict probability of 30-day readmission to target discharge planning.",
                "model_type": "Binary Classification (XGBoost / Logistic Regression)",
                "priority": "HIGH",
                "business_value": "Reduces readmission penalties and improves patient outcomes."
            },
            {
                "title": "Mortality / Adverse Event Prediction",
                "description": "Identify high-risk patients in ICU or post-surgery settings.",
                "model_type": "Survival Analysis (Cox PH / DeepSurv)",
                "priority": "CRITICAL",
                "business_value": "Enables early clinical intervention to reduce adverse outcomes."
            },
            {
                "title": "Disease Diagnosis Classification",
                "description": "Classify patient records into disease categories from clinical features.",
                "model_type": "Multi-Class Classification",
                "priority": "HIGH",
                "business_value": "Supports clinical decision support tools."
            },
        ],
        "kpi_targets": ["Readmission Rate", "Average Length of Stay", "Mortality Rate", "Bed Utilization"],
        "recommended_analyses": [
            "Patient cohort analysis by diagnosis group",
            "Length-of-stay distribution by department",
            "Readmission rate trends over time",
            "Comorbidity correlation analysis",
        ],
        "use_cases": [
            "Clinical decision support system",
            "Population health management",
            "Operational capacity planning",
            "Quality metrics reporting",
        ],
    },

    "Human Resources & HR": {
        "keywords": {
            "employee": 3, "salary": 3, "attrition": 3, "department": 2,
            "hire": 3, "termination": 3, "performance": 2, "rating": 2,
            "tenure": 3, "headcount": 3, "payroll": 3, "workforce": 3,
            "job": 2, "role": 2, "level": 1, "manager": 2, "engagement": 2,
            "satisfaction": 2, "promotion": 3, "training": 2, "absenteeism": 3,
        },
        "icon": "👥",
        "ml_goals": [
            {
                "title": "Employee Attrition Prediction",
                "description": "Predict which employees are likely to leave within 6 months.",
                "model_type": "Binary Classification (XGBoost / Random Forest)",
                "priority": "HIGH",
                "business_value": "Reducing attrition by 10% can save 1.5–2× annual salary per employee."
            },
            {
                "title": "Performance Tier Prediction",
                "description": "Classify employees into performance tiers based on behavioural signals.",
                "model_type": "Multi-Class Classification",
                "priority": "MEDIUM",
                "business_value": "Enables data-driven promotion and succession planning."
            },
            {
                "title": "Salary Equity Analysis",
                "description": "Detect pay gaps across gender, department, and tenure bands.",
                "model_type": "Regression + Fairness Analysis",
                "priority": "HIGH",
                "business_value": "Reduces regulatory risk and improves employee trust."
            },
        ],
        "kpi_targets": ["Attrition Rate", "Average Tenure", "Headcount", "Engagement Score", "Time to Hire"],
        "recommended_analyses": [
            "Attrition rate by department, role, and tenure band",
            "Salary distribution vs. performance rating",
            "Workforce demographic breakdown",
            "Absenteeism trend analysis",
        ],
        "use_cases": [
            "Talent retention program targeting",
            "Succession planning",
            "Compensation benchmarking",
            "DEI reporting and gap analysis",
        ],
    },

    "Marketing & CRM": {
        "keywords": {
            "lead": 3, "campaign": 3, "conversion": 3, "click": 3, "impression": 3,
            "ctr": 3, "roas": 3, "cpa": 3, "email": 2, "open_rate": 3,
            "segment": 2, "engagement": 2, "subscriber": 3, "unsubscribe": 3,
            "channel": 2, "acquisition": 2, "funnel": 3, "pipeline": 2,
            "opportunity": 2, "marketing": 3, "advertisement": 2, "ad": 2,
        },
        "icon": "📣",
        "ml_goals": [
            {
                "title": "Lead Scoring & Conversion Prediction",
                "description": "Score marketing leads by probability of converting to paid customers.",
                "model_type": "Binary Classification / Regression",
                "priority": "HIGH",
                "business_value": "Sales teams focus on high-probability leads — improves conversion efficiency by 20–40%."
            },
            {
                "title": "Customer Segmentation",
                "description": "Cluster customers by behavioural and demographic attributes for targeted campaigns.",
                "model_type": "Clustering (K-Means / Hierarchical / UMAP)",
                "priority": "HIGH",
                "business_value": "Segment-specific campaigns outperform generic blasts by 3–5×."
            },
            {
                "title": "Campaign Attribution Modeling",
                "description": "Attribute conversion credit across multi-touch marketing channels.",
                "model_type": "Markov Chain / Shapley Attribution",
                "priority": "MEDIUM",
                "business_value": "Reallocate budget toward highest-ROI channels."
            },
        ],
        "kpi_targets": ["Conversion Rate", "CAC", "ROAS", "CTR", "Pipeline Value"],
        "recommended_analyses": [
            "Channel performance comparison (CTR, CPA, ROAS)",
            "Funnel drop-off analysis by stage",
            "Email engagement trends",
            "Campaign ROI by segment",
        ],
        "use_cases": [
            "Personalization engine",
            "Marketing mix optimization",
            "A/B test result analysis",
            "Next-best-action recommendation",
        ],
    },

    "Operations & Logistics": {
        "keywords": {
            "shipment": 3, "delivery": 3, "warehouse": 3, "inventory": 3,
            "lead_time": 3, "supplier": 3, "purchase_order": 3, "sku": 2,
            "dispatch": 3, "logistics": 3, "route": 3, "fleet": 3,
            "defect": 3, "quality": 2, "sla": 3, "on_time": 3, "delay": 3,
            "capacity": 2, "throughput": 3, "cycle_time": 3, "oee": 3,
        },
        "icon": "🚚",
        "ml_goals": [
            {
                "title": "Delivery Delay Prediction",
                "description": "Predict shipment delays based on route, supplier, and weather features.",
                "model_type": "Binary / Regression (XGBoost)",
                "priority": "HIGH",
                "business_value": "Proactive customer communication reduces complaint volume by 30%+."
            },
            {
                "title": "Demand Forecasting for Inventory",
                "description": "Forecast SKU-level demand to minimize overstock and stockout events.",
                "model_type": "Time-Series (Prophet / LSTM)",
                "priority": "HIGH",
                "business_value": "Reduces carrying cost and maximizes service level."
            },
            {
                "title": "Anomaly / Quality Defect Detection",
                "description": "Detect production defects or quality anomalies in manufacturing lines.",
                "model_type": "Anomaly Detection (Isolation Forest / OCSVM)",
                "priority": "MEDIUM",
                "business_value": "Early detection reduces waste and rework costs."
            },
        ],
        "kpi_targets": ["On-Time Delivery Rate", "Inventory Turnover", "Lead Time", "Defect Rate", "OEE"],
        "recommended_analyses": [
            "On-time delivery trends by supplier and route",
            "Inventory turnover by SKU category",
            "Lead time distribution by supplier",
            "SLA breach analysis",
        ],
        "use_cases": [
            "Supply chain risk management",
            "Warehouse optimization",
            "Carrier performance benchmarking",
            "Predictive maintenance",
        ],
    },
}

GENERAL_DOMAIN = {
    "name": "General / Mixed",
    "icon": "📊",
    "ml_goals": [
        {
            "title": "Regression / Prediction",
            "description": "Predict a continuous numerical target variable from available features.",
            "model_type": "Random Forest / XGBoost / Linear Regression",
            "priority": "MEDIUM",
            "business_value": "Enables data-driven forecasting of any numerical KPI."
        },
        {
            "title": "Classification",
            "description": "Classify records into categories based on feature patterns.",
            "model_type": "Logistic Regression / XGBoost / SVM",
            "priority": "MEDIUM",
            "business_value": "Automates categorical decision-making at scale."
        },
        {
            "title": "Clustering / Segmentation",
            "description": "Discover natural groupings and segments in the data.",
            "model_type": "K-Means / DBSCAN / Hierarchical",
            "priority": "MEDIUM",
            "business_value": "Reveals hidden structure for targeted strategies."
        },
    ],
    "kpi_targets": ["Row Count", "Completeness", "Column Coverage"],
    "recommended_analyses": [
        "Descriptive statistics and distributions",
        "Correlation analysis between numerical features",
        "Missing value pattern analysis",
        "Categorical frequency distributions",
    ],
    "use_cases": [
        "Exploratory data analysis",
        "Feature engineering for ML pipeline",
        "Data quality improvement",
    ],
}


# ──────────────────────────────────────────────────────────────────────────────
# DOMAIN MAPPER
# ──────────────────────────────────────────────────────────────────────────────

class DomainMapper:
    """
    Infers the business domain of a dataset from its column names, column roles,
    and value patterns. Returns a rich domain profile with ML goals, KPIs, and
    recommended analyses.
    """

    @classmethod
    def infer_domain(
        cls,
        metadata: Dict[str, Any],
        df_sample: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """
        Main entry point. Returns:
          {
            domain, icon, confidence, score, runner_up,
            ml_goals, kpi_targets, recommended_analyses, use_cases,
            domain_signals_matched
          }
        """
        columns_lower = [c.lower().replace(" ", "_") for c in metadata.get("columns", [])]
        col_text = " ".join(columns_lower)

        domain_scores: Dict[str, int] = {}
        domain_matches: Dict[str, List[str]] = {}

        for domain_name, domain_def in DOMAIN_SIGNALS.items():
            score = 0
            matched = []
            for keyword, weight in domain_def["keywords"].items():
                # Check if keyword appears as substring in any column name
                kw_pattern = re.compile(r"\b" + re.escape(keyword) + r"\b", re.IGNORECASE)
                for col in columns_lower:
                    if kw_pattern.search(col) or keyword in col:
                        score += weight
                        if keyword not in matched:
                            matched.append(keyword)
                        break
            domain_scores[domain_name] = score
            domain_matches[domain_name] = matched

        # Sort by score descending
        sorted_domains = sorted(domain_scores.items(), key=lambda x: x[1], reverse=True)
        top_domain_name, top_score = sorted_domains[0]
        runner_up_name = sorted_domains[1][0] if len(sorted_domains) > 1 else None

        # Confidence calculation
        if top_score >= 15:
            confidence = "High"
        elif top_score >= 7:
            confidence = "Medium"
        elif top_score >= 3:
            confidence = "Low"
        else:
            confidence = "Unknown"

        if confidence == "Unknown":
            domain_def = GENERAL_DOMAIN
            domain_name = GENERAL_DOMAIN["name"]
            icon = GENERAL_DOMAIN["icon"]
        else:
            domain_def = DOMAIN_SIGNALS[top_domain_name]
            domain_name = top_domain_name
            icon = domain_def["icon"]

        return {
            "domain": domain_name,
            "icon": icon,
            "confidence": confidence,
            "score": top_score,
            "runner_up": runner_up_name,
            "ml_goals": domain_def.get("ml_goals", []),
            "kpi_targets": domain_def.get("kpi_targets", []),
            "recommended_analyses": domain_def.get("recommended_analyses", []),
            "use_cases": domain_def.get("use_cases", []),
            "domain_signals_matched": domain_matches.get(top_domain_name, []),
        }

    @classmethod
    def get_sql_goal_queries(cls, domain: str, dataset_name: str, metadata: Dict[str, Any]) -> Dict[str, str]:
        """
        Returns domain-specific pre-built SQL queries for the SQL Lab tab.
        """
        num_cols = metadata.get("column_buckets", {}).get("numerical_columns", [])
        cat_cols = metadata.get("column_buckets", {}).get("categorical_columns", [])
        dt_cols = metadata.get("column_buckets", {}).get("datetime_columns", [])

        table = dataset_name
        queries: Dict[str, str] = {}

        if num_cols:
            queries["Top Rows by Key Metric"] = (
                f"SELECT * FROM {table}\n"
                f"ORDER BY {num_cols[0]} DESC\n"
                f"LIMIT 20;"
            )
            queries["Metric Summary Statistics"] = (
                f"SELECT\n"
                + ",\n".join([
                    f"  ROUND(AVG({c}), 2) AS avg_{c},\n"
                    f"  ROUND(MIN({c}), 2) AS min_{c},\n"
                    f"  ROUND(MAX({c}), 2) AS max_{c}"
                    for c in num_cols[:3]
                ]) +
                f"\nFROM {table};"
            )

        if cat_cols and num_cols:
            queries["Group-by Aggregation"] = (
                f"SELECT\n"
                f"  {cat_cols[0]},\n"
                f"  COUNT(*) AS record_count,\n"
                f"  ROUND(SUM({num_cols[0]}), 2) AS total_{num_cols[0]},\n"
                f"  ROUND(AVG({num_cols[0]}), 2) AS avg_{num_cols[0]}\n"
                f"FROM {table}\n"
                f"GROUP BY {cat_cols[0]}\n"
                f"ORDER BY total_{num_cols[0]} DESC;"
            )

        if dt_cols and num_cols:
            queries["Time Trend Analysis"] = (
                f"SELECT\n"
                f"  {dt_cols[0]},\n"
                f"  COUNT(*) AS volume,\n"
                f"  ROUND(SUM({num_cols[0]}), 2) AS total_{num_cols[0]}\n"
                f"FROM {table}\n"
                f"GROUP BY {dt_cols[0]}\n"
                f"ORDER BY {dt_cols[0]};"
            )

        return queries
