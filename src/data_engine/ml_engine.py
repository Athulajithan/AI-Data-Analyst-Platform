import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    mean_absolute_error, mean_squared_error, r2_score
)
from typing import Dict, Any, Tuple, Optional

class MLEngine:
    """
    Automated baseline ML model training & evaluation for Classification and Regression tasks.
    """

    @classmethod
    def train_baseline_model(
        cls,
        df: pd.DataFrame,
        target_col: str,
        problem_type: Optional[str] = None
    ) -> Dict[str, Any]:
        
        if target_col not in df.columns:
            return {"status": f"Target column '{target_col}' not found in dataset."}

        clean_df = df.dropna(subset=[target_col]).copy()
        if len(clean_df) < 15:
            return {"status": "Insufficient rows (< 15) for machine learning baseline training."}

        # Infer problem type if not supplied
        y_raw = clean_df[target_col]
        if problem_type is None:
            if y_raw.nunique() <= 10 or y_raw.dtype == "object" or pd.api.types.is_bool_dtype(y_raw):
                problem_type = "classification"
            else:
                problem_type = "regression"

        # Feature selection: drop high cardinality IDs, dates, and target itself
        drop_cols = [c for c in clean_df.columns if c.lower().endswith("id") or "date" in c.lower() or c == target_col]
        X_df = clean_df.drop(columns=drop_cols)
        
        # Convert categoricals to dummy variables
        X_encoded = pd.get_dummies(X_df, drop_first=True)
        X_encoded = X_encoded.fillna(X_encoded.median(numeric_only=True))

        if X_encoded.shape[1] == 0:
            return {"status": "No valid predictor features remain after ID/date exclusion."}

        if problem_type == "classification":
            y = pd.Series(pd.factorize(y_raw)[0])
            if y.nunique() < 2:
                return {"status": "Target contains only 1 class; classification impossible."}

            X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.25, random_state=42, stratify=y if y.value_counts().min() > 1 else None)
            
            clf = RandomForestClassifier(n_estimators=50, random_state=42)
            clf.fit(X_train, y_train)
            
            y_pred = clf.predict(X_test)
            y_prob = clf.predict_proba(X_test)[:, 1] if len(np.unique(y)) == 2 else None

            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
            rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
            f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
            auc = roc_auc_score(y_test, y_prob) if y_prob is not None else None

            # Feature Importance
            importances = pd.Series(clf.feature_importances_, index=X_encoded.columns).sort_values(ascending=False).head(5).to_dict()

            return {
                "problem_type": "Classification",
                "target_column": target_col,
                "model_name": "Random Forest Classifier Baseline",
                "sample_size_train": len(X_train),
                "sample_size_test": len(X_test),
                "metrics": {
                    "accuracy": round(float(acc), 4),
                    "precision": round(float(prec), 4),
                    "recall": round(float(rec), 4),
                    "f1_score": round(float(f1), 4),
                    "roc_auc": round(float(auc), 4) if auc is not None else "N/A (Multi-class)"
                },
                "top_feature_importance": importances,
                "interpretation": f"Baseline classification achieved {round(acc*100, 2)}% accuracy and F1-score of {round(f1, 3)}."
            }

        else:
            y = pd.to_numeric(y_raw, errors="coerce").fillna(y_raw.median() if pd.api.types.is_numeric_dtype(y_raw) else 0)
            X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.25, random_state=42)

            reg = RandomForestRegressor(n_estimators=50, random_state=42)
            reg.fit(X_train, y_train)

            y_pred = reg.predict(X_test)

            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)

            importances = pd.Series(reg.feature_importances_, index=X_encoded.columns).sort_values(ascending=False).head(5).to_dict()

            return {
                "problem_type": "Regression",
                "target_column": target_col,
                "model_name": "Random Forest Regressor Baseline",
                "sample_size_train": len(X_train),
                "sample_size_test": len(X_test),
                "metrics": {
                    "mae": round(float(mae), 4),
                    "rmse": round(float(rmse), 4),
                    "r2_score": round(float(r2), 4),
                },
                "top_feature_importance": importances,
                "interpretation": f"Baseline regression model explained {round(r2*100, 2)}% of variance (R²={round(r2, 3)}) with MAE={round(mae, 2)}."
            }
