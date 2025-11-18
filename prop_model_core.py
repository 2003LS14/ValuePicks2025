# prop_model_core.py
#
# Core model logic for training and scoring player prop edges.
# This module is imported by app.py (Streamlit UI).

from typing import List, Tuple

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# =========================
# CONFIGURATION
# =========================

TARGET_COL = "hit"  # 1 if prop hit, 0 otherwise

# Numerical features used by the model
NUM_FEATURES = [
    "line",
    "player_avg_last5",
    "player_std_last5",
    "minutes_avg_last5",
    "opp_def_rank",
    "opp_pace",
    "rest_days",
    "implied_prob",  # from betting odds
]

# Categorical features used by the model
CAT_FEATURES = [
    "sport",
    "prop_type",
    "team",
    "opponent",
    "home",  # can be treated as categorical or numeric; here we one-hot it
]


# =========================
# UTILITY FUNCTIONS
# =========================

def american_odds_to_implied_prob(odds: float) -> float:
    """
    Convert American odds to implied probability (without removing vig).

    Examples:
        -150 -> 0.60
        +120 -> 0.4545...
    """
    if odds < 0:
        return (-odds) / ((-odds) + 100)
    else:
        return 100 / (odds + 100)


def build_pipeline(num_features: List[str], cat_features: List[str]) -> Pipeline:
    """
    Build a sklearn Pipeline that:
      1) scales numeric features,
      2) one-hot encodes categorical features,
      3) fits a Gradient Boosting classifier.
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features),
        ]
    )

    clf = GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=3,
        subsample=0.8,
        random_state=42,
    )

    model = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", clf),
        ]
    )

    return model


# =========================
# TRAINING LOGIC
# =========================

def train_prop_model(
    df: pd.DataFrame,
    num_features: List[str] = None,
    cat_features: List[str] = None,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[Pipeline, pd.DataFrame]:
    """
    Train a model to predict whether a player prop will hit.

    Args:
        df: DataFrame with historical props. Must include TARGET_COL ("hit")
            and all feature columns.
        num_features: Optional override for numeric feature list.
        cat_features: Optional override for categorical feature list.
        test_size: Fraction of data used for testing.
        random_state: Random seed for reproducibility.

    Returns:
        model: Trained sklearn Pipeline.
        metrics_df: DataFrame with basic evaluation metrics.
    """
    if num_features is None:
        num_features = NUM_FEATURES
    if cat_features is None:
        cat_features = CAT_FEATURES

    required_cols = num_features + cat_features + [TARGET_COL]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in historical data: {missing}")

    # Drop rows with missing required values
    df = df.dropna(subset=required_cols).copy()

    # Ensure target is integer 0/1
    df[TARGET_COL] = df[TARGET_COL].astype(int)

    X = df[num_features + cat_features]
    y = df[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    model = build_pipeline(num_features, cat_features)
    model.fit(X_train, y_train)

    # Evaluate on test set
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    try:
        auc = roc_auc_score(y_test, y_proba)
    except ValueError:
        # This can happen if only one class is present in y_test
        auc = float("nan")

    metrics_df = pd.DataFrame(
        {
            "metric": ["accuracy", "roc_auc"],
            "value": [acc, auc],
        }
    )

    return model, metrics_df


# =========================
# SCORING / RANKING LOGIC
# =========================

def rank_props_by_value(
    model: Pipeline,
    new_props_df: pd.DataFrame,
    num_features: List[str] = None,
    cat_features: List[str] = None,
) -> pd.DataFrame:
    """
    Use a trained model to score new props (e.g., today's props),
    estimate hit probabilities, and compute "edge" vs implied probability.

    Args:
        model: Trained sklearn Pipeline returned by train_prop_model.
        new_props_df: DataFrame with today's props, including features.
        num_features: Optional override for numeric features.
        cat_features: Optional override for categorical features.

    Returns:
        DataFrame with added columns:
            - predicted_prob_hit
            - edge = predicted_prob_hit - implied_prob
        Sorted by edge descending.
    """
    if num_features is None:
        num_features = NUM_FEATURES
    if cat_features is None:
        cat_features = CAT_FEATURES

    df = new_props_df.copy()

    # If implied_prob is missing but we have american_odds, compute it
    if "implied_prob" in num_features and "implied_prob" not in df.columns:
        if "american_odds" in df.columns:
            df["implied_prob"] = df["american_odds"].apply(american_odds_to_implied_prob)
        else:
            raise ValueError("implied_prob missing and american_odds not provided.")

    # Ensure all feature columns exist
    required_cols = num_features + cat_features
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in new props data: {missing}")

    X_new = df[num_features + cat_features]

    # Predict probability that the prop hits
    pred_probs = model.predict_proba(X_new)[:, 1]
    df["predicted_prob_hit"] = pred_probs

    # Compute edge if implied_prob is available
    if "implied_prob" in df.columns:
        df["edge"] = df["predicted_prob_hit"] - df["implied_prob"]

    # Sort by edge if present, else by predicted_prob_hit
    sort_col = "edge" if "edge" in df.columns else "predicted_prob_hit"
    df = df.sort_values(by=sort_col, ascending=False).reset_index(drop=True)

    return df
