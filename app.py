# app.py
#
# Streamlit UI for the AI Player Prop Edge Finder.
# Requires prop_model_core.py in the same folder.

import streamlit as st
import pandas as pd

from prop_model_core import (
    train_prop_model,
    rank_props_by_value,
    NUM_FEATURES,
    CAT_FEATURES,
)

st.set_page_config(page_title="AI Player Prop Edge Finder", layout="wide")

st.title("🧠 AI Player Prop Edge Finder")
st.write(
    """
This app uses a machine learning model trained on historical player prop data
to estimate the probability that each prop will hit, then compares that to the
implied probability from the odds to estimate an **edge**.

**For educational / analytical purposes only. Not betting advice.**
"""
)

# =========================
# 1. Upload historical data (training)
# =========================

st.header("1️⃣ Upload historical prop data (for training)")

st.write(
    """
Upload a CSV containing **historical player props**, including:
- A `hit` column (1 if the prop covered, 0 otherwise)
- Feature columns such as:
  - `sport`, `prop_type`, `team`, `opponent`, `home`
  - `line`, `player_avg_last5`, `player_std_last5`
  - `minutes_avg_last5`, `opp_def_rank`, `opp_pace`, `rest_days`
  - `implied_prob` (or at least `american_odds` so you can compute it beforehand)
"""
)

hist_file = st.file_uploader(
    "Choose historical props CSV...",
    type=["csv"],
    key="hist_upload",
)

model = None
metrics_df = None

if hist_file is not None:
    df_hist = pd.read_csv(hist_file)

    st.subheader("Preview of historical data")
    st.dataframe(df_hist.head())

    # Allow the user to see which features the model expects
    with st.expander("Expected feature columns", expanded=False):
        st.write("**Numeric features:**", NUM_FEATURES)
        st.write("**Categorical features:**", CAT_FEATURES)
        st.write("**Target column:**", "hit")

    with st.spinner("Training model on historical data..."):
        try:
            model, metrics_df = train_prop_model(df_hist)
            st.success("✅ Model trained successfully!")
            st.subheader("Evaluation metrics")
            st.table(metrics_df)
        except Exception as e:
            st.error(f"Error training model: {e}")
            model = None

else:
    st.info("Upload a historical CSV to train the model.")


# =========================
# 2. Upload today's props (scoring)
# =========================

st.header("2️⃣ Upload today's props to score")

st.write(
    """
Upload a CSV with **today's props** that includes the *same feature columns*
used during training (except `hit`).

- If `implied_prob` is missing but `american_odds` is present,
  it will be computed automatically.
- The app will output:
  - `predicted_prob_hit` — model's estimate
  - `edge` — predicted_prob_hit − implied_prob
"""
)

props_file = st.file_uploader(
    "Choose today's props CSV...",
    type=["csv"],
    key="props_upload",
)

if props_file is not None:
    if model is None:
        st.warning("You need to upload historical data and train a model first.")
    else:
        df_props = pd.read_csv(props_file)

        st.subheader("Preview of today's props data")
        st.dataframe(df_props.head())

        with st.expander("Reminder of expected feature columns", expanded=False):
            st.write("**Numeric features:**", NUM_FEATURES)
            st.write("**Categorical features:**", CAT_FEATURES)

        with st.spinner("Scoring today's props..."):
            try:
                ranked_df = rank_props_by_value(model, df_props)

                st.success("✅ Props scored and ranked by estimated edge!")

                # Show a subset of most relevant columns if they exist
                cols_to_show = [
                    "sport",
                    "player",
                    "team",
                    "opponent",
                    "prop_type",
                    "line",
                    "american_odds",
                    "implied_prob",
                    "predicted_prob_hit",
                    "edge",
                ]
                existing_cols = [c for c in cols_to_show if c in ranked_df.columns]

                st.subheader("Top ranked props")
                st.dataframe(ranked_df[existing_cols].head(50))

                # Download button for full results
                csv_out = ranked_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download full ranked results as CSV",
                    data=csv_out,
                    file_name="ranked_props.csv",
                    mime="text/csv",
                )
            except Exception as e:
                st.error(f"Error scoring props: {e}")
else:
    st.info("Upload a CSV of today's props after training the model.")


# =========================
# Footer
# =========================

st.markdown("---")
st.caption(
    "This tool is for **educational and analytical purposes only** and does not "
    "constitute betting advice. Always gamble responsibly."
)
