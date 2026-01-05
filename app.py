import streamlit as st
import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(page_title="Salary Predictor", page_icon="💼", layout="wide")
st.title("💼 Salary Predictor using Decision Tree")

# --------------------------------------------------
# FILE UPLOADER
# --------------------------------------------------
uploaded_file = st.file_uploader("📁 Upload your salary CSV file", type=["csv"])

if uploaded_file is not None:

    # --------------------------------------------------
    # LOAD DATA
    # --------------------------------------------------
    df = pd.read_csv(uploaded_file)
    st.success(f"✅ Dataset loaded successfully ({df.shape[0]} rows)")

    st.subheader("📊 Dataset Preview")
    st.dataframe(df.head(), use_container_width=True)

    # --------------------------------------------------
    # AUTO DETECT SALARY COLUMN
    # --------------------------------------------------
    st.subheader("🔍 Column Detection")
    st.write("Columns found:", df.columns.tolist())

    try:
        salary_col = next(col for col in df.columns if 'salary' in col.lower())
        st.success(f"🎯 Detected target column: `{salary_col}`")
    except StopIteration:
        st.error("❌ No column containing the word 'salary' found.")
        st.stop()

    # --------------------------------------------------
    # ENCODE CATEGORICAL COLUMNS
    # --------------------------------------------------
    df_encoded = df.copy()
    le = LabelEncoder()

    for col in df_encoded.columns:
        if df_encoded[col].dtype == 'object':
            df_encoded[col] = le.fit_transform(df_encoded[col])

    # --------------------------------------------------
    # SPLIT DATA
    # --------------------------------------------------
    X = df_encoded.drop(salary_col, axis=1)
    y = df_encoded[salary_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # --------------------------------------------------
    # TRAIN MODEL
    # --------------------------------------------------
    model = DecisionTreeRegressor(random_state=42)
    model.fit(X_train, y_train)

    # --------------------------------------------------
    # PREDICTION & METRICS
    # --------------------------------------------------
    y_pred = model.predict(X_test)

    st.subheader("📈 Model Performance")
    st.write("**Mean Absolute Error:**", mean_absolute_error(y_test, y_pred))
    st.write("**R² Score:**", r2_score(y_test, y_pred))

    # --------------------------------------------------
    # USER INPUT FOR PREDICTION
    # --------------------------------------------------
    st.subheader("🧮 Predict Salary")

    user_input = {}
    for col in X.columns:
        user_input[col] = st.number_input(
            f"Enter value for {col}",
            value=float(X[col].mean())
        )

    input_df = pd.DataFrame([user_input])

    if st.button("🔮 Predict Salary"):
        prediction = model.predict(input_df)
        st.success(f"💰 Predicted Salary: {prediction[0]:,.2f}")

else:
    st.info("👆 Please upload a CSV file to begin.")
