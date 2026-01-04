# Save this file as app.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# --- CONFIG ---
DATA_FILE = "yield_df.csv"
TARGET_COLUMN = "hg/ha_yield"
SEED = 42

# --- MODEL TRAINING ---
@st.cache_resource
def load_and_train_model():
    df = pd.read_csv(DATA_FILE)

    df = df.drop(columns=["Year", "average_item_value", "Unnamed: 0"], errors="ignore")
    df = df.dropna()

    X = df.drop(TARGET_COLUMN, axis=1)
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED
    )

    num_cols = X.select_dtypes(include="number").columns
    cat_cols = X.select_dtypes(include="object").columns

    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
    ])

    models = {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(random_state=SEED),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=SEED)
    }

    best_r2 = -1
    best_model_name = ""
    best_pipeline = None
    results = {}

    for name, model in models.items():
        pipe = Pipeline([
            ("preprocessor", preprocessor),
            ("model", model)
        ])
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)

        r2 = r2_score(y_test, preds)
        mae = mean_absolute_error(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))

        results[name] = {"R2": r2, "MAE": mae, "RMSE": rmse}

        if r2 > best_r2:
            best_r2 = r2
            best_pipeline = pipe
            best_model_name = name

    return best_pipeline, best_model_name, pd.DataFrame(results).T, df


# --- STREAMLIT UI ---
st.set_page_config(page_title="Crop Yield Prediction", layout="wide")
st.title("🌾 Crop Yield Prediction System")

model, model_name, results_df, df = load_and_train_model()
st.success(f"Best Model Selected: **{model_name}**")

# INPUTS
rainfall = st.slider(
    "Average Rainfall (mm)",
    float(df["average_rain_fall_mm_per_year"].min()),
    float(df["average_rain_fall_mm_per_year"].max()),
    1500.0
)

temp = st.slider("Average Temperature (°C)", 0.0, 45.0, 25.0)
pesticides = st.slider("Pesticides (tonnes)", 0.0, 100000.0, 50000.0)

area = st.selectbox("Area", sorted(df["Area"].unique()))
item = st.selectbox("Crop", sorted(df["Item"].unique()))

input_df = pd.DataFrame({
    "average_rain_fall_mm_per_year": [rainfall],
    "avg_temp": [temp],
    "pesticides_tonnes": [pesticides],
    "Area": [area],
    "Item": [item]
})

prediction = model.predict(input_df)[0]
st.subheader(f"📈 Predicted Yield: **{prediction:.2f} hg/ha**")

st.subheader("📊 Model Performance")
st.dataframe(results_df)
