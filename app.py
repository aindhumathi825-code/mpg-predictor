"""
Automobile Fuel Efficiency (MPG) Prediction App
------------------------------------------------
Streamlit app with a full scikit-learn pipeline:
  1. Data extraction   (UCI Auto MPG dataset / local file / uploaded CSV)
  2. Cleaning          (type fixes, '?' -> NaN, duplicates)
  3. Missing values    (median for numeric, most-frequent for categorical)
  4. Feature engineering (power-to-weight ratio, vehicle age)
  5. Encoding          (OneHotEncoder for Origin)
  6. Scaling           (StandardScaler for numeric features)
  7. Model selection   (cross-validated comparison of several regressors)
  8. Prediction UI
"""

import os

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------
st.set_page_config(page_title="MPG Predictor", page_icon="🚗", layout="centered")

DATA_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data"
)
LOCAL_FILE = "auto-mpg.csv"  # optional: place a local copy next to app.py
COLUMNS = [
    "mpg", "cylinders", "displacement", "horsepower",
    "weight", "acceleration", "model_year", "origin", "car_name",
]
TARGET = "mpg"
NUMERIC_FEATURES = ["cylinders", "displacement", "horsepower",
                    "weight", "acceleration", "model_year"]
CATEGORICAL_FEATURES = ["origin"]
ORIGIN_MAP = {1: "USA", 2: "Europe", 3: "Japan"}
RANDOM_STATE = 42


# --------------------------------------------------------------------------
# 1. Data extraction
# --------------------------------------------------------------------------
def _read_raw(source) -> pd.DataFrame:
    """Read the whitespace-separated UCI file or a CSV with the same columns."""
    try:
        df = pd.read_csv(
            source, names=COLUMNS, na_values="?", comment="\t",
            sep=r"\s+", skipinitialspace=True,
        )
        if pd.to_numeric(df[TARGET], errors="coerce").notna().mean() > 0.5 \
                and pd.to_numeric(df["cylinders"], errors="coerce").notna().mean() > 0.5:
            return df
    except Exception:
        pass
    if hasattr(source, "seek"):
        source.seek(0)
    df = pd.read_csv(source, na_values=["?", "NA", ""])
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    return df


@st.cache_data(show_spinner="Loading data...")
def load_data(uploaded_bytes: bytes | None = None) -> pd.DataFrame:
    import io

    if uploaded_bytes is not None:
        return _read_raw(io.BytesIO(uploaded_bytes))
    if os.path.exists(LOCAL_FILE):
        return _read_raw(LOCAL_FILE)
    return _read_raw(DATA_URL)


# --------------------------------------------------------------------------
# 2. Cleaning
# --------------------------------------------------------------------------
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.drop_duplicates()

    missing_cols = [c for c in NUMERIC_FEATURES + CATEGORICAL_FEATURES + [TARGET]
                    if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Dataset is missing required columns: {missing_cols}")

    for col in NUMERIC_FEATURES + [TARGET]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Rows without a target value cannot be used for training.
    df = df.dropna(subset=[TARGET])

    # Origin: accept either codes (1/2/3) or names.
    if pd.api.types.is_numeric_dtype(df["origin"]):
        df["origin"] = df["origin"].map(ORIGIN_MAP)
    df["origin"] = df["origin"].astype("string").str.strip().str.title()
    df.loc[df["origin"] == "Usa", "origin"] = "USA"
    return df.reset_index(drop=True)


# --------------------------------------------------------------------------
# 3-6. Preprocessing pipeline (impute -> engineer -> encode -> scale)
# --------------------------------------------------------------------------
def add_features(X: pd.DataFrame) -> pd.DataFrame:
    """Feature engineering (runs after imputation, so no NaNs)."""
    X = X.copy()
    X["power_to_weight"] = X["horsepower"] / X["weight"]
    X["vehicle_age"] = X["model_year"].max() - X["model_year"]
    return X


def build_preprocessor() -> ColumnTransformer:
    numeric_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),          # missing values
    ])
    categorical_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),   # missing values
        ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),  # encoding
    ])
    stage1 = ColumnTransformer(
        [("num", numeric_pipe, NUMERIC_FEATURES),
         ("cat", categorical_pipe, CATEGORICAL_FEATURES)],
        verbose_feature_names_out=False,
    )

    # Feature engineering + scaling on the numeric part
    def engineer_and_scale():
        def _fe(arr):
            df = pd.DataFrame(arr[:, :len(NUMERIC_FEATURES)],
                              columns=NUMERIC_FEATURES).astype(float)
            df = add_features(df)
            return np.hstack([df.values, arr[:, len(NUMERIC_FEATURES):]])
        return FunctionTransformer(_fe, validate=False)

    n_num = len(NUMERIC_FEATURES) + 2  # + engineered features
    stage2 = ColumnTransformer(
        [("scale", StandardScaler(), list(range(n_num)))],
        remainder="passthrough",  # one-hot columns are left unscaled
    )
    return Pipeline([
        ("stage1", stage1.set_output(transform="default")),
        ("features", engineer_and_scale()),
        ("stage2", stage2),
    ])


def candidate_models() -> dict:
    return {
        "Ridge Regression": Ridge(alpha=1.0),
        "Random Forest": RandomForestRegressor(
            n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=300, learning_rate=0.05, max_depth=3,
            random_state=RANDOM_STATE),
    }


# --------------------------------------------------------------------------
# 7. Training + model selection
# --------------------------------------------------------------------------
@st.cache_resource(show_spinner="Training models...")
def train(df: pd.DataFrame):
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE)

    results, pipes = [], {}
    for name, model in candidate_models().items():
        pipe = Pipeline([("prep", build_preprocessor()), ("model", model)])
        cv_r2 = cross_val_score(pipe, X_train, y_train, cv=5, scoring="r2").mean()
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        results.append({
            "Model": name,
            "CV R²": round(cv_r2, 3),
            "Test R²": round(r2_score(y_test, pred), 3),
            "Test RMSE": round(float(np.sqrt(mean_squared_error(y_test, pred))), 3),
            "Test MAE": round(mean_absolute_error(y_test, pred), 3),
        })
        pipes[name] = pipe

    results_df = pd.DataFrame(results).sort_values("CV R²", ascending=False)
    best_name = results_df.iloc[0]["Model"]

    # Refit the winner on ALL data for the final app
    final = Pipeline([("prep", build_preprocessor()),
                      ("model", candidate_models()[best_name])])
    final.fit(X, y)
    return final, results_df.reset_index(drop=True), best_name


# --------------------------------------------------------------------------
# 8. UI
# --------------------------------------------------------------------------
def main():
    st.title("🚗 Automobile Fuel Efficiency Predictor")
    st.caption("Predict a car's MPG from its specifications.")

    with st.sidebar:
        st.header("Data source")
        up = st.file_uploader("Optional: upload your own CSV", type=["csv", "data", "txt"])
        st.markdown("Default: UCI Auto MPG dataset.")

    try:
        raw = load_data(up.getvalue() if up else None)
        df = clean_data(raw)
    except Exception as e:
        st.error(f"Could not load the data: {e}")
        st.info("Upload a CSV from the sidebar, or put `auto-mpg.csv` next to app.py.")
        st.stop()

    model, results, best_name = train(df)

    tab_pred, tab_data, tab_model = st.tabs(["Predict", "Data", "Model performance"])

    # ---- Predict
    with tab_pred:
        st.subheader("Enter vehicle details")
        c1, c2 = st.columns(2)
        with c1:
            cylinders = st.selectbox("Cylinders", [3, 4, 5, 6, 8], index=1)
            displacement = st.slider("Displacement (cu. in.)", 60.0, 460.0, 150.0, 1.0)
            horsepower = st.slider("Horsepower", 40.0, 250.0, 95.0, 1.0)
        with c2:
            weight = st.slider("Weight (lbs)", 1500, 5200, 2800, 10)
            acceleration = st.slider("Acceleration (0-60 mph, sec)", 7.0, 25.0, 15.0, 0.1)
            model_year = st.slider("Model year (19xx)", 70, 82, 76, 1)
        origin = st.selectbox("Origin", list(ORIGIN_MAP.values()))

        if st.button("Predict MPG", type="primary"):
            row = pd.DataFrame([{
                "cylinders": cylinders, "displacement": displacement,
                "horsepower": horsepower, "weight": weight,
                "acceleration": acceleration, "model_year": model_year,
                "origin": origin,
            }])
            mpg = float(model.predict(row)[0])
            kmpl = mpg * 0.425144  # US MPG -> km/l
            m1, m2 = st.columns(2)
            m1.metric("Predicted fuel efficiency", f"{mpg:.1f} MPG")
            m2.metric("Approx.", f"{kmpl:.1f} km/l")
            st.caption(f"Model used: {best_name}. MPG is in US gallons.")

    # ---- Data
    with tab_data:
        st.subheader("Cleaned data")
        st.write(f"{len(df)} rows. Missing values before imputation:")
        st.dataframe(df[NUMERIC_FEATURES + CATEGORICAL_FEATURES + [TARGET]]
                     .isna().sum().rename("missing").to_frame().T)
        st.dataframe(df.head(50))
        st.subheader("Summary statistics")
        st.dataframe(df[NUMERIC_FEATURES + [TARGET]].describe().T)
        st.subheader("MPG vs weight")
        st.scatter_chart(df, x="weight", y="mpg", color="origin")

    # ---- Model
    with tab_model:
        st.subheader("Model comparison")
        st.dataframe(results, hide_index=True)
        st.success(f"Best model by cross-validated R²: **{best_name}** "
                   "(refit on all data for predictions).")
        st.markdown(
            "**Pipeline:** median/mode imputation → feature engineering "
            "(power-to-weight, vehicle age) → one-hot encoding (origin) → "
            "standard scaling → regressor."
        )


if __name__ == "__main__":
    main()
