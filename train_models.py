import os
import numpy as np
import pandas as pd
import joblib
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, RandomizedSearchCV, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    RBF, Matern, RationalQuadratic, WhiteKernel, ConstantKernel as C
)

# =========================
# PATHS
# =========================
DATA_XLSX = r"D:\irrrigation water quality index\Data\Real data_with_indices.xlsx"
MODEL_DIR = r"C:\code\Mostafa\iwqi_streamlit_app\models"
os.makedirs(MODEL_DIR, exist_ok=True)

# =========================
# SETTINGS
# =========================
FEATURES = ["EC", "PH", "T"]
TARGETS = ["IWQI", "SAR", "PS"]

TEST_SIZE = 0.30
RANDOM_STATE = 42
CV_SPLITS = 3
N_ITER = 20

# =========================
# MODELS
# =========================
models = {
    "GBM": GradientBoostingRegressor(random_state=RANDOM_STATE),
    "RF": RandomForestRegressor(random_state=RANDOM_STATE),
    "DT": DecisionTreeRegressor(random_state=RANDOM_STATE),
    "GPR": GaussianProcessRegressor(normalize_y=True, random_state=RANDOM_STATE),
}

param_spaces = {
    "GBM": {
        "model__n_estimators": [200, 400, 600],
        "model__learning_rate": [0.03, 0.05, 0.1],
        "model__max_depth": [2, 3, 4],
    },
    "RF": {
        "model__n_estimators": [300, 600],
        "model__max_depth": [None, 10, 20],
    },
    "DT": {
        "model__max_depth": [None, 5, 10, 20],
    },
    "GPR": {
        "model__kernel": [
            C(1.0) * RBF(1.0) + WhiteKernel(),
            C(1.0) * Matern(1.0, nu=1.5) + WhiteKernel(),
        ],
        "model__alpha": [1e-6, 1e-4, 1e-2],
    },
}

def make_pipeline(name, model):
    if name == "GPR":
        return Pipeline([("scaler", StandardScaler()), ("model", model)])
    return Pipeline([("model", model)])

def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

# =========================
# LOAD DATA
# =========================
df = pd.read_excel(DATA_XLSX)
df = df[FEATURES + TARGETS].dropna()

X = df[FEATURES].values
cv = KFold(n_splits=CV_SPLITS, shuffle=True, random_state=RANDOM_STATE)

# SAME split for all targets
idx = np.arange(len(df))
idx_train, idx_test = train_test_split(idx, test_size=TEST_SIZE, random_state=RANDOM_STATE)

X_train = X[idx_train]
X_test = X[idx_test]

# =========================
# TRAIN & SAVE
# =========================
for target in TARGETS:
    print(f"\nTraining for {target}")
    y = df[target].values
    y_train = y[idx_train]
    y_test = y[idx_test]

    best_rmse = np.inf
    best_estimator = None
    best_name = None

    for name, base_model in models.items():
        pipe = make_pipeline(name, base_model)

        search = RandomizedSearchCV(
            pipe,
            param_spaces[name],
            n_iter=N_ITER,
            cv=cv,
            scoring="neg_root_mean_squared_error",
            n_jobs=-1,
            random_state=RANDOM_STATE
        )
        search.fit(X_train, y_train)

        pred = search.best_estimator_.predict(X_test)
        score = rmse(y_test, pred)

        print(f"{name} RMSE = {score:.4f}")

        if score < best_rmse:
            best_rmse = score
            best_estimator = search.best_estimator_
            best_name = name

    # SAVE MODEL
    out_path = os.path.join(MODEL_DIR, f"best_{target}.joblib")
    joblib.dump(best_estimator, out_path)
    print(f"Saved BEST model for {target}: {best_name} → {out_path}")
