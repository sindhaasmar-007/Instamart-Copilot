
from pathlib import Path
import pandas as pd

# Project folders
BASE_DIR = Path(__file__).resolve().parent
CLEAN_DIR = BASE_DIR / "data" / "clean"
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load Part A cleaned orders
orders = pd.read_csv(CLEAN_DIR / "orders.csv")

# Prepare the required columns
orders["order_ts"] = pd.to_datetime(
    orders["order_ts"], errors="coerce"
)
orders["order_amount"] = pd.to_numeric(
    orders["order_amount"], errors="coerce"
)
orders["status"] = orders["status"].astype(str).str.strip().str.lower()

# Use completed orders only
completed = orders[orders["status"] == "completed"].copy()
completed = completed.dropna(
    subset=["customer_id", "order_id", "order_ts"]
)

# Snapshot moment
snapshot = pd.Timestamp("2025-10-01 00:00:00")

# Build the RFM table
rfm = (
    completed.groupby("customer_id")
    .agg(
        last_order_ts=("order_ts", "max"),
        frequency=("order_id", "nunique"),
        monetary=("order_amount", "sum"),
    )
    .reset_index()
)

rfm["recency"] = (snapshot - rfm["last_order_ts"]).dt.days

# Keep the required columns
rfm = rfm[
    ["customer_id", "recency", "frequency", "monetary"]
]

# Save the RFM table
rfm.to_csv(OUTPUT_DIR / "rfm_table.csv", index=False)

# Display summary
print("RFM table created successfully!")
print("Customers:", len(rfm))
print("Recency median:", rfm["recency"].median())
print("Recency minimum:", rfm["recency"].min())
print("Recency maximum:", rfm["recency"].max())
print("Frequency mean:", round(rfm["frequency"].mean(), 2))
print("Frequency maximum:", rfm["frequency"].max())
print("Monetary mean:", round(rfm["monetary"].mean(), 2))
print("Monetary maximum:", round(rfm["monetary"].max(), 2))
print("\nFirst 5 rows:")
print(rfm.head())

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# STEP 2: K-means clustering

# Select RFM columns
rfm_features = rfm[["recency", "frequency", "monetary"]]

# Standardise the columns
scaler = StandardScaler()
rfm_scaled = scaler.fit_transform(rfm_features)

# Test K values from 2 to 8
cluster_results = []

for k in range(2, 9):
    kmeans = KMeans(
        n_clusters=k,
        n_init=10,
        random_state=42
    )

    labels = kmeans.fit_predict(rfm_scaled)

    inertia = kmeans.inertia_
    silhouette = silhouette_score(rfm_scaled, labels)

    cluster_results.append({
        "K": k,
        "Inertia": round(inertia, 1),
        "Silhouette": round(silhouette, 3)
    })

# Display comparison table
cluster_table = pd.DataFrame(cluster_results)

print("\nK-MEANS COMPARISON")
print(cluster_table.to_string(index=False))

# STEP 3: Final K=4 customer segments and PCA

from sklearn.decomposition import PCA

# Fit final K-means model
final_kmeans = KMeans(
    n_clusters=4,
    n_init=10,
    random_state=42
)

rfm["segment_id"] = final_kmeans.fit_predict(rfm_scaled)

# Show each segment's size and average RFM values
segment_profile = (
    rfm.groupby("segment_id")
    .agg(
        customer_count=("customer_id", "count"),
        mean_recency=("recency", "mean"),
        mean_frequency=("frequency", "mean"),
        mean_monetary=("monetary", "mean")
    )
    .round(2)
)

print("\nFINAL K=4 SEGMENT PROFILE")
print(segment_profile.to_string())

# PCA using the scaled RFM data
pca = PCA(n_components=2)
pca_result = pca.fit_transform(rfm_scaled)

print("\nPCA EXPLAINED VARIANCE RATIO")
print(pca.explained_variance_ratio_.round(3))
print("Total explained variance:",
      round(pca.explained_variance_ratio_.sum(), 3))
# STEP 4: Name the customer segments

segment_names = {
    0: "Regular_Low_Spend",
    1: "Premium",
    2: "Loyal_Frequent",
    3: "At_Risk_Inactive"
}

rfm["segment_name"] = rfm["segment_id"].map(segment_names)

# Save customer segmentation results
rfm.to_csv(
    OUTPUT_DIR / "customer_segments.csv",
    index=False
)

print("\nCUSTOMER SEGMENTS SAVED")
print(rfm["segment_name"].value_counts())
print("\nSaved file: outputs/customer_segments.csv")

# STEP 5: Create customer lapse labels

orders["order_ts"] = pd.to_datetime(
    orders["order_ts"], errors="coerce"
)

cutoff_date = pd.Timestamp("2025-09-01")

# Use completed orders only
completed_orders = orders[
    (orders["status"] == "completed") &
    (orders["order_ts"].notna())
].copy()

# Customer activity before the cutoff
historical_orders = completed_orders[
    completed_orders["order_ts"] < cutoff_date
]

# Customer activity on or after the cutoff
future_orders = completed_orders[
    completed_orders["order_ts"] >= cutoff_date
]

# Customers who ordered before the cutoff
customer_history = historical_orders.groupby(
    "customer_id"
).agg(
    last_order_before_cutoff=("order_ts", "max"),
    order_count_before_cutoff=("order_id", "nunique"),
    spending_before_cutoff=("order_amount", "sum")
).reset_index()

# Lapsed = no completed order in the future period
future_customer_ids = set(future_orders["customer_id"].unique())

customer_history["lapsed"] = (
    ~customer_history["customer_id"].isin(future_customer_ids)
).astype(int)

print("\nLAPSE LABEL SUMMARY")
print(customer_history["lapsed"].value_counts().sort_index())
print("\nTotal eligible customers:", len(customer_history))
print("Lapsed customers:", int(customer_history["lapsed"].sum()))
print(
    "Not lapsed customers:",
    int((customer_history["lapsed"] == 0).sum())
)

# STEP 6: Build pre-cutoff customer features

customer_history["recency_days"] = (
    cutoff_date - customer_history["last_order_before_cutoff"]
).dt.days

customer_history["frequency"] = (
    customer_history["order_count_before_cutoff"]
)

customer_history["monetary"] = (
    customer_history["spending_before_cutoff"]
)

feature_columns = [
    "recency_days",
    "frequency",
    "monetary"
]

X = customer_history[feature_columns].copy()
y = customer_history["lapsed"].copy()

print("\nFEATURE CHECK")
print("Feature columns:", feature_columns)
print("Feature rows:", X.shape[0])
print("Feature missing values:")
print(X.isna().sum())
print("\nLabel counts:")
print(y.value_counts().sort_index())

# STEP 7: Split data into training and testing sets

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTRAIN / TEST SPLIT")
print("Training rows:", len(X_train))
print("Testing rows:", len(X_test))

print("\nTraining label distribution:")
print(y_train.value_counts().sort_index())

print("\nTesting label distribution:")
print(y_test.value_counts().sort_index())

# STEP 8: Train and evaluate two baseline models

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

# Logistic Regression pipeline
logistic_model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(
        max_iter=1000,
        random_state=42
    ))
])

# Random Forest pipeline
forest_model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("model", RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced"
    ))
])

# Train both models using training data only
logistic_model.fit(X_train, y_train)
forest_model.fit(X_train, y_train)

# Evaluate model predictions
def evaluate_model(name, model, X_eval, y_eval):
    predictions = model.predict(X_eval)
    probabilities = model.predict_proba(X_eval)[:, 1]

    return {
        "Model": name,
        "Accuracy": round(accuracy_score(y_eval, predictions), 3),
        "Precision": round(
            precision_score(y_eval, predictions, zero_division=0), 3
        ),
        "Recall": round(
            recall_score(y_eval, predictions, zero_division=0), 3
        ),
        "F1": round(
            f1_score(y_eval, predictions, zero_division=0), 3
        ),
        "ROC_AUC": round(roc_auc_score(y_eval, probabilities), 3)
    }

baseline_results = [
    evaluate_model("Logistic Regression", logistic_model, X_test, y_test),
    evaluate_model("Random Forest", forest_model, X_test, y_test)
]

baseline_table = pd.DataFrame(baseline_results)

print("\nBASELINE MODEL COMPARISON")
print(baseline_table.to_string(index=False))

# STEP 9: Oversample the lapsed class in training data only

training_data = X_train.copy()
training_data["lapsed"] = y_train.to_numpy()

majority_count = training_data["lapsed"].value_counts().max()

class_0 = training_data[training_data["lapsed"] == 0]
class_1 = training_data[training_data["lapsed"] == 1]

# Upsample the smaller class using training data only
class_1_oversampled = class_1.sample(
    n=majority_count,
    replace=True,
    random_state=42
)

balanced_training_data = pd.concat(
    [class_0, class_1_oversampled],
    ignore_index=True
).sample(frac=1, random_state=42).reset_index(drop=True)

X_train_balanced = balanced_training_data[feature_columns]
y_train_balanced = balanced_training_data["lapsed"]

print("\nOVERSAMPLING SUMMARY")
print("Before oversampling:")
print(y_train.value_counts().sort_index())

print("\nAfter oversampling:")
print(y_train_balanced.value_counts().sort_index())

# STEP 10: Retrain models using balanced training data

# Train Logistic Regression on balanced data
logistic_model.fit(X_train_balanced, y_train_balanced)

# Train Random Forest on balanced data
forest_model.fit(X_train_balanced, y_train_balanced)

# Evaluate on the ORIGINAL, unchanged test data
balanced_results = [
    evaluate_model(
        "Logistic Regression Balanced",
        logistic_model,
        X_test,
        y_test
    ),
    evaluate_model(
        "Random Forest Balanced",
        forest_model,
        X_test,
        y_test
    )
]

balanced_table = pd.DataFrame(balanced_results)

print("\nBALANCED MODEL COMPARISON")
print(balanced_table.to_string(index=False))

# STEP 11: Save the selected model and feature importance

import joblib

# Save the balanced Logistic Regression pipeline
MODEL_PATH = OUTPUT_DIR / "lapse_model.joblib"

joblib.dump(logistic_model, MODEL_PATH)

# Logistic Regression coefficients show feature influence
coefficient_values = (
    logistic_model.named_steps["model"].coef_[0]
)

importance_table = pd.DataFrame({
    "feature": feature_columns,
    "coefficient": coefficient_values
})

importance_table["abs_coefficient"] = (
    importance_table["coefficient"].abs()
)

importance_table = importance_table.sort_values(
    "abs_coefficient",
    ascending=False
)

print("\nFEATURE COEFFICIENTS")
print(importance_table.to_string(index=False))

print("\nMODEL SAVED:", MODEL_PATH)

# STEP 12: Create the Premium customer retention list

segments_file = OUTPUT_DIR / "customer_segments.csv"

segments = pd.read_csv(segments_file)

premium_customers = segments[
    segments["segment_name"] == "Premium"
].copy()

# Add lapse predictions using the saved model's feature format
premium_features = customer_history[
    customer_history["customer_id"].isin(
        premium_customers["customer_id"]
    )
][["customer_id"] + feature_columns].copy()

premium_features["lapse_probability"] = (
    logistic_model.predict_proba(
        premium_features[feature_columns]
    )[:, 1]
)

retention_list = premium_customers.merge(
    premium_features[["customer_id", "lapse_probability"]],
    on="customer_id",
    how="left"
)

retention_list = retention_list.sort_values(
    "lapse_probability",
    ascending=False
)

retention_path = OUTPUT_DIR / "retention_list.csv"
retention_list.to_csv(retention_path, index=False)

print("\nPREMIUM RETENTION LIST CREATED")
print("Premium customers:", len(retention_list))
print("Saved file:", retention_path)
print("\nTop 5 retention candidates:")
print(retention_list.head().to_string(index=False))
joblib.dump(logistic_model, MODEL_PATH)