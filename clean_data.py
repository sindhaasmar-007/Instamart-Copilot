
import pandas as pd
import sqlite3
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
RAW_DIR = BASE_DIR / "data" / "raw"
CLEAN_DIR = BASE_DIR / "data" / "clean"
CLEAN_DIR.mkdir(parents=True, exist_ok=True)

# Load raw data
customers = pd.read_csv(RAW_DIR / "customers.csv")
products = pd.read_csv(RAW_DIR / "products.csv")
orders = pd.read_csv(RAW_DIR / "orders.csv")

# -------------------------
# 1. Clean customers
# -------------------------
customers["age"] = pd.to_numeric(customers["age"], errors="coerce")
customers.loc[~customers["age"].between(10, 100), "age"] = pd.NA

customers["city"] = customers["city"].astype("string").str.strip().str.title()
customers["city"] = customers["city"].replace("", pd.NA)

customers["gender"] = (
    customers["gender"].astype("string").str.strip().str.lower()
)
customers["gender"] = customers["gender"].replace("", pd.NA)
customers["gender"] = customers["gender"].replace({
    "m": "male",
    "man": "male",
    "men": "male",
    "f": "female",
    "woman": "female",
    "women": "female",
})
customers["gender"] = customers["gender"].where(
    customers["gender"].isin(["male", "female"]), "unknown"
)

if "signup_date" in customers.columns:
    customers["signup_date"] = pd.to_datetime(
        customers["signup_date"], errors="coerce"
    )

customers["age"] = customers["age"].fillna(
    customers.groupby("city")["age"].transform("median")
)


# -------------------------
# 2. Clean products
# -------------------------
products["price_inr"] = pd.to_numeric(
    products["price_inr"], errors="coerce"
)

# -------------------------
# 3. Clean orders
# -------------------------
orders = orders.drop_duplicates().copy()

orders["order_ts"] = pd.to_datetime(orders["order_ts"], errors="coerce")
orders["order_amount"] = pd.to_numeric(
    orders["order_amount"], errors="coerce"
)
orders["delivery_minutes"] = pd.to_numeric(
    orders["delivery_minutes"], errors="coerce"
)

orders["status"] = (
    orders["status"].astype("string").str.strip().str.lower()
)
orders["status"] = orders["status"].replace({
    "complete": "completed",
    "cancel": "cancelled",
    "canceled": "cancelled",
})

# Keep the latest record for each order ID
orders = orders.sort_values("order_ts")
orders = orders.drop_duplicates(subset=["order_id"], keep="last").copy()

# Calculate missing amounts using product price, quantity and discount
price_lookup = products[["product_id", "price_inr"]].drop_duplicates(
    subset=["product_id"]
)
orders = orders.merge(price_lookup, on="product_id", how="left")

quantity = pd.to_numeric(orders["quantity"], errors="coerce")
discount = pd.to_numeric(
    orders["discount_pct"], errors="coerce"
).fillna(0)

# Convert percentage discount into a fraction
discount = discount / 100

calculated_amount = orders["price_inr"] * quantity * (1 - discount)

orders["order_amount"] = orders["order_amount"].fillna(calculated_amount)

orders = orders.drop(columns=["price_inr"])
# -------------------------
# 3. Clean orders
# -------------------------
orders = orders.drop_duplicates().copy()

orders["order_ts"] = pd.to_datetime(orders["order_ts"], errors="coerce")
orders["order_amount"] = pd.to_numeric(
    orders["order_amount"], errors="coerce"
)
orders["delivery_minutes"] = pd.to_numeric(
    orders["delivery_minutes"], errors="coerce"
)

orders["status"] = (
    orders["status"].astype("string").str.strip().str.lower()
)
orders["status"] = orders["status"].replace({
    "complete": "completed",
    "cancel": "cancelled",
    "canceled": "cancelled",
})

# Keep the latest record for each order ID
orders = orders.sort_values("order_ts")
orders = orders.drop_duplicates(subset=["order_id"], keep="last").copy()

# Fill missing order amounts from product price × quantity × (1 - discount)
product_price_col = (
    "price" if "price" in products.columns else "product_price"
)

if product_price_col in products.columns and "product_id" in orders.columns:
    price_lookup = products[["product_id", product_price_col]].drop_duplicates(
        subset=["product_id"]
    )
    orders = orders.merge(price_lookup, on="product_id", how="left")

    if "quantity" in orders.columns:
        quantity = pd.to_numeric(orders["quantity"], errors="coerce")
    else:
        quantity = pd.Series(1, index=orders.index, dtype="float64")

    if "discount" in orders.columns:
        discount = pd.to_numeric(orders["discount"], errors="coerce").fillna(0)
        discount = discount.where(discount <= 1, discount / 100)
    else:
        discount = pd.Series(0, index=orders.index, dtype="float64")

    calculated_amount = (
        orders[product_price_col] * quantity * (1 - discount)
    )
    orders["order_amount"] = orders["order_amount"].fillna(calculated_amount)
    orders = orders.drop(columns=[product_price_col])

orders.loc[orders["order_amount"] < 0, "order_amount"] = pd.NA
orders.loc[orders["delivery_minutes"] < 0, "delivery_minutes"] = pd.NA

# Fill missing delivery time for completed orders using completed-order median
if "category" in orders.columns:
    completed_median = orders.loc[
        orders["status"].eq("completed")
    ].groupby("category")["delivery_minutes"].transform("median")
    missing_completed = (
        orders["status"].eq("completed")
        & orders["delivery_minutes"].isna()
    )
    # Calculate each category's median from completed orders
    category_medians = orders.loc[
        orders["status"].eq("completed")
    ].groupby("category")["delivery_minutes"].median()
    orders.loc[missing_completed, "delivery_minutes"] = (
        orders.loc[missing_completed, "category"].map(category_medians)
    )
else:
    completed_median = orders.loc[
        orders["status"].eq("completed"), "delivery_minutes"
    ].median()
    mask = orders["status"].eq("completed") & orders["delivery_minutes"].isna()
    orders.loc[mask, "delivery_minutes"] = completed_median

# Pending orders are excluded from the cleaned dataset
orders = orders.loc[~orders["status"].eq("pending")].copy()

# Cancelled orders should not have a delivery time
orders.loc[orders["status"].eq("cancelled"), "delivery_minutes"] = pd.NA

# Save cleaned CSV files
customers.to_csv(CLEAN_DIR / "customers.csv", index=False)
products.to_csv(CLEAN_DIR / "products.csv", index=False)
orders.to_csv(CLEAN_DIR / "orders.csv", index=False)

# Save SQLite database
DB_PATH = BASE_DIR / "data" / "instamart.db"

with sqlite3.connect(DB_PATH) as conn:
    customers.to_sql("customers", conn, if_exists="replace", index=False)
    products.to_sql("products", conn, if_exists="replace", index=False)
    orders.to_sql("orders", conn, if_exists="replace", index=False)

print("Cleaned data saved to:", CLEAN_DIR)
print("SQLite database saved to:", DB_PATH)
print("Customers:", len(customers))
print("Products:", len(products))
print("Orders:", len(orders))
print("Order status counts:")
print(orders["status"].value_counts(dropna=False))
print("Completed order amount total:", round(
    orders.loc[orders["status"].eq("completed"), "order_amount"].sum(), 2
))
