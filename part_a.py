
import pandas as pd
from pathlib import Path

# Project folders
BASE_DIR = Path(__file__).resolve().parent
RAW_DIR = BASE_DIR / "data" / "raw"
OUTPUT_DIR = BASE_DIR / "outputs"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load raw CSV files
customers = pd.read_csv(RAW_DIR / "customers.csv")
products = pd.read_csv(RAW_DIR / "products.csv")
orders = pd.read_csv(RAW_DIR / "orders.csv")

# Audit each table
for name, df in [
    ("customers", customers),
    ("products", products),
    ("orders", orders),
]:
    print(f"\n--- {name.upper()} AUDIT ---")
    print("Rows and columns:", df.shape)
    print("\nData types:")
    print(df.dtypes)
    print("\nMissing values:")
    missing = pd.DataFrame({
        "missing_count": df.isna().sum(),
        "missing_percent": (df.isna().mean() * 100).round(2),
    })
    print(missing)

# Required raw-order checks
print("\n--- ORDER CHECKS ---")
print("Raw order rows:", len(orders))
print("Exact duplicate rows:", orders.duplicated().sum())
print(
    "Pending rows:",
    orders["status"].astype(str).str.strip().str.lower().eq("pending").sum()
)
print("Distinct order IDs:", orders["order_id"].nunique())
print(
    "Non-numeric order amounts:",
    pd.to_numeric(orders["order_amount"], errors="coerce").isna().sum()
)
delivery = pd.to_numeric(orders["delivery_minutes"], errors="coerce")
print("Negative delivery times:", delivery.lt(0).sum())
print("Blank delivery times:", delivery.isna().sum())

# Required customer checks
print("\n--- CUSTOMER CHECKS ---")
print("Blank ages:", customers["age"].isna().sum())
age = pd.to_numeric(customers["age"], errors="coerce")
print("Ages outside 10-100:", ((age < 10) | (age > 100)).sum())
print(
    "Distinct raw city spellings:",
    customers["city"].dropna().nunique()
)
gender = customers["gender"].fillna("").astype(str).str.strip()
print(
    "Distinct non-blank gender spellings:",
    gender[gender.ne("")].nunique()
)
print("Blank genders:", gender.eq("").sum())

print("\nRaw data audit completed.")
