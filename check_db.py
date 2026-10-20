
import sqlite3

conn = sqlite3.connect("data/instamart.db")

tables = conn.execute(
    "SELECT name FROM sqlite_master WHERE type='table'"
).fetchall()
print("Tables:", tables)

counts = conn.execute(
    "SELECT status, COUNT(*) FROM orders GROUP BY status"
).fetchall()
print("Order counts:", counts)

revenue = conn.execute(
    "SELECT ROUND(SUM(order_amount), 2) "
    "FROM orders WHERE status='completed'"
).fetchone()
print("Completed revenue:", revenue)

conn.close()
