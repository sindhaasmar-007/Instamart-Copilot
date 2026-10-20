
-- Instamart Customer Intelligence & Support Copilot
-- Part A: SQL Analysis Queries

-- Query 1: Total customers
SELECT COUNT(*) AS total_customers
FROM customers;

-- Query 2: Total products
SELECT COUNT(*) AS total_products
FROM products;

-- Query 3: Total orders
SELECT COUNT(*) AS total_orders
FROM orders;

-- Query 4: Orders grouped by status
SELECT status, COUNT(*) AS order_count
FROM orders
GROUP BY status
ORDER BY status;

-- Query 5: Completed order revenue
SELECT ROUND(SUM(order_amount), 2) AS completed_revenue_inr
FROM orders
WHERE status = 'completed';

-- Query 6: Average order amount for completed orders
SELECT ROUND(AVG(order_amount), 2) AS average_order_amount
FROM orders
WHERE status = 'completed';

-- Query 7: Orders by product
SELECT
    product_id,
    COUNT(*) AS order_count,
    SUM(quantity) AS total_quantity
FROM orders
GROUP BY product_id
ORDER BY order_count DESC;

-- Query 8: Orders by customer
SELECT
    customer_id,
    COUNT(*) AS total_orders
FROM orders
GROUP BY customer_id
ORDER BY total_orders DESC
LIMIT 10;

-- Query 9: Completed orders by month
SELECT
    strftime('%Y-%m', order_ts) AS order_month,
    COUNT(*) AS completed_orders,
    ROUND(SUM(order_amount), 2) AS revenue_inr
FROM orders
WHERE status = 'completed'
GROUP BY strftime('%Y-%m', order_ts)
ORDER BY order_month;

-- Query 10: Average delivery time by order status
SELECT
    status,
    ROUND(AVG(delivery_minutes), 2) AS avg_delivery_minutes
FROM orders
GROUP BY status;
