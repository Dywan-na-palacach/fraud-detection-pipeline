-- Customer historical profile: median & 90th percentile of transaction amount
CREATE MATERIALIZED VIEW customer_amount_profile AS
SELECT 
    sender_account,
    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY amount) AS median_amount,
    PERCENTILE_CONT(0.9) WITHIN GROUP (ORDER BY amount) AS p90_amount
FROM transactions
GROUP BY sender_account;
