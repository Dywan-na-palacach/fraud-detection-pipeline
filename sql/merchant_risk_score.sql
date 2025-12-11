-- Merchant risk score based on historical fraud rate
CREATE MATERIALIZED VIEW merchant_risk_score AS
SELECT 
    merchant_category,
    COUNT(*) AS n_tx,
    SUM(CASE WHEN is_fraud THEN 1 ELSE 0 END) AS n_fraud,
    SUM(CASE WHEN is_fraud THEN 1 ELSE 0 END)::NUMERIC / COUNT(*) AS fraud_rate
FROM transactions
GROUP BY merchant_category;
