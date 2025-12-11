-- Rolling average amount per sender (last 5 transactions)
SELECT
    transaction_id,
    sender_account,
    amount,
    AVG(amount) OVER (
        PARTITION BY sender_account
        ORDER BY timestamp
        ROWS BETWEEN 5 PRECEDING AND 1 PRECEDING
    ) AS rolling_avg_amount_5
FROM transactions;


-- Z-score of amount per sender
SELECT
    t.*,
    (amount - AVG(amount) OVER (PARTITION BY sender_account))
    / NULLIF(STDDEV_POP(amount) OVER (PARTITION BY sender_account), 0) AS amount_zscore
FROM transactions t;
