from src.load_data import load_transactions
from src.features import build_X_y

df = load_transactions()
X, y, feats = build_X_y(df)

print("X shape:", X.shape)
print("Fraud rate:", y.mean())
print("Sample features:", feats[:10])
