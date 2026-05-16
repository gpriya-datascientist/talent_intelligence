"""
train_baseline_model.py
-----------------------
Trains a baseline GBM on reference.parquet and saves to
data/models/baseline_model.pkl.

Run once after generate_sample_data.py:
    python scripts/train_baseline_model.py
"""
import pickle
import pandas as pd
from pathlib import Path
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

df = pd.read_parquet("data/sample/reference.parquet")
feature_cols = [c for c in df.columns if c.startswith("feature_")]

X = df[feature_cols].values
y = df["label"].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = GradientBoostingClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
acc = accuracy_score(y_test, model.predict(X_test))

out = Path("data/models/baseline_model.pkl")
out.parent.mkdir(parents=True, exist_ok=True)
with open(out, "wb") as f:
    pickle.dump(model, f)

print(f"[OK] baseline_model.pkl saved   test_accuracy={acc:.4f}")
