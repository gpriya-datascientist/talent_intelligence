"""
generate_sample_data.py
-----------------------
Creates reference.parquet, production_nodrift.parquet,
and production_drift.parquet in data/sample/.

Run once before demo:
    python scripts/generate_sample_data.py
"""
import numpy as np
import pandas as pd
from pathlib import Path

OUT = Path("data/sample")
OUT.mkdir(parents=True, exist_ok=True)

N, FEATURES, SEED = 10_000, 10, 42
rng = np.random.default_rng(SEED)
COLS = [f"feature_{i}" for i in range(1, FEATURES + 1)]


def make_df(mean: float, std: float, label_fn, tag: str) -> pd.DataFrame:
    data = rng.normal(mean, std, size=(N, FEATURES))
    df = pd.DataFrame(data, columns=COLS).round(4)
    df["label"] = label_fn(df).astype(int)
    df["timestamp"] = tag
    return df


# Reference — N(0,1)
ref = make_df(0, 1, lambda d: (d["feature_1"] + d["feature_2"] > 0), "2025-01-01")
ref.to_parquet(OUT / "reference.parquet", index=False)
print(f"[OK] reference.parquet         {len(ref):,} rows  mean~{ref['feature_1'].mean():.3f}")

# No-drift production — same distribution
prod_ok = make_df(0, 1, lambda d: (d["feature_1"] + d["feature_2"] > 0), "2026-05-15")
prod_ok.to_parquet(OUT / "production_nodrift.parquet", index=False)
print(f"[OK] production_nodrift.parquet {len(prod_ok):,} rows  mean~{prod_ok['feature_1'].mean():.3f}")

# Drifted production — mean shifted to N(3,1)
prod_drift = make_df(3, 1, lambda d: (d["feature_1"] > 3).astype(int), "2026-05-15")
prod_drift.to_parquet(OUT / "production_drift.parquet", index=False)
print(f"[OK] production_drift.parquet  {len(prod_drift):,} rows  mean~{prod_drift['feature_1'].mean():.3f}  <- drift!")
