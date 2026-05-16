"""
generate_sensor_data.py
-----------------------
Creates realistic industrial sensor parquet files:
  - sensor_nodrift.parquet   : healthy machine (normal ops)
  - sensor_drift.parquet     : degraded machine (bearing wear, overheating)

10 sensors mapped to feature_1 .. feature_10:
  feature_1  = temperature (C)
  feature_2  = vibration (mm/s RMS)
  feature_3  = pressure (bar)
  feature_4  = current (A)
  feature_5  = rpm
  feature_6  = oil_temp (C)
  feature_7  = humidity (%RH)
  feature_8  = torque (Nm)
  feature_9  = voltage (V)
  feature_10 = coolant_flow (L/min)

label = 1 means fault / anomaly detected
"""
import numpy as np
import pandas as pd
from pathlib import Path

rng = np.random.default_rng(42)
N   = 2000
OUT = Path("data/sample")
OUT.mkdir(parents=True, exist_ok=True)

# ── NORMAL operating conditions ────────────────────────────────
temperature  = rng.normal(75,   5,  N)
vibration    = rng.normal(2.5,  0.4,N)
pressure     = rng.normal(4.2,  0.3,N)
current      = rng.normal(12,   1,  N)
rpm          = rng.normal(1480, 30, N)
oil_temp     = rng.normal(55,   4,  N)
humidity     = rng.normal(45,   5,  N)
torque       = rng.normal(98,   6,  N)
voltage      = rng.normal(400,  5,  N)
coolant_flow = rng.normal(8.5,  0.5,N)

df_ref = pd.DataFrame({
    "feature_1":  temperature.round(2),
    "feature_2":  vibration.round(3),
    "feature_3":  pressure.round(3),
    "feature_4":  current.round(2),
    "feature_5":  rpm.round(1),
    "feature_6":  oil_temp.round(2),
    "feature_7":  humidity.round(1),
    "feature_8":  torque.round(2),
    "feature_9":  voltage.round(1),
    "feature_10": coolant_flow.round(3),
})
df_ref["label"]     = ((df_ref["feature_1"] > 82) & (df_ref["feature_2"] > 3.1)).astype(int)
df_ref["timestamp"] = "2025-01-01"
df_ref["sensor_id"] = ["MACH-" + str(i % 10 + 1).zfill(2) for i in range(N)]

df_ref.to_parquet(OUT / "sensor_nodrift.parquet", index=False)
print("[OK] sensor_nodrift.parquet  fault_rate=" + str(round(df_ref["label"].mean(), 3)))

# ── DRIFTED conditions: machine wearing out ────────────────────
temperature_d  = rng.normal(90,   8,  N)   # HIGH  overheating
vibration_d    = rng.normal(4.8,  0.8,N)   # HIGH  bearing wear
pressure_d     = rng.normal(3.5,  0.5,N)   # LOW   pressure drop
current_d      = rng.normal(14,   2,  N)   # HIGH  drawing more power
rpm_d          = rng.normal(1420, 80, N)   # UNSTABLE speed fluctuation
oil_temp_d     = rng.normal(68,   7,  N)   # HIGH  oil degrading
humidity_d     = rng.normal(45,   5,  N)   # unchanged
torque_d       = rng.normal(112,  10, N)   # HIGH  mechanical stress
voltage_d      = rng.normal(395,  10, N)   # slightly low
coolant_flow_d = rng.normal(7.2,  0.8,N)  # LOW   coolant issue

df_drift = pd.DataFrame({
    "feature_1":  temperature_d.round(2),
    "feature_2":  vibration_d.round(3),
    "feature_3":  pressure_d.round(3),
    "feature_4":  current_d.round(2),
    "feature_5":  rpm_d.round(1),
    "feature_6":  oil_temp_d.round(2),
    "feature_7":  humidity_d.round(1),
    "feature_8":  torque_d.round(2),
    "feature_9":  voltage_d.round(1),
    "feature_10": coolant_flow_d.round(3),
})
df_drift["label"]     = ((df_drift["feature_1"] > 85) | (df_drift["feature_2"] > 4.5)).astype(int)
df_drift["timestamp"] = "2026-05-15"
df_drift["sensor_id"] = ["MACH-" + str(i % 10 + 1).zfill(2) for i in range(N)]

df_drift.to_parquet(OUT / "sensor_drift.parquet", index=False)
print("[OK] sensor_drift.parquet    fault_rate=" + str(round(df_drift["label"].mean(), 3)))

# ── Summary table ──────────────────────────────────────────────
print("")
print("Sensor         Normal    Drifted   Shift")
print("-" * 45)
pairs = [
    ("Temperature C",  "feature_1"),
    ("Vibration mm/s", "feature_2"),
    ("Pressure bar",   "feature_3"),
    ("Current A",      "feature_4"),
    ("RPM",            "feature_5"),
    ("Oil Temp C",     "feature_6"),
    ("Torque Nm",      "feature_8"),
    ("Coolant L/min",  "feature_10"),
]
for name, col in pairs:
    n = df_ref[col].mean()
    d = df_drift[col].mean()
    arrow = "^^" if d > n * 1.05 else ("vv" if d < n * 0.95 else "==")
    print(f"  {name:16s}  {n:7.2f}   {d:7.2f}   {arrow}")
