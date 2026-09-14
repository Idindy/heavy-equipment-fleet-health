from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import roc_auc_score, classification_report
from simulate_fleet import simulate_fleet

FEATURES = [
    "load", "temperature", "vibration", "oil_pressure",
    "hydraulic_pressure", "fuel_rate", "battery_voltage",
]


def machine_split(df, test_fraction=0.25, seed=42):
    rng = np.random.default_rng(seed)
    assets = df["asset_id"].unique()
    rng.shuffle(assets)
    n_test = max(1, int(len(assets) * test_fraction))
    test_assets = set(assets[:n_test])
    return df[~df.asset_id.isin(test_assets)], df[df.asset_id.isin(test_assets)]


def risk_score(df):
    temp_risk = np.clip((df["temperature"] - 92) / 30, 0, 1)
    vib_risk = np.clip((df["vibration"] - 0.55) / 1.1, 0, 1)
    hydraulic_risk = np.clip((2400 - df["hydraulic_pressure"]) / 1000, 0, 1)
    oil_risk = np.clip((40 - df["oil_pressure"]) / 12, 0, 1)
    return 100 * (0.30 * temp_risk + 0.30 * vib_risk + 0.25 * hydraulic_risk + 0.15 * oil_risk)


def main():
    Path("outputs").mkdir(exist_ok=True)
    data_path = Path("data/fleet_telemetry.csv")
    df = pd.read_csv(data_path) if data_path.exists() else simulate_fleet()
    train_df, test_df = machine_split(df)

    iso = IsolationForest(n_estimators=250, contamination=0.08, random_state=42, n_jobs=-1)
    iso.fit(train_df[FEATURES])
    test_df = test_df.copy()
    test_df["anomaly_score"] = -iso.decision_function(test_df[FEATURES])
    test_df["risk_score"] = risk_score(test_df)

    clf = RandomForestClassifier(
        n_estimators=300, max_depth=12, min_samples_leaf=2,
        class_weight="balanced", random_state=42, n_jobs=-1
    )
    clf.fit(train_df[FEATURES], train_df["fault"])
    prob = clf.predict_proba(test_df[FEATURES])[:, 1]
    pred = (prob >= 0.5).astype(int)
    auc = roc_auc_score(test_df["fault"], prob)
    metrics = {
        "roc_auc": float(auc),
        "train_assets": int(train_df.asset_id.nunique()),
        "test_assets": int(test_df.asset_id.nunique()),
    }
    print(json.dumps(metrics, indent=2))
    print(classification_report(test_df["fault"], pred, digits=3))
    Path("outputs/metrics.json").write_text(json.dumps(metrics, indent=2))

    latest = test_df.sort_values("hour").groupby("asset_id", as_index=False).tail(1)
    latest.sort_values("risk_score", ascending=False)[[
        "asset_id", "equipment_type", "risk_score", "anomaly_score", "fault", "fault_type"
    ]].to_csv("outputs/maintenance_priority.csv", index=False)


if __name__ == "__main__":
    main()
