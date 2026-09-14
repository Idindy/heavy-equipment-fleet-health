import numpy as np
import pandas as pd

EQUIPMENT_TYPES = ["excavator", "bulldozer", "wheel_loader"]


def simulate_fleet(n_assets=90, hours=220, seed=7):
    rng = np.random.default_rng(seed)
    rows = []
    for asset_id in range(1, n_assets + 1):
        equipment_type = rng.choice(EQUIPMENT_TYPES)
        fault_start = int(rng.integers(int(hours * 0.55), int(hours * 0.9)))
        has_fault = rng.random() < 0.65
        fault_kind = rng.choice(["cooling", "hydraulic", "bearing"]) if has_fault else "none"

        for hour in range(hours):
            progress = max(0, hour - fault_start) / max(1, hours - fault_start)
            active_fault = has_fault and hour >= fault_start
            load = np.clip(rng.normal(0.62, 0.16), 0.15, 1.0)
            temperature = 76 + 17 * load + rng.normal(0, 2)
            vibration = 0.28 + 0.34 * load + rng.normal(0, 0.05)
            oil_pressure = 47 - 4 * load + rng.normal(0, 1.2)
            hydraulic_pressure = 2650 + 420 * load + rng.normal(0, 90)
            fuel_rate = 7.5 + 8 * load + rng.normal(0, 0.7)
            battery_voltage = 27.4 + rng.normal(0, 0.35)

            if active_fault:
                if fault_kind == "cooling":
                    temperature += 24 * progress + 5
                elif fault_kind == "hydraulic":
                    hydraulic_pressure -= 900 * progress + 180
                    oil_pressure -= 8 * progress
                elif fault_kind == "bearing":
                    vibration += 1.0 * progress + 0.18

            rows.append({
                "asset_id": asset_id, "equipment_type": equipment_type,
                "hour": hour, "load": load, "temperature": temperature,
                "vibration": vibration, "oil_pressure": oil_pressure,
                "hydraulic_pressure": hydraulic_pressure, "fuel_rate": fuel_rate,
                "battery_voltage": battery_voltage, "fault": int(active_fault),
                "fault_type": fault_kind if active_fault else "none",
            })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = simulate_fleet()
    df.to_csv("data/fleet_telemetry.csv", index=False)
    print(df.head())
    print(f"Generated {len(df):,} telemetry records.")
