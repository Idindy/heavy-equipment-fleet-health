# Heavy Equipment Fleet Simulator & Health Monitoring

A synthetic heavy-equipment telematics project for simulating fleet telemetry, detecting anomalies, classifying developing faults, and ranking assets for maintenance attention.

## Fleet simulation
The simulator generates hourly telemetry for excavators, bulldozers, and wheel loaders. Signals include temperature, vibration, oil pressure, hydraulic pressure, fuel rate, battery voltage, and load. The simulation includes progressive cooling, hydraulic, and bearing faults.

## Modeling
- Isolation Forest for unsupervised anomaly detection
- Random Forest for fault/no-fault classification
- Equipment-level train/test split
- ROC AUC evaluation
- Composite health/risk score for maintenance prioritization

## Tech
Python, pandas, NumPy, scikit-learn, IoT/telematics simulation, anomaly detection, reliability analytics.

## Run
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python src/simulate_fleet.py
PYTHONPATH=src python src/health_model.py
```

Outputs are written to `outputs/metrics.json` and `outputs/maintenance_priority.csv`.

## Resume summary
Built a fleet simulator generating time-series telemetry for excavators, bulldozers, and wheel loaders, modeling temperature, vibration, oil/hydraulic pressure, fuel, battery voltage, load, and progressive equipment faults. Applied Isolation Forest anomaly detection and Random Forest fault classification and developed an equipment risk score to prioritize maintenance.
