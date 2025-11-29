from src.monitoring.drift import check_drift
from src.training import train as train_module


def main():
    drift_info = check_drift()
    print("Drift info:", drift_info)

    if not drift_info["has_drift"]:
        print("No significant drift. Skipping retraining.")
        return

    print("Drift detected! Starting retraining...")
    train_module.main()
    print("Retraining complete.")


if __name__ == "__main__":
    main()
