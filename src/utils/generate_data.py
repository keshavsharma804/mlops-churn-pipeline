import os
import pandas as pd
import numpy as np

def generate_churn_data(output_path="data/raw/churn.csv", rows=1000):
    np.random.seed(42)

    # numeric features
    tenure = np.random.randint(1, 72, rows)
    monthly_charges = np.random.uniform(20, 120, rows)
    total_charges = tenure * monthly_charges + np.random.uniform(-50, 50, rows)

    # categorical features
    contract_type = np.random.choice(["month-to-month", "one-year", "two-year"], rows)
    gender = np.random.choice(["male", "female"], rows)
    senior_citizen = np.random.choice([0, 1], rows)

    # churn probability logic
    churn_prob = (
        (tenure < 12) * 0.6 +
        (monthly_charges > 90) * 0.2 +
        (contract_type == "month-to-month") * 0.3 +
        (senior_citizen == 1) * 0.1
    ).clip(0, 1)

    churn = np.random.binomial(1, churn_prob)

    df = pd.DataFrame({
        "tenure": tenure,
        "monthly_charges": monthly_charges.round(2),
        "total_charges": total_charges.round(2),
        "contract_type": contract_type,
        "gender": gender,
        "senior_citizen": senior_citizen,
        "churn": churn
    })

    # ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    df.to_csv(output_path, index=False)
    print(f"Dataset created at: {output_path}")
    print(df.head())

if __name__ == "__main__":
    generate_churn_data()
