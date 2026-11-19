"""Generate a realistic synthetic loan default classification dataset."""

import os
import numpy as np
import pandas as pd


def generate_loan_data(
    n_rows: int = 5000,
    random_state: int = 42,
    output_path: str = "sample_data/loans.csv",
) -> pd.DataFrame:
    """
    Generate synthetic loan data with realistic relationships,
    mixed types, missing values, and ~15% default rate.
    """
    rng = np.random.RandomState(random_state)

    # 1. Feature generation
    age = rng.randint(21, 68, size=n_rows)
    annual_income = np.round(rng.lognormal(mean=10.9, sigma=0.45, size=n_rows), -2)
    annual_income = np.clip(annual_income, 18000, 220000)

    loan_amount = np.round(rng.uniform(2000, 35000, size=n_rows), -2)
    credit_score = rng.normal(loc=675, scale=55, size=n_rows).astype(int)
    credit_score = np.clip(credit_score, 520, 840)

    debt_to_income = np.round(rng.beta(2, 6, size=n_rows) * 0.65, 3)
    employment_years = np.round(rng.exponential(scale=5.0, size=n_rows), 1)
    employment_years = np.clip(employment_years, 0, 35)

    home_ownership = rng.choice(
        ["RENT", "MORTGAGE", "OWN"],
        size=n_rows,
        p=[0.45, 0.45, 0.10],
    )

    loan_purpose = rng.choice(
        ["debt_consolidation", "credit_card", "home_improvement", "major_purchase", "small_business"],
        size=n_rows,
        p=[0.50, 0.22, 0.13, 0.09, 0.06],
    )

    employment_type = rng.choice(
        ["salaried", "self_employed", "contractor", "unemployed"],
        size=n_rows,
        p=[0.70, 0.16, 0.10, 0.04],
    )

    # Interest rate depends on credit score
    interest_rate = np.round(
        28.0 - (credit_score - 520) / (840 - 520) * 18.0 + rng.normal(0, 1.5, size=n_rows),
        2,
    )
    interest_rate = np.clip(interest_rate, 5.0, 32.0)

    # 2. Realistic default probability model
    # Higher default with: low credit score, high DTI, high loan/income ratio, unemployed, high interest rate
    loan_to_income = loan_amount / annual_income
    logit = (
        -2.70
        - 0.009 * (credit_score - 675)
        + 3.2 * (debt_to_income - 0.20)
        + 1.8 * (loan_to_income - 0.35)
        + 0.08 * (interest_rate - 12.0)
        + 1.2 * (employment_type == "unemployed")
        + 0.4 * (home_ownership == "RENT")
    )

    prob_default = 1.0 / (1.0 + np.exp(-logit))
    # Calibrate intercept so average default rate is ~15%
    default = (rng.uniform(0, 1, size=n_rows) < prob_default).astype(int)

    # 3. Assemble DataFrame
    df = pd.DataFrame({
        "age": age,
        "annual_income": annual_income,
        "loan_amount": loan_amount,
        "credit_score": credit_score,
        "interest_rate": interest_rate,
        "debt_to_income": debt_to_income,
        "employment_years": employment_years,
        "home_ownership": home_ownership,
        "loan_purpose": loan_purpose,
        "employment_type": employment_type,
        "default": default,
    })

    # 4. Inject realistic missing values (2% to 4%)
    mask_income = rng.rand(n_rows) < 0.02
    mask_dti = rng.rand(n_rows) < 0.03
    mask_emp = rng.rand(n_rows) < 0.04
    mask_home = rng.rand(n_rows) < 0.015

    df.loc[mask_income, "annual_income"] = np.nan
    df.loc[mask_dti, "debt_to_income"] = np.nan
    df.loc[mask_emp, "employment_years"] = np.nan
    df.loc[mask_home, "home_ownership"] = np.nan

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Generated {len(df)} rows to {output_path}")
    print(f"Default rate: {df['default'].mean()*100:.2f}%")
    return df


if __name__ == "__main__":
    generate_loan_data()
