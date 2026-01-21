import argparse
import random
from datetime import datetime, timedelta
from pathlib import Path
from typing import Literal

import pandas as pd
import numpy as np

CREDIT_CATEGORIES = [
    "Salary", "Business_Income", "Rental_Income",
    "Investment_Returns", "Refunds", "Transfers_In", "Other_Credit"
]

DEBIT_CATEGORIES = [
    "Groceries", "Utilities", "Rent", "EMI", "Insurance",
    "Entertainment", "Shopping", "Travel", "Healthcare",
    "Education", "Transfers_Out", "Cash_Withdrawal", "Other_Debit"
]

TRANSACTION_MODES = ["UPI", "NEFT", "RTGS", "IMPS", "CASH", "CARD", "CHEQUE", "AUTO_DEBIT"]

PersonaType = Literal[
    "salaried", "business_owner", "gig_worker",
    "investor", "pensioner", "student", "hybrid"
]


class CustomerProfile:
    def __init__(
        self,
        persona: PersonaType,
        monthly_income_range: tuple[float, float],
        income_sources: dict[str, float],
        expense_pattern: dict[str, tuple[float, float]],
        income_volatility: float = 0.1,
        emi_count: int = 0,
        emi_amount_range: tuple[float, float] = (0, 0),
    ):
        self.persona = persona
        self.monthly_income_range = monthly_income_range
        self.income_sources = income_sources
        self.expense_pattern = expense_pattern
        self.income_volatility = income_volatility
        self.emi_count = emi_count
        self.emi_amount_range = emi_amount_range


PERSONA_PROFILES: dict[PersonaType, CustomerProfile] = {
    "salaried": CustomerProfile(
        persona="salaried",
        monthly_income_range=(40000, 150000),
        income_sources={"Salary": 0.9, "Investment_Returns": 0.05, "Refunds": 0.05},
        expense_pattern={
            "Rent": (0.15, 0.30),
            "EMI": (0.10, 0.25),
            "Groceries": (0.08, 0.15),
            "Utilities": (0.03, 0.06),
            "Insurance": (0.02, 0.05),
            "Entertainment": (0.03, 0.08),
            "Shopping": (0.05, 0.12),
            "Travel": (0.02, 0.08),
            "Healthcare": (0.01, 0.04),
            "Transfers_Out": (0.05, 0.15),
            "Cash_Withdrawal": (0.05, 0.10),
        },
        income_volatility=0.05,
        emi_count=2,
        emi_amount_range=(5000, 25000),
    ),
    "business_owner": CustomerProfile(
        persona="business_owner",
        monthly_income_range=(80000, 500000),
        income_sources={"Business_Income": 0.85, "Investment_Returns": 0.10, "Other_Credit": 0.05},
        expense_pattern={
            "Rent": (0.05, 0.15),
            "EMI": (0.05, 0.15),
            "Groceries": (0.03, 0.08),
            "Utilities": (0.02, 0.05),
            "Insurance": (0.02, 0.04),
            "Entertainment": (0.02, 0.06),
            "Shopping": (0.03, 0.10),
            "Travel": (0.03, 0.10),
            "Healthcare": (0.01, 0.03),
            "Transfers_Out": (0.15, 0.30),
            "Cash_Withdrawal": (0.10, 0.20),
            "Other_Debit": (0.05, 0.15),
        },
        income_volatility=0.35,
        emi_count=1,
        emi_amount_range=(15000, 50000),
    ),
    "gig_worker": CustomerProfile(
        persona="gig_worker",
        monthly_income_range=(20000, 80000),
        income_sources={
            "Business_Income": 0.40,
            "Transfers_In": 0.35,
            "Other_Credit": 0.20,
            "Refunds": 0.05
        },
        expense_pattern={
            "Rent": (0.20, 0.35),
            "Groceries": (0.10, 0.18),
            "Utilities": (0.04, 0.08),
            "Entertainment": (0.05, 0.12),
            "Shopping": (0.05, 0.10),
            "Travel": (0.08, 0.15),
            "Healthcare": (0.02, 0.05),
            "Transfers_Out": (0.03, 0.10),
            "Cash_Withdrawal": (0.08, 0.15),
        },
        income_volatility=0.45,
        emi_count=0,
        emi_amount_range=(0, 0),
    ),
    "investor": CustomerProfile(
        persona="investor",
        monthly_income_range=(100000, 400000),
        income_sources={
            "Investment_Returns": 0.50,
            "Rental_Income": 0.30,
            "Salary": 0.15,
            "Other_Credit": 0.05
        },
        expense_pattern={
            "Rent": (0.0, 0.05),
            "Groceries": (0.03, 0.06),
            "Utilities": (0.02, 0.04),
            "Insurance": (0.03, 0.06),
            "Entertainment": (0.03, 0.08),
            "Shopping": (0.05, 0.12),
            "Travel": (0.05, 0.15),
            "Healthcare": (0.02, 0.05),
            "Transfers_Out": (0.15, 0.30),
            "Cash_Withdrawal": (0.03, 0.08),
        },
        income_volatility=0.25,
        emi_count=0,
        emi_amount_range=(0, 0),
    ),
    "pensioner": CustomerProfile(
        persona="pensioner",
        monthly_income_range=(25000, 60000),
        income_sources={
            "Salary": 0.70,
            "Investment_Returns": 0.20,
            "Rental_Income": 0.10
        },
        expense_pattern={
            "Groceries": (0.12, 0.20),
            "Utilities": (0.05, 0.10),
            "Healthcare": (0.08, 0.18),
            "Insurance": (0.03, 0.06),
            "Entertainment": (0.02, 0.05),
            "Shopping": (0.03, 0.08),
            "Travel": (0.02, 0.06),
            "Transfers_Out": (0.10, 0.20),
            "Cash_Withdrawal": (0.08, 0.15),
        },
        income_volatility=0.03,
        emi_count=0,
        emi_amount_range=(0, 0),
    ),
    "student": CustomerProfile(
        persona="student",
        monthly_income_range=(10000, 30000),
        income_sources={
            "Transfers_In": 0.70,
            "Salary": 0.20,
            "Refunds": 0.10
        },
        expense_pattern={
            "Rent": (0.25, 0.40),
            "Groceries": (0.15, 0.25),
            "Utilities": (0.03, 0.06),
            "Education": (0.10, 0.20),
            "Entertainment": (0.08, 0.15),
            "Shopping": (0.05, 0.12),
            "Travel": (0.03, 0.08),
            "Cash_Withdrawal": (0.05, 0.12),
        },
        income_volatility=0.20,
        emi_count=0,
        emi_amount_range=(0, 0),
    ),
    "hybrid": CustomerProfile(
        persona="hybrid",
        monthly_income_range=(50000, 150000),
        income_sources={
            "Salary": 0.50,
            "Business_Income": 0.25,
            "Investment_Returns": 0.15,
            "Rental_Income": 0.10
        },
        expense_pattern={
            "Rent": (0.10, 0.20),
            "EMI": (0.08, 0.18),
            "Groceries": (0.06, 0.12),
            "Utilities": (0.03, 0.06),
            "Insurance": (0.02, 0.05),
            "Entertainment": (0.04, 0.10),
            "Shopping": (0.05, 0.12),
            "Travel": (0.04, 0.10),
            "Healthcare": (0.02, 0.04),
            "Transfers_Out": (0.08, 0.15),
            "Cash_Withdrawal": (0.05, 0.10),
        },
        income_volatility=0.15,
        emi_count=1,
        emi_amount_range=(8000, 20000),
    ),
}


def generate_customer_id(index: int) -> str:
    return f"CUST{index:04d}"


def get_transaction_mode(category: str, is_credit: bool) -> str:
    if category == "Salary":
        return random.choice(["NEFT", "RTGS", "IMPS"])
    elif category in ["Business_Income", "Rental_Income"]:
        return random.choice(["NEFT", "RTGS", "IMPS", "CHEQUE", "UPI"])
    elif category == "Investment_Returns":
        return random.choice(["NEFT", "RTGS"])
    elif category == "Transfers_In":
        return random.choice(["UPI", "NEFT", "IMPS"])
    elif category == "Refunds":
        return random.choice(["UPI", "NEFT", "CARD"])
    elif category == "Cash_Withdrawal":
        return "CASH"
    elif category == "EMI":
        return random.choice(["AUTO_DEBIT", "NEFT"])
    elif category == "Rent":
        return random.choice(["NEFT", "UPI", "CHEQUE"])
    elif category in ["Utilities", "Insurance"]:
        return random.choice(["AUTO_DEBIT", "UPI", "NEFT"])
    elif category in ["Groceries", "Shopping", "Entertainment", "Travel", "Healthcare"]:
        return random.choice(["UPI", "CARD", "CASH"])
    elif category == "Education":
        return random.choice(["NEFT", "UPI", "CARD"])
    elif category == "Transfers_Out":
        return random.choice(["UPI", "NEFT", "IMPS"])
    else:
        return random.choice(TRANSACTION_MODES)


def generate_income_transactions(
    profile: CustomerProfile,
    base_monthly_income: float,
    month_start: datetime,
    month_end: datetime,
) -> list[dict]:
    transactions = []
    monthly_income = base_monthly_income * (1 + random.gauss(0, profile.income_volatility))
    monthly_income = max(monthly_income, profile.monthly_income_range[0] * 0.5)
    remaining_income = monthly_income

    for category, weight in profile.income_sources.items():
        if random.random() > weight * 1.5:
            continue

        category_income = monthly_income * weight * random.uniform(0.8, 1.2)
        category_income = min(category_income, remaining_income)

        if category_income < 100:
            continue

        remaining_income -= category_income

        if category == "Salary":
            num_txns = 1
            day = random.randint(1, 5) if random.random() > 0.3 else random.randint(25, 28)
        elif category == "Business_Income":
            num_txns = random.randint(3, 15)
        elif category == "Rental_Income":
            num_txns = 1
            day = random.randint(1, 10)
        elif category == "Investment_Returns":
            num_txns = random.randint(1, 3)
        else:
            num_txns = random.randint(1, 5)

        amounts = np.random.dirichlet(np.ones(num_txns)) * category_income

        for amount in amounts:
            if amount < 50:
                continue

            if category == "Salary" or category == "Rental_Income":
                txn_date = month_start + timedelta(days=day - 1)
            else:
                txn_date = month_start + timedelta(days=random.randint(0, (month_end - month_start).days))

            transactions.append({
                "dr_cr_indctor": "CR",
                "tran_date": txn_date.strftime("%Y-%m-%d"),
                "tran_amt_in_ac": round(amount, 2),
                "tran_type": get_transaction_mode(category, is_credit=True),
                "category_of_txn": category,
            })

    return transactions


def generate_expense_transactions(
    profile: CustomerProfile,
    monthly_income: float,
    month_start: datetime,
    month_end: datetime,
    emi_amounts: list[float],
) -> list[dict]:
    transactions = []

    for i, emi_amount in enumerate(emi_amounts):
        emi_day = 5 + i * 3
        txn_date = month_start + timedelta(days=min(emi_day, 28) - 1)
        transactions.append({
            "dr_cr_indctor": "DR",
            "tran_date": txn_date.strftime("%Y-%m-%d"),
            "tran_amt_in_ac": round(emi_amount, 2),
            "tran_type": get_transaction_mode("EMI", is_credit=False),
            "category_of_txn": "EMI",
        })

    for category, (min_pct, max_pct) in profile.expense_pattern.items():
        if category == "EMI":
            continue

        expense_pct = random.uniform(min_pct, max_pct)
        category_expense = monthly_income * expense_pct

        if category_expense < 50:
            continue

        if category == "Rent":
            num_txns = 1
            amounts = [category_expense]
            days = [random.randint(1, 5)]
        elif category in ["Utilities", "Insurance"]:
            num_txns = random.randint(1, 3)
            amounts = np.random.dirichlet(np.ones(num_txns)) * category_expense
            days = [random.randint(1, 28) for _ in range(num_txns)]
        elif category == "Groceries":
            num_txns = random.randint(4, 12)
            amounts = np.random.dirichlet(np.ones(num_txns)) * category_expense
            days = [random.randint(1, 28) for _ in range(num_txns)]
        elif category in ["Entertainment", "Shopping"]:
            num_txns = random.randint(2, 8)
            amounts = np.random.dirichlet(np.ones(num_txns)) * category_expense
            days = [random.randint(1, 28) for _ in range(num_txns)]
        elif category == "Cash_Withdrawal":
            num_txns = random.randint(2, 6)
            amounts = np.random.dirichlet(np.ones(num_txns)) * category_expense
            days = [random.randint(1, 28) for _ in range(num_txns)]
        else:
            num_txns = random.randint(1, 5)
            amounts = np.random.dirichlet(np.ones(num_txns)) * category_expense
            days = [random.randint(1, 28) for _ in range(num_txns)]

        for amount, day in zip(amounts, days):
            if amount < 10:
                continue

            txn_date = month_start + timedelta(days=min(day, (month_end - month_start).days))
            transactions.append({
                "dr_cr_indctor": "DR",
                "tran_date": txn_date.strftime("%Y-%m-%d"),
                "tran_amt_in_ac": round(amount, 2),
                "tran_type": get_transaction_mode(category, is_credit=False),
                "category_of_txn": category,
            })

    return transactions


def generate_customer_transactions(
    cust_id: str,
    persona: PersonaType,
    start_date: datetime,
    num_months: int,
) -> list[dict]:
    profile = PERSONA_PROFILES[persona]
    transactions = []
    base_income = random.uniform(*profile.monthly_income_range)

    emi_amounts = [
        random.uniform(*profile.emi_amount_range)
        for _ in range(profile.emi_count)
    ] if profile.emi_count > 0 else []

    current_date = start_date

    for month_idx in range(num_months):
        month_start = current_date.replace(day=1)
        if month_start.month == 12:
            month_end = month_start.replace(year=month_start.year + 1, month=1) - timedelta(days=1)
        else:
            month_end = month_start.replace(month=month_start.month + 1) - timedelta(days=1)

        trend_factor = 1 + (month_idx * random.uniform(-0.02, 0.03))
        monthly_base_income = base_income * trend_factor

        income_txns = generate_income_transactions(
            profile, monthly_base_income, month_start, month_end
        )

        actual_monthly_income = sum(t["tran_amt_in_ac"] for t in income_txns)

        expense_txns = generate_expense_transactions(
            profile, actual_monthly_income, month_start, month_end, emi_amounts
        )

        for txn in income_txns + expense_txns:
            txn["cust_id"] = cust_id
            transactions.append(txn)

        if current_date.month == 12:
            current_date = current_date.replace(year=current_date.year + 1, month=1)
        else:
            current_date = current_date.replace(month=current_date.month + 1)

    return transactions


def generate_sample_data(
    num_customers: int = 5,
    num_months: int = 6,
    start_date: datetime | None = None,
    personas: list[PersonaType] | None = None,
    seed: int | None = None,
) -> pd.DataFrame:
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    if start_date is None:
        start_date = datetime.now() - timedelta(days=num_months * 30)

    if personas is None:
        personas = list(PERSONA_PROFILES.keys())

    all_transactions = []

    for i in range(num_customers):
        cust_id = generate_customer_id(i + 1)
        persona = personas[i % len(personas)]

        customer_txns = generate_customer_transactions(
            cust_id, persona, start_date, num_months
        )
        all_transactions.extend(customer_txns)

        print(f"Generated {len(customer_txns)} transactions for {cust_id} ({persona})")

    df = pd.DataFrame(all_transactions)
    df["tran_date"] = pd.to_datetime(df["tran_date"])
    df = df.sort_values(["cust_id", "tran_date"]).reset_index(drop=True)
    df["tran_date"] = df["tran_date"].dt.strftime("%Y-%m-%d")
    df = df[["cust_id", "dr_cr_indctor", "tran_date", "tran_amt_in_ac", "tran_type", "category_of_txn"]]

    return df


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic banking transaction data for testing"
    )
    parser.add_argument(
        "--customers", "-c",
        type=int,
        default=5,
        help="Number of customers to generate (default: 5)"
    )
    parser.add_argument(
        "--months", "-m",
        type=int,
        default=6,
        help="Number of months of transaction history (default: 6)"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="data/sample_transactions.csv",
        help="Output file path (default: data/sample_transactions.csv)"
    )
    parser.add_argument(
        "--personas", "-p",
        type=str,
        nargs="+",
        choices=list(PERSONA_PROFILES.keys()),
        default=None,
        help="Specific personas to generate (default: all types)"
    )
    parser.add_argument(
        "--seed", "-s",
        type=int,
        default=None,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default=None,
        help="Start date for transactions (YYYY-MM-DD format)"
    )

    args = parser.parse_args()

    start_date = None
    if args.start_date:
        start_date = datetime.strptime(args.start_date, "%Y-%m-%d")

    print(f"Generating transaction data for {args.customers} customers over {args.months} months...")

    df = generate_sample_data(
        num_customers=args.customers,
        num_months=args.months,
        start_date=start_date,
        personas=args.personas,
        seed=args.seed,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"\nGenerated {len(df)} total transactions")
    print(f"Saved to: {output_path}")

    print("\nSummary by customer:")
    summary = df.groupby("cust_id").agg({
        "tran_amt_in_ac": ["count", "sum"],
        "dr_cr_indctor": lambda x: (x == "CR").sum()
    })
    summary.columns = ["Total Txns", "Total Amount", "Credit Txns"]
    print(summary)

    print("\nCategory distribution:")
    print(df["category_of_txn"].value_counts())


if __name__ == "__main__":
    main()
