"""
Spending-related tools for analyzing debit transactions.
"""

from langchain_core.tools import tool
from .schemas import TopSpendingInput, DateRangeInput
from data.loader import get_transactions_df


@tool
def get_total_spending(customer_id: str) -> str:
    """
    Get the total spending (debit transactions) for a specific customer.

    Use this tool when asked about how much a customer spent in total.

    Args:
        customer_id: The customer ID (e.g., 'CUST0001')
    """
    df = get_transactions_df()
    filtered = df[
        (df['cust_id'] == customer_id) &
        (df['dr_cr_indctor'] == 'DR')
    ]
    total = filtered['tran_amt_in_ac'].sum()
    return f"Customer {customer_id} total spending: ${total:,.2f}"


@tool
def get_spending_by_category(customer_id: str, category: str) -> str:
    """
    Get spending for a specific category for a customer.

    Categories include: Groceries, Rent, EMI, Entertainment, Shopping,
    Utilities, Healthcare, Travel, Insurance, Transfers_Out, Cash_Withdrawal.

    Use this when asked about spending in a specific category.

    Args:
        customer_id: The customer ID (e.g., 'CUST0001')
        category: The spending category (e.g., 'Groceries', 'Rent')
    """
    df = get_transactions_df()
    filtered = df[
        (df['cust_id'] == customer_id) &
        (df['dr_cr_indctor'] == 'DR') &
        (df['category_of_txn'] == category)
    ]
    total = filtered['tran_amt_in_ac'].sum()
    count = len(filtered)
    return f"Customer {customer_id} spent ${total:,.2f} on {category} across {count} transactions"


@tool(args_schema=TopSpendingInput)
def top_spending_categories(customer_id: str, top_n: int = 5) -> str:
    """
    Get the top N spending categories for a customer, ranked by total amount.

    Use this when asked about where a customer spends the most money
    or their biggest expense categories.

    Args:
        customer_id: The customer ID
        top_n: How many top categories to return (default 5)
    """
    df = get_transactions_df()
    filtered = df[
        (df['cust_id'] == customer_id) &
        (df['dr_cr_indctor'] == 'DR')
    ]

    category_totals = filtered.groupby('category_of_txn')['tran_amt_in_ac'].sum()
    top_cats = category_totals.sort_values(ascending=False).head(top_n)

    result = f"Top {top_n} spending categories for {customer_id}:\n"
    for i, (cat, amt) in enumerate(top_cats.items(), 1):
        result += f"  {i}. {cat}: ${amt:,.2f}\n"
    return result


@tool(args_schema=DateRangeInput)
def spending_in_date_range(customer_id: str, start_date: str, end_date: str) -> str:
    """
    Get total spending for a customer within a specific date range.

    Use this when asked about spending during a particular time period.

    Args:
        customer_id: The customer ID
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
    """
    df = get_transactions_df()
    filtered = df[
        (df['cust_id'] == customer_id) &
        (df['dr_cr_indctor'] == 'DR') &
        (df['tran_date'] >= start_date) &
        (df['tran_date'] <= end_date)
    ]

    total = filtered['tran_amt_in_ac'].sum()
    count = len(filtered)
    return f"Customer {customer_id} spent ${total:,.2f} between {start_date} and {end_date} ({count} transactions)"
