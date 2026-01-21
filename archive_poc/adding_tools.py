"""
CONCEPT: What are Tools?
------------------------
Tools are functions that agents can use to interact with the world.
They enable agents to:
- Query databases
- Perform calculations
- Access external APIs
- Read/write files
- Execute code

The @tool decorator converts Python functions into tools that agents can call.

Key Tool Properties:
-------------------
1. name: How the agent refers to the tool
2. description: Tells the agent WHEN to use the tool (very important!)
3. args_schema: Defines what arguments the tool accepts
4. return_direct: If True, returns result directly without agent processing

Run this file: python adding_tools.py
"""

import pandas as pd
from typing import Optional, List
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field

# Load our transaction data globally for tools to access
DATA_PATH = "../data/sample_transactions.csv"
TRANSACTIONS_DF = pd.read_csv(DATA_PATH)

# ----- TOOLS ---
# Creating Simple Tools with @tool decorator
@tool
def get_total_spending(customer_id):
    """
    Get the total spending (debit transactions) for a specific customer.
    Use this tool when asked about how much a customer spent in total.
    """
    df = TRANSACTIONS_DF[
        (TRANSACTIONS_DF['cust_id'] == customer_id) &
        (TRANSACTIONS_DF['dr_cr_indctor'] == 'DR')
    ]
    total = df['tran_amt_in_ac'].sum()
    return f"Customer {customer_id} total spending: ${total:,.2f}"


@tool
def get_total_credit(customer_id):
    """
    Get the total income (credit transactions) for a specific customer.
    Use this tool when asked about how much money a customer received.
    """
    df = TRANSACTIONS_DF[
        (TRANSACTIONS_DF['cust_id'] == customer_id) &
        (TRANSACTIONS_DF['dr_cr_indctor'] == 'CR')
    ]
    total = df['tran_amt_in_ac'].sum()
    return f"Customer {customer_id} total income: ${total:,.2f}"


@tool
def get_spending_by_category(customer_id: str, category: str) -> str:
    """
    Get spending for a specific category for a customer.
    """
    df = TRANSACTIONS_DF[
        (TRANSACTIONS_DF['cust_id'] == customer_id) &
        (TRANSACTIONS_DF['dr_cr_indctor'] == 'DR') &
        (TRANSACTIONS_DF['category_of_txn'] == category)
    ]
    total = df['tran_amt_in_ac'].sum()
    count = len(df)
    return f"Customer {customer_id} spent ${total:,.2f} on {category} across {count} transactions"


@tool
def list_categories() -> str:
    """
    List all available spending categories in the transaction data.
    """
    categories = TRANSACTIONS_DF['category_of_txn'].unique().tolist()
    return f"Available categories: {', '.join(categories)}"


@tool
def list_customers() -> str:
    """
    List all customer IDs in the transaction data.
    """
    customers = TRANSACTIONS_DF['cust_id'].unique().tolist()
    return f"Available customers: {', '.join(customers)}"



# Creating Tools with Structured Arguments (Pydantic)
class TopSpendingInput(BaseModel):
    """Input schema for top_spending_categories tool"""
    customer_id: str = Field(description="The customer ID")
    top_n: int = Field(default=5, description="Number of top categories to return")


@tool(args_schema=TopSpendingInput)
def top_spending_categories(customer_id: str, top_n: int = 5) -> str:
    """
    Get the top N spending categories for a customer, ranked by total amount.
    Use this when asked about where a customer spends the most money
    or their biggest expense categories.
    """
    df = TRANSACTIONS_DF[
        (TRANSACTIONS_DF['cust_id'] == customer_id) &
        (TRANSACTIONS_DF['dr_cr_indctor'] == 'DR')
    ]

    category_totals = df.groupby('category_of_txn')['tran_amt_in_ac'].sum()
    top_categories = category_totals.sort_values(ascending=False).head(top_n)

    result = f"Top {top_n} spending categories for {customer_id}:\n"
    for i, (cat, amount) in enumerate(top_categories.items(), 1):
        result += f"  {i}. {cat}: ${amount:,.2f}\n"
    return result


class DateRangeInput(BaseModel):
    """Input schema for spending in date range"""
    customer_id: str = Field(description="The customer ID")
    start_date: str = Field(description="Start date in YYYY-MM-DD format")
    end_date: str = Field(description="End date in YYYY-MM-DD format")


@tool(args_schema=DateRangeInput)
def spending_in_date_range(customer_id: str, start_date: str, end_date: str) -> str:
    """
    Get total spending for a customer within a specific date range.
    Use this when asked about spending during a particular time period.
    """
    df = TRANSACTIONS_DF[
        (TRANSACTIONS_DF['cust_id'] == customer_id) &
        (TRANSACTIONS_DF['dr_cr_indctor'] == 'DR') &
        (TRANSACTIONS_DF['tran_date'] >= start_date) &
        (TRANSACTIONS_DF['tran_date'] <= end_date)
    ]

    total = df['tran_amt_in_ac'].sum()
    count = len(df)
    return f"Customer {customer_id} spent ${total:,.2f} between {start_date} and {end_date} ({count} transactions)"



#  Testing Tools Directly
def demonstrate_tools():
    """Show how tools work by calling them directly"""

    print("\n[Tool 1] list_customers()")
    print(list_customers.invoke({}))

    print("\n[Tool 2] list_categories()")
    print(list_categories.invoke({}))

    print("\n[Tool 3] get_total_spending('CUST0001')")
    print(get_total_spending.invoke({"customer_id": "CUST0001"}))

    print("\n[Tool 4] get_total_income('CUST0001')")
    print(get_total_credit.invoke({"customer_id": "CUST0001"}))

    print("\n[Tool 5] get_spending_by_category('CUST0001', 'Groceries')")
    print(get_spending_by_category.invoke({"customer_id": "CUST0001", "category": "Groceries"}))

    print("\n[Tool 6] top_spending_categories('CUST0001', 3)")
    print(top_spending_categories.invoke({"customer_id": "CUST0001", "top_n": 3}))

    print("\n[Tool 7] spending_in_date_range('CUST0001', '2025-07-01', '2025-07-31')")
    print(spending_in_date_range.invoke({
        "customer_id": "CUST0001",
        "start_date": "2025-07-01",
        "end_date": "2025-07-31"
    }))


# Showing Tool Metadata (What the Agent Sees)
def show_tool_metadata():
    """Display what the agent sees about each tool"""

    tools = [
        get_total_spending,
        get_total_credit,
        get_spending_by_category,
        list_categories,
        list_customers,
        top_spending_categories,
        spending_in_date_range
    ]

    for t in tools:
        print(f"\nTool: {t.name}")
        print(f"Description: {t.description[:100]}...")
        print(f"Args: {t.args}")



if __name__ == "__main__":

    demonstrate_tools()
    show_tool_metadata()