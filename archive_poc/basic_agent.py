"""
CONCEPT: What is an Agent?
--------------------------
An agent combines a language model (LLM) with tools to create a system that can:
1. REASON about tasks (understand what needs to be done)
2. DECIDE which tools to use
3. ACT by executing those tools
4. OBSERVE results and iterate until done

This follows the ReAct pattern: Reasoning + Acting

transaction Data:
----------------------
We have bank transaction data with columns:
- cust_id: Customer identifier (CUST0001, etc.)
- dr_cr_indctor: DR (Debit/Expense) or CR (Credit/Income)
- tran_date: Transaction date
- tran_amt_in_ac: Amount
- tran_type: IMPS, NEFT, UPI, CARD, CASH, CHEQUE, AUTO_DEBIT
- category_of_txn: Salary, Rent, EMI, Groceries, Entertainment, etc.

Run this file: python basic_agent.py
"""

import pandas as pd
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage

DATA_PATH = "../data/sample_transactions.csv"

def load_transaction_data():
    """Load the transaction CSV file"""
    df = pd.read_csv(DATA_PATH)
    return df

def get_data_summary(df: pd.DataFrame) -> str:
    """Create a summary of the transaction data"""
    summary = f"""
    Transaction Data Summary:
    -------------------------
    Total Records: {len(df)}
    Customers: {df['cust_id'].nunique()}
    Date Range: {df['tran_date'].min()} to {df['tran_date'].max()}

    Transaction Types: {df['tran_type'].unique().tolist()}
    Categories: {df['category_of_txn'].unique().tolist()}

    Total Credits (Income): ${df[df['dr_cr_indctor'] == 'CR']['tran_amt_in_ac'].sum():,.2f}
    Total Debits (Expenses): ${df[df['dr_cr_indctor'] == 'DR']['tran_amt_in_ac'].sum():,.2f}
    """
    return summary


## LLM Provider setup
def create_ollama_llm(model_name: str = "llama3.2"):
    """
    Create an Ollama LLM instance
    Other good models to try:
    - mistral (7B, good for reasoning)
    - codellama (for code tasks)
    - llama3.2 (8B, balanced)
    """
    llm = ChatOllama(model=model_name, temperature=0)
    return llm


#  Basic interaction - No agent yet, just LLM
def basic_llm_query(llm, data_summary: str, question: str) -> str:
    """
    Basic LLM interaction WITHOUT agent capabilities.

    This shows the limitation: the LLM can only use information
    we explicitly provide in the prompt.
    """
    messages = [
        SystemMessage(content="""You are a financial analyst assistant.
        Analyze the provided transaction data and answer questions clearly.
        If you cannot determine something from the data, say so."""),
        HumanMessage(content=f"""
        Here is the transaction data summary:
        {data_summary}

        Question: {question}
        """)
    ]

    response = llm.invoke(messages)
    return response.content


if __name__ == "__main__":
    print("=" * 60)
    print("  Basic LLM Interaction (Pre-Agent)")
    print("=" * 60)

    # Step 1: Load data
    print("\n Loading transaction data...")
    df = load_transaction_data()
    print(f"Loaded {len(df)} transactions")

    # Step 2: Create data summary
    print("\n Creating data summary...")
    summary = get_data_summary(df)
    print(summary)

    # Step 3: Initialize Ollama
    print("\n Initializing Ollama LLM...")
    print("(Make sure Ollama is running: 'ollama serve')")
    llm = create_ollama_llm()

    # Step 4: Ask questions
    print("\n Testing basic LLM queries...")
    print("-" * 40)

    questions = [
        "What are the main expense categories in this data?",
        "How much did customers spend on Rent?",
    ]

    for q in questions:
        print(f"\nQuestion: {q}")
        print("-" * 40)
        try:
            answer = basic_llm_query(llm, summary, q)
            print(f"Answer: {answer}")
        except Exception as e:
            print(f"Error: {e}")
            print("Make sure Ollama is running and llama3.2 is installed")


