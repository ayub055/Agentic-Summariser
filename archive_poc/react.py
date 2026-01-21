"""
CONCEPT: ReAct = Reasoning + Acting
-----------------------------------
The ReAct pattern is how agents work:

1. THINK   - LLM reasons about what to do
2. ACT     - LLM calls a tool
3. OBSERVE - LLM sees the tool result
4. REPEAT  - Until the task is complete

Example flow for "How much did CUST0001 spend on groceries?":

    User: How much did CUST0001 spend on groceries?

    THINK:  I need to find spending by category. I'll use get_spending_by_category.
    ACT:    get_spending_by_category(customer_id="CUST0001", category="Groceries")
    OBSERVE: Customer CUST0001 spent $18,766.35 on Groceries across 25 transactions

    THINK:  I have the answer now.
    RESPOND: CUST0001 spent $18,766.35 on groceries across 25 transactions.

Run this file: python react.py
"""

import pandas as pd
from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from pydantic import BaseModel, Field
import json


DATA_PATH = "../data/sample_transactions.csv"
TRANSACTIONS_DF = pd.read_csv(DATA_PATH)


@tool
def get_total_spending(customer_id):
    """Get the total spending (debit transactions) for a specific customer.
    Use this when asked about total spending or expenses."""
    df = TRANSACTIONS_DF[
        (TRANSACTIONS_DF['cust_id'] == customer_id) &
        (TRANSACTIONS_DF['dr_cr_indctor'] == 'DR')
    ]
    total = df['tran_amt_in_ac'].sum()
    return f"Customer {customer_id} total spending: ${total:,.2f}"


@tool
def get_total_credit(customer_id):
    """Get the total income (credit transactions) for a specific customer.
    Use this when asked about income or money received."""
    df = TRANSACTIONS_DF[
        (TRANSACTIONS_DF['cust_id'] == customer_id) &
        (TRANSACTIONS_DF['dr_cr_indctor'] == 'CR')
    ]
    total = df['tran_amt_in_ac'].sum()
    return f"Customer {customer_id} total income: ${total:,.2f}"


@tool
def get_spending_by_category(customer_id, category):
    """Get spending for a specific category for a customer.
    Categories include: Groceries, Rent, EMI, Entertainment, Shopping,
    Utilities, Healthcare, Travel, Insurance, Transfers_Out, Cash_Withdrawal.
    Use this when asked about spending in a specific category."""
    df = TRANSACTIONS_DF[
        (TRANSACTIONS_DF['cust_id'] == customer_id) &
        (TRANSACTIONS_DF['dr_cr_indctor'] == 'DR') &
        (TRANSACTIONS_DF['category_of_txn'] == category)
    ]
    total = df['tran_amt_in_ac'].sum()
    count = len(df)
    return f"Customer {customer_id} spent ${total:,.2f} on {category} across {count} transactions"


@tool
def list_customers():
    """List all customer IDs available in the data.
    Use this when you need to know which customers exist."""
    customers = TRANSACTIONS_DF['cust_id'].unique().tolist()
    return f"Available customers: {', '.join(customers[:10])}{'...' if len(customers) > 10 else ''}"


@tool
def list_categories():
    """List all spending categories available in the data.
    Use this when you need to know what categories exist."""
    categories = TRANSACTIONS_DF['category_of_txn'].unique().tolist()
    return f"Available categories: {', '.join(categories)}"


class TopSpendingInput(BaseModel):
    customer_id: str = Field(description="The customer ID")
    top_n: int = Field(default=5, description="Number of top categories to return")


@tool(args_schema=TopSpendingInput)
def top_spending_categories(customer_id, top_n: int = 5) -> str:
    """Get the top N spending categories for a customer ranked by amount.
    Use this when asked about biggest expenses or where money goes."""
    df = TRANSACTIONS_DF[
        (TRANSACTIONS_DF['cust_id'] == customer_id) &
        (TRANSACTIONS_DF['dr_cr_indctor'] == 'DR')
    ]
    category_totals = df.groupby('category_of_txn')['tran_amt_in_ac'].sum()
    top_cats = category_totals.sort_values(ascending=False).head(top_n)

    result = f"Top {top_n} spending categories for {customer_id}:\n"
    for i, (cat, amt) in enumerate(top_cats.items(), 1):
        result += f"  {i}. {cat}: ${amt:,.2f}\n"
    return result


# Collect all tools
TOOLS = [get_total_spending,get_total_credit,get_spending_by_category,
        list_customers,list_categories,top_spending_categories,]



# Bind Tools to the LLM
def create_agent_llm(model_name: str = "llama3.2"):
    """
    Create an LLM with tools bound to it.
    bind_tools() tells the LLM what tools are available.
    The LLM can then decide to call these tools in its responses.
    """
    llm = ChatOllama(model=model_name,temperature=0,)
    llm_with_tools = llm.bind_tools(TOOLS) #Bind tools to the LLM

    return llm_with_tools


# The Agent Loop (ReAct Pattern)
def run_agent(llm_with_tools, user_question: str, verbose: bool = True):
    """
    Run the ReAct agent loop.
    This is the core agent pattern:
    1. Send message to LLM
    2. Check if LLM wants to call tools
    3. If yes: execute tools, add results, go back to step 1
    4. If no: return the final response
    """
    # tool lookup dictionary for easy access
    tool_map = {t.name: t for t in TOOLS}

    # user's question - starting point
    # STRONGER PROMPT for smaller models like llama3.2
    messages = [
        HumanMessage(content=f"""You are a financial analyst. Use the available tools to answer questions.

CRITICAL RULES:
1. Call the appropriate tool to get data
2. After receiving tool results, REPEAT THE EXACT NUMBERS in your answer
3. Your final answer MUST contain the specific dollar amounts and counts from the tool

Question: {user_question}""")
    ]

    if verbose:
        print(f"\n{'='*60}")
        print(f"USER QUESTION: {user_question}")
        print('='*60)

    # The Agent Loop
    iteration = 0
    max_iterations = 10 

    while iteration < max_iterations:
        iteration += 1
        if verbose: print(f"\n--- Iteration {iteration} ---")

        # Step 1: Call the LLM
        response = llm_with_tools.invoke(messages)
        if verbose: print(f"LLM Response Type: {'Tool Call' if response.tool_calls else 'Final Answer'}")

        # Step 2: Check if LLM wants to call tools
        if response.tool_calls: # LLM wants to use tools
            messages.append(response)  # Add AImessage with tool calls

            # Step 3: Execute each tool call
            for tool_call in response.tool_calls:
                tool_name = tool_call['name']
                tool_args = tool_call['args']
                tool_id = tool_call['id']

                if verbose:
                    print(f"\n  TOOL CALL: {tool_name}")
                    print(f"  ARGUMENTS: {tool_args}")

                # Execute the tool
                if tool_name in tool_map: tool_result = tool_map[tool_name].invoke(tool_args)
                else: tool_result = f"Error: Unknown tool {tool_name}"

                if verbose: print(f"  RESULT: {tool_result}")

                # Add tool result to messages
                messages.append(ToolMessage(content=str(tool_result), tool_call_id=tool_id))

        else: # No tool calls - LLM is giving final answer
            if verbose: print(f"\nFINAL ANSWER: {response.content}")
            return response.content

    return "Max iterations reached without final answer"


#  Demonstrate the Agent
def demonstrate_react_agent():
    """Run several example queries to show the agent in action"""

    print("LESSON 3: ReAct Agent in Action")
    print("\nInitializing Ollama LLM with tools...")
    print("(Make sure Ollama is running: 'ollama serve')")

    try:
        llm = create_agent_llm()

        questions = [
            "How much did CUST0001 spend in total?",
            "What are the top 3 spending categories for CUST0001?",
            "How much did CUST0001 spend on Rent?",
        ]

        for q in questions:
            try: result = run_agent(llm, q, verbose=True)
            except Exception as e:
                print(f"\nError: {e}")
                print("Make sure Ollama is running and llama3.2 supports tool calling")
            print("\n" + "-" * 60)

    except Exception as e: print(f"Failed to initialize: {e}")



# Step-by-Step Breakdown (Educational)
def explain_react_flow():
    """Print an explanation of what happens in the ReAct loop"""

    print("""
    ╔════════════════════════════════════════════════════════════╗
    ║              THE ReAct AGENT LOOP EXPLAINED                ║
    ╠════════════════════════════════════════════════════════════╣
    ║                                                            ║
    ║  User: "How much did CUST0001 spend on groceries?"         ║
    ║                           │                                ║
    ║                           ▼                                ║
    ║  ┌─────────────────────────────────────────────────────┐   ║
    ║  │ LLM THINKS:                                         │   ║
    ║  │ "I need spending by category. I'll use              │   ║
    ║  │  get_spending_by_category tool."                    │   ║
    ║  └─────────────────────────────────────────────────────┘   ║
    ║                           │                                ║
    ║                           ▼                                ║
    ║  ┌─────────────────────────────────────────────────────┐   ║
    ║  │ LLM ACTS:                                           │   ║
    ║  │ Calls: get_spending_by_category(                    │   ║
    ║  │          customer_id="CUST0001",                    │   ║
    ║  │          category="Groceries"                       │   ║
    ║  │        )                                            │   ║
    ║  └─────────────────────────────────────────────────────┘   ║
    ║                           │                                ║
    ║                           ▼                                ║
    ║  ┌─────────────────────────────────────────────────────┐   ║
    ║  │ TOOL EXECUTES:                                      │   ║
    ║  │ Queries DataFrame, calculates sum                   │   ║
    ║  │ Returns: "$18,766.35 on Groceries, 25 transactions" │   ║
    ║  └─────────────────────────────────────────────────────┘   ║
    ║                           │                                ║
    ║                           ▼                                ║
    ║  ┌─────────────────────────────────────────────────────┐   ║
    ║  │ LLM OBSERVES:                                       │   ║
    ║  │ Sees the tool result in the conversation            │   ║
    ║  └─────────────────────────────────────────────────────┘   ║
    ║                           │                                ║
    ║                           ▼                                ║
    ║  ┌─────────────────────────────────────────────────────┐   ║
    ║  │ LLM RESPONDS:                                       │   ║
    ║  │ "CUST0001 spent $18,766.35 on groceries across      │   ║
    ║  │  25 transactions."                                  │   ║
    ║  └─────────────────────────────────────────────────────┘   ║
    ║                                                            ║
    ╚════════════════════════════════════════════════════════════╝

    KEY CODE COMPONENTS:

    1. llm.bind_tools(TOOLS)
       → Tells LLM what tools exist and their schemas

    2. response.tool_calls
       → List of tools the LLM wants to call
       → Each has: name, args, id

    3. ToolMessage(content=result, tool_call_id=id)
       → How we send tool results back to the LLM

    4. The while loop
       → Keeps running until LLM gives final answer (no tool_calls)
    """)



if __name__ == "__main__":

    explain_react_flow()

    print("\nPress Enter to run the agent demo...")
    input()

    demonstrate_react_agent()


    print("KEY TAKEAWAY:")
    print("=" * 60)
    print("""
    The ReAct loop is the heart of LangChain agents:

    1. bind_tools() - Give LLM access to tools
    2. LLM decides if/which tool to call
    3. We execute the tool and return results
    4. Loop until LLM has enough info to answer

    This is a MANUAL agent loop. In Lesson 4, we'll use
    LangChain's create_react_agent() which handles this
    automatically with LangGraph!
    """)
