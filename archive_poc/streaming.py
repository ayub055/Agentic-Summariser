"""
CONCEPT: Streaming
-----------------------
Without streaming: User waits... waits... then sees complete response
With streaming: User sees tokens appear as they're generated (like ChatGPT)

Benefits:
- Better user experience (immediate feedback)
- Can show "thinking" in real-time
- Can cancel early if response is wrong

Key Methods:
- .invoke()  → Wait for complete response (blocking)
- .stream()  → Get tokens one by one (generator)
- .astream() → Async streaming

Run this file: python streaming.py
"""

import pandas as pd
from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, ToolMessage, AIMessageChunk
from pydantic import BaseModel, Field
import sys


DATA_PATH = "../data/sample_transactions.csv"
TRANSACTIONS_DF = pd.read_csv(DATA_PATH)


@tool
def get_total_spending(customer_id: str) -> str:
    """Get total spending for a customer."""
    df = TRANSACTIONS_DF[
        (TRANSACTIONS_DF['cust_id'] == customer_id) &
        (TRANSACTIONS_DF['dr_cr_indctor'] == 'DR')
    ]
    total = df['tran_amt_in_ac'].sum()
    return f"Customer {customer_id} total spending: ${total:,.2f}"


@tool
def get_spending_by_category(customer_id: str, category: str) -> str:
    """Get spending for a specific category."""
    df = TRANSACTIONS_DF[
        (TRANSACTIONS_DF['cust_id'] == customer_id) &
        (TRANSACTIONS_DF['dr_cr_indctor'] == 'DR') &
        (TRANSACTIONS_DF['category_of_txn'] == category)
    ]
    total = df['tran_amt_in_ac'].sum()
    count = len(df)
    return f"Customer {customer_id} spent ${total:,.2f} on {category} ({count} transactions)"


class TopSpendingInput(BaseModel):
    customer_id: str = Field(description="The customer ID")
    top_n: int = Field(default=5, description="Number of categories")


@tool(args_schema=TopSpendingInput)
def top_spending_categories(customer_id: str, top_n: int = 5) -> str:
    """Get top N spending categories for a customer."""
    df = TRANSACTIONS_DF[
        (TRANSACTIONS_DF['cust_id'] == customer_id) &
        (TRANSACTIONS_DF['dr_cr_indctor'] == 'DR')
    ]
    category_totals = df.groupby('category_of_txn')['tran_amt_in_ac'].sum()
    top_cats = category_totals.sort_values(ascending=False).head(top_n)

    result = f"Top {top_n} categories for {customer_id}:\n"
    for i, (cat, amt) in enumerate(top_cats.items(), 1):
        result += f"  {i}. {cat}: ${amt:,.2f}\n"
    return result


TOOLS = [get_total_spending, get_spending_by_category, top_spending_categories]


# Basic Streaming (No Tools)
def demo_basic_streaming():
    """
    Stream LLM output token by token.
    Each chunk contains a small piece of the response.
    """

    print("... Basic Streaming (No Tools) \n")


    llm = ChatOllama(model="llama3.2", temperature=0)
    question = "Explain what a bank transaction is in 2 sentences."
    print(f"\nQuestion: {question}")
    print("\nStreaming response:")
    print("-" * 40)

    # .stream() returns a generator of chunks
    for chunk in llm.stream([HumanMessage(content=question)]):
        # Each chunk has .content with partial text
        print(chunk.content, end="", flush=True)
    print("\n" + "-" * 40)


# Streaming with Tools (Agent)
def demo_streaming_with_tools():
    """
    Streaming with tool-enabled LLM is more complex because
    chunks can contain:
    - Text content (stream it)
    - Tool calls (collect and execute)
    """
    print(" ...Streaming with Tools \n")

    llm = ChatOllama(model="llama3.2", temperature=0)
    llm_with_tools = llm.bind_tools(TOOLS)
    tool_map = {t.name: t for t in TOOLS}

    question = "How much did CUST0001 spend on Groceries?"
    print(f"\nQuestion: {question}")

    messages = [
        HumanMessage(content=f"""You are a financial analyst.
Use tools to get data. Include exact numbers in your answer.
Question: {question}""")
    ]

    iteration = 0
    max_iterations = 5

    while iteration < max_iterations:
        iteration += 1
        print(f"\n--- Iteration {iteration} ---")

        # Collect the full response from streaming
        collected_content = ""
        collected_tool_calls = []
        print("Streaming: ", end="")

        # Stream the response
        for chunk in llm_with_tools.stream(messages):
            # Handle text content - print as it arrives
            if chunk.content:
                print(chunk.content, end="", flush=True)
                collected_content += chunk.content

            # Handle tool calls - collect them
            if chunk.tool_calls: collected_tool_calls.extend(chunk.tool_calls)

        print()  

        # If there are tool calls, execute them
        if collected_tool_calls:
            print(f"\n  Tool calls detected: {len(collected_tool_calls)}")

            from langchain_core.messages import AIMessage
            ai_msg = AIMessage(content=collected_content, tool_calls=collected_tool_calls)
            messages.append(ai_msg)

            # Execute each tool
            for tool_call in collected_tool_calls:
                tool_name = tool_call['name']
                tool_args = tool_call['args']
                tool_id = tool_call['id']

                print(f"  Executing: {tool_name}({tool_args})")

                if tool_name in tool_map: result = tool_map[tool_name].invoke(tool_args)
                else:  result = f"Unknown tool: {tool_name}"

                print(f"  Result: {result}")
                messages.append(ToolMessage(content=str(result), tool_call_id=tool_id))
        else:
            print(f"\nFinal Answer: {collected_content}")
            return collected_content

    return "Max iterations reached"


def demo_invoke_vs_stream():
    """
    Side-by-side comparison of invoke vs stream
    """
    print("\n" + "=" * 60)
    print("PART 3: invoke() vs stream() Comparison")
    print("=" * 60)

    llm = ChatOllama(model="llama3.2", temperature=0)
    question = "Count from 1 to 5."

    # Method 1: invoke() - blocks until complete
    print("\n[invoke()] - Waits for complete response:")
    print("-" * 40)
    response = llm.invoke([HumanMessage(content=question)])
    print(response.content)

    # Method 2: stream() - yields chunks
    print("\n[stream()] - Shows tokens as they arrive:")
    print("-" * 40)
    for chunk in llm.stream([HumanMessage(content=question)]):
        # Simulate showing tokens arriving
        print(f"[{chunk.content}]", end="", flush=True)
    print()


if __name__ == "__main__":

    print("-----Streaming LLM Output------")
    print("\nMake sure Ollama is running: 'ollama serve'")

    try:
        demo_basic_streaming()

        input("\nPress Enter for (Streaming with Tools)...")
        demo_streaming_with_tools()

        input("\nPress Enter for (invoke vs stream)...")
        demo_invoke_vs_stream()

    except Exception as e:
        print(f"\nError: {e}")
        print("Make sure Ollama is running with llama3.2 model")

    print("\n" + "=" * 60)
    print("KEY TAKEAWAYS:")
    print("=" * 60)
    print("""
    1. .stream() returns a generator - iterate to get chunks

    2. Each chunk has:
       - .content  → Text (may be partial)
       - .tool_calls → Tools the LLM wants to call

    3. For tools + streaming:
       - Collect all chunks first
       - Then check for tool_calls
       - Execute tools and continue the loop

    4. Use .astream() for async applications (web servers)

    5. Streaming improves UX but adds complexity to tool handling
    """)
