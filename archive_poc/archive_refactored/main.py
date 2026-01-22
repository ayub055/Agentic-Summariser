"""
Main entry point for the Financial Analysis Agent.
"""

import sys
import os

# Add the archive_refactored directory to Python path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config.settings import MODEL_NAME, VERBOSE_MODE
from config.prompts import FINANCIAL_ANALYST_PROMPT
from data.loader import get_data_summary
from tools import ALL_TOOLS
from agents.financial_agent import FinancialAgent
from utils.helpers import print_header


def demo_basic_usage():
    """Demonstrate basic agent usage."""
    print_header("Basic Agent Usage")
    agent = FinancialAgent(verbose=VERBOSE_MODE)
    question = "How much did CUST0001 spend on Groceries?"
    result = agent.invoke(question)
    print(f"\nResult: {result}")


def demo_multiple_questions():
    """Demonstrate asking multiple questions."""
    print_header("Multiple Questions")
    agent = FinancialAgent(verbose=VERBOSE_MODE)
    questions = [
        "What is CUST0001's total income?",
        "What are the top 3 spending categories for CUST0001?",
        "How much did CUST0001 spend on Rent?",
    ]

    for q in questions:
        print(f"\n{'='*60}")
        result = agent.invoke(q)
        print(f"\nAnswer: {result}")
        print("-" * 60)


def demo_customer_analysis():
    """Demonstrate the specialized customer analysis method."""
    print_header("Customer Analysis")
    agent = FinancialAgent(verbose=VERBOSE_MODE)
    analysis = agent.analyze_customer("CUST0001")
    print(f"\nAnalysis Result:\n{analysis}")


def demo_streaming():
    """Demonstrate streaming output."""
    print_header("Streaming Output")

    agent = FinancialAgent(verbose=False)  # Disable verbose for cleaner streaming
    question = "How much did CUST0001 spend in total?"
    print(f"\nQuestion: {question}")
    print("\nStreaming response: ", end="")
    for token in agent.stream(question): print(token, end="", flush=True)

    print("\n")


def interactive_mode():
    """Run the agent in interactive mode."""
    print_header("Interactive Mode")
    print("Type your questions about customer transactions.")
    print("Type 'quit' or 'exit' to stop.\n")

    agent = FinancialAgent(verbose=True)

    while True:
        try:
            question = input("\nYou: ").strip()
            if question.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            if not question: continue
            result = agent.invoke(question)
            print(f"\nAgent: {result}")

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break


def main():
    """Main entry point."""
    print_header("Financial Analysis Agent", "=", 60)
    print(f"\nModel: {MODEL_NAME}")
    print("Make sure Ollama is running: 'ollama serve'\n")

    # Show data summary
    print("Loading transaction data...")
    print(get_data_summary())

    # Menu
    print("\nSelect a demo:")
    print("1. Basic usage (single question)")
    print("2. Multiple questions")
    print("3. Customer analysis")
    print("4. Streaming output")
    print("5. Interactive mode")
    print("q. Quit")

    choice = input("\nEnter choice (1-5, q): ").strip()

    if choice == '1': demo_basic_usage()
    elif choice == '2': demo_multiple_questions()
    elif choice == '3': demo_customer_analysis()
    elif choice == '4': demo_streaming()
    elif choice == '5': interactive_mode()
    elif choice.lower() == 'q': print("Goodbye!")
    else:
        print("Invalid choice. Running basic demo...")
        demo_basic_usage()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nError: {e}")
        print("\nMake sure:")
        print("1. Ollama is running ('ollama serve')")
        print(f"2. Model '{MODEL_NAME}' is installed ('ollama pull {MODEL_NAME}')")
        print("3. Transaction data exists in ../data/sample_transactions.csv")
