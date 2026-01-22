"""
Financial Agent - Specialized agent for transaction analysis.
"""

from typing import Optional, List
from langchain_core.tools import BaseTool

from .base import BaseAgent
from config.prompts import FINANCIAL_ANALYST_PROMPT
from config.settings import MODEL_NAME, TEMPERATURE, MAX_ITERATIONS


class FinancialAgent(BaseAgent):
    """
    Agent specialized for financial transaction analysis.

    Inherits from BaseAgent and configures it with:
    - Financial analyst prompt
    - Transaction analysis tools
    """

    def __init__(
        self,
        tools: Optional[List[BaseTool]] = None,
        model_name: str = MODEL_NAME,
        verbose: bool = True,
    ):
        # Import tools here to avoid circular imports
        if tools is None:
            from tools import ALL_TOOLS
            tools = ALL_TOOLS

        super().__init__(
            model_name=model_name,
            temperature=TEMPERATURE,
            tools=tools,
            system_prompt=FINANCIAL_ANALYST_PROMPT,
            max_iterations=MAX_ITERATIONS,
            verbose=verbose,
        )

    def analyze_customer(self, customer_id: str) -> str:
        """
        Run a comprehensive analysis on a customer.
        Args:ustomer_id: The customer ID to analyze
        Returns: Analysis summary
        """
        question = f"""Provide a comprehensive financial analysis for {customer_id}:
1. What is their total income?
2. What is their total spending?
3. What are their top 3 spending categories?
"""
        return self.invoke(question)

    def compare_spending(self, customer_id: str, category1: str, category2: str) -> str:
        """
        Compare spending between two categories.
        Args:
            customer_id: Customer to analyze
            category1: First category
            category2: Second category

        Returns:
            Comparison result
        """
        question = f"Compare {customer_id}'s spending on {category1} vs {category2}. Which is higher and by how much?"
        return self.invoke(question)
