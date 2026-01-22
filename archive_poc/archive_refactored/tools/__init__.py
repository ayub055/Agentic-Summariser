"""
Tools module - All agent tools organized by functionality.
"""

from .spending import (
    get_total_spending,
    get_spending_by_category,
    top_spending_categories,
    spending_in_date_range,
)

from .income import get_total_income

from .lookup import list_customers, list_categories

from .schemas import TopSpendingInput, DateRangeInput

# All tools in a single list for easy binding
ALL_TOOLS = [
    get_total_spending,
    get_total_income,
    get_spending_by_category,
    top_spending_categories,
    spending_in_date_range,
    list_customers,
    list_categories,
]
