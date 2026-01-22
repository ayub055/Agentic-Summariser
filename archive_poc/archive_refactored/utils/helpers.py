"""
Utility functions used across the project.
"""


def format_currency(amount: float) -> str:
    """Format a number as currency."""
    return f"${amount:,.2f}"


def print_header(title: str, char: str = "=", width: int = 60):
    """Print a formatted header."""
    print(char * width)
    print(title.center(width))
    print(char * width)


def print_section(title: str, char: str = "-", width: int = 40):
    """Print a section divider."""
    print(f"\n{title}")
    print(char * width)
