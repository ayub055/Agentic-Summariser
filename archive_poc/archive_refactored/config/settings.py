"""
Configuration settings for the agent system.
Centralized place for all configurable parameters.
"""

# =============================================================================
# MODEL CONFIGURATION
# =============================================================================

MODEL_NAME = "llama3.2"  # Ollama model to use
TEMPERATURE = 0          # 0 = deterministic, higher = more creative
MAX_ITERATIONS = 10      # Safety limit for agent loop

# Alternative models to try:
# MODEL_NAME = "mistral"      # Good for reasoning
# MODEL_NAME = "llama3.1:8b"  # Larger, more capable
# MODEL_NAME = "codellama"    # Better for code tasks

# =============================================================================
# DATA PATHS
# =============================================================================

DATA_DIR = "../../data"
TRANSACTIONS_FILE = f"{DATA_DIR}/sample_transactions.csv"

# =============================================================================
# AGENT SETTINGS
# =============================================================================

VERBOSE_MODE = True      # Print detailed agent steps
STREAMING_ENABLED = True # Stream LLM output token by token
