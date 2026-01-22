"""
Base agent implementation with the ReAct loop.
This is the core agent pattern that can be reused.
"""

from typing import List, Optional, Generator
from langchain_ollama import ChatOllama
from langchain_core.tools import BaseTool
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, BaseMessage


class BaseAgent:
    """
    Base agent class implementing the ReAct pattern.

    The ReAct loop:
    1. THINK  - LLM reasons about what to do
    2. ACT    - LLM calls a tool
    3. OBSERVE - LLM sees the result
    4. REPEAT - Until task is complete
    """

    def __init__(
        self,
        model_name: str = "llama3.2",
        temperature: float = 0,
        tools: Optional[List[BaseTool]] = None,
        system_prompt: str = "",
        max_iterations: int = 10,
        verbose: bool = True,
    ):
        self.model_name = model_name
        self.temperature = temperature
        self.tools = tools or []
        self.system_prompt = system_prompt
        self.max_iterations = max_iterations
        self.verbose = verbose
        self.llm = ChatOllama(model=model_name,temperature=temperature) # LLM

        # Bind tools
        if self.tools: self.llm_with_tools = self.llm.bind_tools(self.tools)
        else: self.llm_with_tools = self.llm
        self.tool_map = {t.name: t for t in self.tools} # Tool lookup map

    def _log(self, message: str):
        """Print if verbose mode is enabled."""
        if self.verbose: print(message)

    def invoke(self, question: str) -> str:
        """
        Run the agent on a question and return the final answer.
        Args: question: User's question
        Returns: Final answer string
        """
        # Build initial message
        messages = [ HumanMessage(content=f"{self.system_prompt}\n\nQuestion: {question}")]

        self._log(f"\n{'-'*60}")
        self._log(f"USER QUESTION: {question}")
        self._log('-'*60)

        # ReAct loop
        for iteration in range(1, self.max_iterations + 1):
            self._log(f"\n--- Iteration {iteration} ---")

            response = self.llm_with_tools.invoke(messages) # Call LLM
            if response.tool_calls:
                self._log(f"LLM Response: Tool Call")
                messages.append(response)

                # Execute each tool
                for tool_call in response.tool_calls:
                    result = self._execute_tool(tool_call)
                    messages.append(ToolMessage(content=str(result),tool_call_id=tool_call['id']))
            else:
                self._log(f"LLM Response: Final Answer")
                self._log(f"\nFINAL ANSWER: {response.content}")
                return response.content

        return "Max iterations reached without final answer"

    def _execute_tool(self, tool_call: dict) -> str:
        """Execute a single tool call."""
        tool_name = tool_call['name']
        tool_args = tool_call['args']
        tool_id = tool_call['id']

        self._log(f"  TOOL: {tool_name}")
        self._log(f"  ARGS: {tool_args}")

        if tool_name in self.tool_map: result = self.tool_map[tool_name].invoke(tool_args)
        else: result = f"Error: Unknown tool '{tool_name}'"

        self._log(f"  RESULT: {result}")
        return result

    def stream(self, question: str) -> Generator[str, None, None]:
        """
        Stream the agent's response token by token.
        Args: question: User's question
        Yields Tokens as they are generated
        """
        messages = [HumanMessage(content=f"{self.system_prompt}\n\nQuestion: {question}")]

        for iteration in range(1, self.max_iterations + 1):
            collected_content = ""
            collected_tool_calls = []

            # Stream response
            for chunk in self.llm_with_tools.stream(messages):
                if chunk.content:
                    yield chunk.content
                    collected_content += chunk.content
                if chunk.tool_calls:
                    collected_tool_calls.extend(chunk.tool_calls)

            if collected_tool_calls:
                # Handle tool calls
                ai_msg = AIMessage(content=collected_content, tool_calls=collected_tool_calls)
                messages.append(ai_msg)

                for tool_call in collected_tool_calls:
                    result = self._execute_tool(tool_call)
                    messages.append(ToolMessage(content=str(result),tool_call_id=tool_call['id']))
            else:
                return # Done - final answer was streamed


def run_agent_loop(
    llm_with_tools,
    tools: List[BaseTool],
    messages: List[BaseMessage],
    max_iterations: int = 10,
    verbose: bool = True
) -> str:
    """
    Standalone function to run the ReAct loop.
    This is an alternative to using the BaseAgent class.
    """
    tool_map = {t.name: t for t in tools}

    for iteration in range(1, max_iterations + 1):
        if verbose: print(f"\n--- Iteration {iteration} ---")

        response = llm_with_tools.invoke(messages)
        if response.tool_calls:
            if verbose: print("LLM Response: Tool Call")

            messages.append(response)
            for tool_call in response.tool_calls:
                tool_name = tool_call['name']
                tool_args = tool_call['args']

                if verbose: print(f"  TOOL: {tool_name}({tool_args})")
                if tool_name in tool_map: result = tool_map[tool_name].invoke(tool_args)
                else: result = f"Unknown tool: {tool_name}"

                if verbose: print(f"  RESULT: {result}")
                messages.append(ToolMessage(content=str(result), tool_call_id=tool_call['id']))
        else:
            if verbose: print(f"\nFINAL: {response.content}")
            return response.content

    return "Max iterations reached"
