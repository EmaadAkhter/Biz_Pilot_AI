import os
import random
import json
import logging
from typing import Optional, Dict, Any, List, Callable
from dotenv import load_dotenv
from openai import OpenAI, OpenAIError

load_dotenv()
logger = logging.getLogger(__name__)

# Load API keys and models
API_KEYS = [os.getenv(f"O_R_API{i}") for i in range(1, 11)]
API_KEYS = [k for k in API_KEYS if k]

MODELS = [os.getenv(f'MODEL{i}') for i in range(1, 7)]
MODELS = [m for m in MODELS if m]

# Models that support reasoning
REASONING_MODELS = [os.getenv(f'REASONING_MODEL{i}') for i in range(1, 4)]
REASONING_MODELS = [m for m in REASONING_MODELS if m]

if not API_KEYS:
    raise ValueError("No OpenRouter API keys found in environment variables")

if not MODELS:
    raise ValueError("No models found in environment variables")

# ========== FUNCTION DEFINITIONS ==========
AVAILABLE_FUNCTIONS = [
    {
        "type": "function",
        "function": {
            "name": "list_available_files",
            "description": "List all available sales data files for a user",
            "parameters": {
                "type": "object",
                "properties": {
                    "user_id": {
                        "type": "string",
                        "description": "The user's ID"
                    }
                },
                "required": ["user_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "analyze_sales_file",
            "description": "Analyze sales data from an uploaded file. Returns total sales, averages, max/min values, top and bottom products, daily/weekly/monthly trends, and sales by category and region",
            "parameters": {
                "type": "object",
                "properties": {
                    "filename": {
                        "type": "string",
                        "description": "Name or blob_name of the file"
                    },
                    "user_id": {
                        "type": "string",
                        "description": "The user's ID"
                    }
                },
                "required": ["filename", "user_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "query_sales_data",
            "description": "Answer specific questions about sales data by analyzing the dataset",
            "parameters": {
                "type": "object",
                "properties": {
                    "filename": {
                        "type": "string",
                        "description": "Name or blob_name of the file"
                    },
                    "user_id": {
                        "type": "string",
                        "description": "The user's ID"
                    },
                    "question": {
                        "type": "string",
                        "description": "The specific question about the data"
                    }
                },
                "required": ["filename", "user_id", "question"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "forecast_sales_demand",
            "description": "Forecast future sales demand using Facebook Prophet time series model. Returns predictions with confidence intervals and trend analysis",
            "parameters": {
                "type": "object",
                "properties": {
                    "filename": {
                        "type": "string",
                        "description": "Name or blob_name of the file"
                    },
                    "user_id": {
                        "type": "string",
                        "description": "The user's ID"
                    },
                    "periods": {
                        "type": "integer",
                        "description": "Number of days to forecast (1-365)",
                        "default": 30,
                        "minimum": 1,
                        "maximum": 365
                    }
                },
                "required": ["filename", "user_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "market_research",
            "description": "Perform market research on a business idea, customer segment, or market. Returns market size, competitors, trends, and opportunities based on real web data",
            "parameters": {
                "type": "object",
                "properties": {
                    "idea": {
                        "type": "string",
                        "description": "The product/service idea or topic to research"
                    },
                    "customer": {
                        "type": "string",
                        "description": "Target customer segment or market focus"
                    },
                    "geography": {
                        "type": "string",
                        "description": "Geographic market (e.g., 'United States', 'Europe', 'Asia')"
                    },
                    "level": {
                        "type": "integer",
                        "description": "Research depth: 1=quick overview, 2=medium analysis, 3=comprehensive",
                        "enum": [1, 2, 3],
                        "default": 1
                    }
                },
                "required": ["idea", "customer", "geography"]
            }
        }
    }
]


def call_llm_simple(
        prompt: str,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        use_reasoning: bool = False
) -> str:
    """Simple LLM call without function calling

    Args:
        prompt: User prompt
        model: Model to use (random if None)
        temperature: Sampling temperature (0-2)
        max_tokens: Maximum tokens in response
        use_reasoning: Enable reasoning mode for supported models

    Returns:
        LLM response text

    Raises:
        ValueError: If no API keys or models configured
        OpenAIError: If API call fails
    """
    api_key = random.choice(API_KEYS)

    # Select model based on reasoning requirement
    if use_reasoning:
        if not REASONING_MODELS:
            logger.warning("Reasoning requested but no reasoning models available. Using standard model.")
            model = model or random.choice(MODELS)
            use_reasoning = False
        else:
            model = model or random.choice(REASONING_MODELS)
    else:
        model = model or random.choice(MODELS)

    logger.info(f"Calling LLM - Model: {model}, Use Reasoning: {use_reasoning}, API Key: {api_key[:10]}...")

    client = OpenAI(
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1"
    )

    try:
        kwargs = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "timeout": 60.0
        }

        if max_tokens:
            kwargs["max_tokens"] = max_tokens

        if use_reasoning:
            kwargs["reasoning"] = {"enabled": True}

        response = client.chat.completions.create(**kwargs)

        result = response.choices[0].message.content
        logger.info(f"LLM response received - Length: {len(result)} chars")
        return result

    except OpenAIError as e:
        logger.error(f"OpenAI API error: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error in LLM call: {str(e)}")
        raise


def call_llm_with_functions(
        prompt: str,
        function_executor: Callable[[str, Dict[str, Any]], Dict[str, Any]],
        context: Optional[Dict[str, Any]] = None,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_iterations: int = 5,
        use_reasoning: bool = False
) -> Dict[str, Any]:
    """Call LLM with function calling support and optional reasoning

    Args:
        prompt: User prompt
        function_executor: Callback to execute function calls
                          Signature: function_executor(function_name: str, args: dict) -> dict
        context: Additional context to merge into function arguments (e.g., user_id)
        model: Model to use (random if None)
        temperature: Sampling temperature (0-2)
        max_iterations: Maximum number of function call loops to prevent infinite loops
        use_reasoning: Enable reasoning mode for supported models

    Returns:
        Dictionary with 'response' and optionally 'reasoning_details'
    """
    api_key = random.choice(API_KEYS)

    # Select model based on reasoning requirement
    if use_reasoning:
        if not REASONING_MODELS:
            logger.warning("Reasoning requested but no reasoning models available. Using standard model.")
            model = model or random.choice(MODELS)
            use_reasoning = False
        else:
            model = model or random.choice(REASONING_MODELS)
    else:
        model = model or random.choice(MODELS)

    context = context or {}

    logger.info(
        f"Starting function calling - Model: {model}, Reasoning: {use_reasoning}, Max iterations: {max_iterations}")

    client = OpenAI(
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1"
    )

    messages = [{"role": "user", "content": prompt}]
    reasoning_details = None

    for iteration in range(max_iterations):
        logger.info(f"Function calling iteration {iteration + 1}/{max_iterations}")

        try:
            kwargs = {
                "model": model,
                "messages": messages,
                "tools": AVAILABLE_FUNCTIONS,
                "tool_choice": "auto",
                "temperature": temperature,
                "timeout": 60.0
            }

            if use_reasoning:
                kwargs["reasoning"] = {"enabled": True}

            response = client.chat.completions.create(**kwargs)

            message = response.choices[0].message

            # Capture reasoning details if available
            if use_reasoning and hasattr(message, 'reasoning_details'):
                reasoning_details = message.reasoning_details

            # If no tool calls, we're done - return final answer
            if not message.tool_calls:
                logger.info("No more function calls - returning final answer")
                return {
                    "response": message.content or "I apologize, but I couldn't generate a response.",
                    "reasoning_details": reasoning_details
                }

            # Log function calls
            logger.info(f"LLM requesting {len(message.tool_calls)} function call(s)")

            # Add assistant message with tool calls to history
            assistant_msg = {
                "role": "assistant",
                "content": message.content,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments
                        }
                    }
                    for tc in message.tool_calls
                ]
            }

            # Preserve reasoning details in message history
            if reasoning_details:
                assistant_msg["reasoning_details"] = reasoning_details

            messages.append(assistant_msg)

            # Execute each tool call
            for tool_call in message.tool_calls:
                function_name = tool_call.function.name

                try:
                    function_args = json.loads(tool_call.function.arguments)
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse function arguments: {str(e)}")
                    function_result = {"error": "Invalid function arguments"}
                    result_content = json.dumps(function_result)
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": result_content
                    })
                    continue

                # Merge context (like user_id) into function arguments
                function_args.update(context)

                logger.info(f"Executing function: {function_name} with args: {function_args}")

                # Execute the function
                try:
                    function_result = function_executor(function_name, function_args)
                    result_content = json.dumps(function_result)
                    logger.info(f"Function {function_name} completed successfully")
                except Exception as e:
                    logger.error(f"Function execution error in {function_name}: {str(e)}")
                    function_result = {"error": str(e), "status": "error"}
                    result_content = json.dumps(function_result)

                # Add function result to messages
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result_content
                })

        except OpenAIError as e:
            logger.error(f"OpenAI API error during function calling: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error during function calling: {str(e)}")
            raise

    # If we hit max iterations, return explanation
    logger.warning(f"Reached maximum iterations ({max_iterations})")
    return {
        "response": "I've gathered a lot of information but need to stop here. Please ask a more specific question or break your request into smaller parts.",
        "reasoning_details": reasoning_details
    }


def call_llm(prompt: str, model: Optional[str] = None, use_mcp: bool = False) -> str:
    """Legacy function for backwards compatibility

    Args:
        prompt: User prompt
        model: Model to use
        use_mcp: Ignored (kept for compatibility)

    Returns:
        LLM response
    """
    return call_llm_simple(prompt, model)