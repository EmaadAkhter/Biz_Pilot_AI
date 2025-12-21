import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# Initialize OpenAI client with OpenRouter's base URL
client = OpenAI(
    api_key=OPENROUTER_API_KEY,
    base_url="https://openrouter.ai/api/v1"
)


def call_llm(
        prompt: str,
        model: str = "openai/gpt-oss-120b:free",
        site_url: str = None,
        site_name: str = None
) -> str:
    """Call LLM via OpenRouter API using OpenAI client

    Args:
        prompt: The user prompt to send to the model
        model: The model to use
        site_url: Optional site URL for rankings on openrouter.ai
        site_name: Optional site name for rankings on openrouter.ai

    Returns:
        The assistant's response content
    """
    extra_headers = {}
    if site_url:
        extra_headers["HTTP-Referer"] = site_url
    if site_name:
        extra_headers["X-Title"] = site_name

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        timeout=60.0,
        extra_headers=extra_headers if extra_headers else None,
        extra_body={}
    )

    return response.choices[0].message.content