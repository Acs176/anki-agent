import asyncio
import os

from dotenv import load_dotenv
from langfuse import get_client
from pydantic_ai import Agent

from .anki_agent.logging_utils import logs_handler
from .anki_agent.orchestrator import AnkiAgentOrchestrator

Agent.instrument_all()

if __name__ == "__main__":
    load_dotenv()
    logs_handler.setup_logging(level="debug")
    logger = logs_handler.get_logger()
    langfuse = get_client()
    if langfuse.auth_check():
        logger.debug("Langfuse client authenticated and ready!")
    else:
        logger.error("Langfuse authentication failed")
    # Configure simple logging; override via LOG_LEVEL env var
    anki_agent = AnkiAgentOrchestrator("gpt-4o-mini", os.getenv("OPENAI_API_KEY"))
    asyncio.run(anki_agent.add_word_async("fasting", "test", "svenska"))
