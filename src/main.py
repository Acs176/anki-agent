import asyncio
import os

from dotenv import load_dotenv

from .anki_agent.agent import AnkiAgent
from .anki_agent.logging_utils import logs_handler

if __name__ == "__main__":
    load_dotenv()
    logs_handler.setup_logging(level="debug")
    # Configure simple logging; override via LOG_LEVEL env var
    anki_agent = AnkiAgent("gpt-5-nano", os.getenv("OPENAI_API_KEY"))
    asyncio.run(anki_agent.add_word("madrugar", "test", "svenska"))
