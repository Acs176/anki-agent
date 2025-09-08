import asyncio
import os

from dotenv import load_dotenv

from .anki_agent.agent import AnkiAgent

if __name__ == "__main__":
    load_dotenv()
    anki_agent = AnkiAgent("gpt-5-nano", os.getenv("OPENAI_API_KEY"))
    asyncio.run(anki_agent.add_word("immediately", "test", "svenska"))
