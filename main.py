import asyncio
import os

from dotenv import load_dotenv

import agent

if __name__ == "__main__":
    load_dotenv()
    anki_agent = agent.AnkiAgent("gpt-5-nano", os.getenv("OPENAI_API_KEY"))
    asyncio.run(anki_agent.add_word("potatis", "test", "eng"))
