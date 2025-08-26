# agent.py
from __future__ import annotations

from dataclasses import dataclass

from pydantic_ai import Agent, RunContext
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

import anki


@dataclass
class Deps:
    deck: str
    target_lang: str


class AnkiAgent:
    agent: Agent

    def __init__(self, model_name, api_key):
        model = OpenAIChatModel(model_name, provider=OpenAIProvider(api_key=api_key))
        agent = Agent(
            model=model,
            deps_type=Deps,
            system_prompt=(
                "You are a language helper. Given a single input word or short phrase, "
                "translate it to the user's target language succinctly.\n"
                "Then call the 'add_flashcard' tool with: word (front) and translation (back).\n"
                "Be concise; no extra chatter."
            ),
        )

        @agent.tool
        def add_flashcard(ctx: RunContext[Deps], word: str, translation: str) -> str:
            """Create a Basic note in Anki with Front=word and Back=translation."""
            anki.ensure_deck(ctx.deps.deck)
            note_id = anki.add_basic_note(ctx.deps.deck, word, translation, tags=["ai"])
            return f"Created note_id={note_id}"

        self.agent = agent

    async def add_word(self, word: str, deck: str, target_lang: str) -> str:
        deps = Deps(deck=deck, target_lang=target_lang)
        user_message = f"Word: {word}\nTarget language: {target_lang}"
        result = await self.agent.run(user_message, deps=deps)
        print(result.all_messages)
        return result.all_messages  # final model text (includes our tool’s return string)
