# agent.py
from __future__ import annotations

from dataclasses import dataclass

from pydantic_ai import Agent, RunContext
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

from . import anki


@dataclass
class Deps:
    deck: str
    target_lang: str


class AnkiAgent:
    agent: Agent

    def __init__(self, model_name, api_key):
        model = OpenAIChatModel(model_name, provider=OpenAIProvider(api_key=api_key))
        # Non-verb path: produce translation + a single short sample phrase.
        nonverb_agent = Agent(
            model=model,
            deps_type=Deps,
            system_prompt=(
                "INPUT: a source word/phrase and a TARGET language.\n"
                "TASK: Return two lines ONLY:\n"
                "1) Translation: <concise translation into TARGET>\n"
                "2) Sample: <natural and useful sample phrase using the word>\n"
                "Keep it brief. No extra commentary."
            ),
        )

        # Verb path: produce translation + a compact table of key present-tense conjugations,
        # each with an ultra-short sample phrase, all as text we can place on one card back.
        verb_agent = Agent(
            model=model,
            deps_type=Deps,
            system_prompt=(
                "INPUT: a verb (infinitive) and a TARGET language.\n"
                "TASK: Build a compact text block for ONE flashcard back with:\n"
                "1) Translation: <concise meaning>\n"
                "2) Conjugations (Present, Past, Imperative, Past perfect): list 4-6 common person/number forms with different and commonly used phrases as '- <tense>: <conjugation> - <sample>'\n"
                "   Example style: '- Present: speak - I speak quickly.'\n"
                "Keep lines short and clean. No extra commentary."
            ),
        )

        # --- Controller agent: decides verb vs not and ROUTES by calling internal tools ---

        controller = Agent(
            model=model,
            deps_type=Deps,
            system_prompt=(
                "You are a router. Given a SOURCE word and TARGET language:\n"
                "- Decide if SOURCE is a verb (in general, dictionary sense).\n"
                "- If VERB: call tool make_verb_card(source, target_lang, deck)\n"
                "- If NOT a verb: call tool make_nonverb_card(source, target_lang, deck)\n"
                "Do not output explanations. Just call exactly one tool."
            ),
        )

        @controller.tool
        async def make_nonverb_card(ctx: RunContext[Deps], source: str):
            # Ask nonverb sub-agent to produce two lines.
            result = await nonverb_agent.run(f"SOURCE: {source}\nTARGET: {ctx.deps.target_lang}")
            text = result.output.strip().splitlines()
            # Expecting: ["Translation: ...", "Sample: ..."]
            translation = ""
            sample = ""
            for line in text:
                if line.lower().startswith("translation:"):
                    translation = line.split(":", 1)[1].strip()
                elif line.lower().startswith("sample:"):
                    sample = line.split(":", 1)[1].strip()

            front = source
            back = f"{translation}\nSample: {sample}"
            note_id = anki.add_basic_note(ctx.deps.deck, front, back, tags=["ai", "nonverb"])
            return f"note_id={note_id}"

        @controller.tool
        async def make_verb_card(ctx: RunContext[Deps], source: str):
            # Ask verb sub-agent to produce full back text in one shot.
            result = await verb_agent.run(f"VERB: {source}\nTARGET: {ctx.deps.target_lang}")
            back_block = result.output.strip()
            # Front shows infinitive and a small hint it's a verb.
            front = f"{source} - (verb)"
            note_id = anki.add_basic_note(ctx.deps.deck, front, back_block, tags=["ai", "verb"])
            return f"note_id={note_id}"

        self.agent = controller

    async def add_word(self, word: str, deck: str, target_lang: str) -> list[str]:
        deps = Deps(deck=deck, target_lang=target_lang)
        user_message = f"Word: {word}\nTarget language: {target_lang}"
        result = await self.agent.run(user_message, deps=deps)
        print(result.all_messages)
        return result.all_messages  # final model text (includes our tool's return string)
