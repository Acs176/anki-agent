# agent.py
from __future__ import annotations

from dataclasses import dataclass

from pydantic_ai import Agent, RunContext
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

from . import anki
from .logging_utils import logs_handler
from .model import NonVerbOut, VerbOut

logger = logs_handler.get_logger()


@dataclass
class Deps:
    deck: str
    target_lang: str


class AnkiAgent:
    agent: Agent

    def __init__(self, model_name, api_key):
        logger.info("Initializing AnkiAgent with model=%s", model_name)
        model = OpenAIChatModel(model_name, provider=OpenAIProvider(api_key=api_key))
        nonverb_agent = Agent(
            model=model,
            deps_type=Deps,
            output_type=NonVerbOut,
            system_prompt=(
                "INPUT: a source word/phrase and a TARGET language.\n"
                "TASK: Return a NonVerbOut object:\n"
                "1) Translation: <concise translation into TARGET>\n"
                "2) Sample phrase: <natural and useful sample phrase using the word>\n"
                "Keep it brief. No extra commentary."
            ),
        )

        verb_agent = Agent(
            model=model,
            deps_type=Deps,
            output_type=VerbOut,
            system_prompt=(
                "INPUT: a verb (infinitive) and a TARGET language.\n"
                "TASK: Build a VerbOut object:\n"
                "1) Translation: <word in target language>\n"
                "2) The rest of fields represent the different tenses and sample phrases in those tenses."
                "Use common and useful phrases. Make up a different phrase for each tense."
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
            logger.info(
                "Router chose: nonverb | source='%s' | deck='%s' | target='%s'",
                source,
                ctx.deps.deck,
                ctx.deps.target_lang,
            )
            prompt = f"SOURCE: {source}\nTARGET: {ctx.deps.target_lang}"
            logger.debug("Nonverb sub-agent prompt: %s", prompt)
            # Ask nonverb sub-agent to produce two lines.
            result = await nonverb_agent.run(prompt)
            logger.debug("Nonverb sub-agent output: %s", result.output)
            logger.debug("Nonverb sub-agent messages: %s", result.all_messages)
            logger.debug(
                "Adding flashcard via anki.add_flashcard for nonverb | deck=%s word=%s",
                ctx.deps.deck,
                source,
            )
            note_id = anki.add_flashcard(
                ctx.deps.deck, source, result.output, tags=["ai", "nonverb"]
            )
            logger.info("Nonverb note created: id=%s", note_id)
            return f"note_id={note_id}"

        @controller.tool
        async def make_verb_card(ctx: RunContext[Deps], source: str):
            logger.info(
                "Router chose: verb | source='%s' | deck='%s' | target='%s'",
                source,
                ctx.deps.deck,
                ctx.deps.target_lang,
            )
            prompt = f"VERB: {source}\nTARGET: {ctx.deps.target_lang}"
            logger.debug("Verb sub-agent prompt: %s", prompt)
            # Ask verb sub-agent to produce full back text in one shot.
            result = await verb_agent.run(prompt)
            logger.debug("Verb sub-agent output: %s", result.output)
            logger.debug("Verb sub-agent messages: %s", result.all_messages)
            logger.debug(
                "Adding flashcard via anki.add_flashcard for verb | deck=%s word=%s",
                ctx.deps.deck,
                source,
            )
            note_id = anki.add_flashcard(ctx.deps.deck, source, result.output, tags=["ai", "verb"])
            logger.info("Verb note created: id=%s", note_id)
            return f"note_id={note_id}"

        self.agent = controller

    async def add_word(self, word: str, deck: str, target_lang: str) -> list[str]:
        logger.info("Adding word: '%s' to deck='%s' target='%s'", word, deck, target_lang)
        deps = Deps(deck=deck, target_lang=target_lang)
        user_message = f"Word: {word}\nTarget language: {target_lang}"
        logger.debug("Controller agent user_message: %s", user_message)
        logger.debug("Controller agent deps: deck=%s target=%s", deps.deck, deps.target_lang)
        result = await self.agent.run(user_message, deps=deps)
        logger.debug("Controller agent messages: %s", result.all_messages)
        return result.all_messages  # final model text (includes our tool's return string)
