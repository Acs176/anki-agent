# agent.py
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from pydantic_ai import Agent, RunContext
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

from . import anki
from .logging_utils import logs_handler
from .model import (
    AdjCard,
    FallbackCard,
    FlashcardType,
    NounCard,
    PhraseCard,
    RouterFailure,
    VerbCard,
)

logger = logs_handler.get_logger()


@dataclass
class Deps:
    deck: str
    target_lang: str


class AnkiAgent:
    agent: Agent

    def __init__(self, model_name, api_key):
        logger.info("Initializing AnkiAgent with model=%s", model_name)
        self._tool_called = False
        model = OpenAIChatModel(model_name, provider=OpenAIProvider(api_key=api_key))
        noun_agent = Agent(
            model=model,
            deps_type=Deps,
            output_type=NounCard,
            system_prompt=(
                "INPUT: a source word/phrase and a TARGET language.\n"
                "TASK: Return a NounCard object:\n"
                "1) Translation: <concise translation into TARGET>\n"
                "2) Some clarifications:\n"
                "definite_sg and definite_pl are singular and plural in the defined form, respectively\n"
                "sample is a useful sample phrase with the noun"
            ),
            instrument=True,
        )

        verb_agent = Agent(
            model=model,
            deps_type=Deps,
            output_type=VerbCard,
            system_prompt=(
                "INPUT: a verb (infinitive) and a TARGET language.\n"
                "TASK: Build a VerbOut object:\n"
                "1) Translation: <word in target language>\n"
                "2) The rest of fields represent the different tenses and sample phrases in those tenses."
                "Use common and useful phrases. Make up a different phrase for each tense."
            ),
            instrument=True,
        )

        adj_agent = Agent(
            model=model,
            deps_type=Deps,
            output_type=AdjCard,
            system_prompt=(
                "INPUT: an adjective and TARGET language."
                " Return AdjCard with translation, positive, comparative, superlative (opt), sample."
            ),
            instrument=True,
        )

        phrase_agent = Agent(
            model=model,
            deps_type=Deps,
            output_type=PhraseCard,
            system_prompt=(
                "INPUT: a phrase/expression and TARGET language."
                " Return PhraseCard with text_sv, translation, pattern (opt), sample."
            ),
            instrument=True,
        )

        fallback_agent = Agent(
            model=model,
            deps_type=Deps,
            output_type=FallbackCard,
            system_prompt=(
                "INPUT may be ambiguous/non-lexical."
                " Return FallbackCard with source, optional translation/sample/notes."
            ),
            instrument=True,
        )
        prompt_dir_path = os.getenv("PROMPTS_PATH")
        # TODO: Centralize environment variable loading in a Config class
        if not prompt_dir_path:
            raise Exception("PROMPTS_PATH must be an env variable")
        router_prompt_path = Path(prompt_dir_path + "/router.txt")
        if not router_prompt_path.is_file():
            raise FileNotFoundError(
                f"Controller system prompt not found at {router_prompt_path.resolve()}"
            )
        controller_system_prompt = router_prompt_path.read_text(encoding="utf-8").strip()
        logger.info("Loaded controller system prompt from %s", router_prompt_path.resolve())

        async def make_noun_card(
            ctx: RunContext[Deps],
            source: str,
        ) -> NounCard:
            if self._tool_called:
                logger.warning("Duplicate tool call prevented: noun | source='%s'", source)
                return "duplicate_tool_call_ignored"
            self._tool_called = True
            logger.info(
                "Router chose: noun | source='%s' | deck='%s' | target='%s'",
                source,
                ctx.deps.deck,
                ctx.deps.target_lang,
            )

            prompt = f"SOURCE: {source}\nTARGET: {ctx.deps.target_lang}"
            logger.debug("noun sub-agent prompt: %s", prompt)
            # Ask noun sub-agent to produce two lines.
            result = await noun_agent.run(prompt)
            logger.debug("noun sub-agent output: %s", result.output)
            logger.debug("noun sub-agent messages: %s", result.all_messages)
            # Return the structured card; caller will post to Anki
            return result.output

        async def make_verb_card(
            ctx: RunContext[Deps],
            source: str,
        ) -> VerbCard:
            if self._tool_called:
                logger.warning("Duplicate tool call prevented: verb | source='%s'", source)
                return "duplicate_tool_call_ignored"
            self._tool_called = True
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
            return result.output

        async def make_adj_card(
            ctx: RunContext[Deps],
            source: str,
        ) -> AdjCard:
            if self._tool_called:
                logger.warning("Duplicate tool call prevented: adjective | source='%s'", source)
                return "duplicate_tool_call_ignored"
            self._tool_called = True
            logger.info(
                "Router chose: adjective | source='%s' | deck='%s' | target='%s'",
                source,
                ctx.deps.deck,
                ctx.deps.target_lang,
            )

            prompt = f"ADJECTIVE: {source}\nTARGET: {ctx.deps.target_lang}"
            logger.debug("Adj sub-agent prompt: %s", prompt)
            result = await adj_agent.run(prompt)
            logger.debug("Adj sub-agent output: %s", result.output)
            return result.output

        async def make_phrase_card(
            ctx: RunContext[Deps],
            source: str,
        ) -> PhraseCard:
            if self._tool_called:
                logger.warning("Duplicate tool call prevented: phrase | source='%s'", source)
                return "duplicate_tool_call_ignored"
            self._tool_called = True
            logger.info(
                "Router chose: phrase | source='%s' | deck='%s' | target='%s'",
                source,
                ctx.deps.deck,
                ctx.deps.target_lang,
            )

            prompt = f"PHRASE: {source}\nTARGET: {ctx.deps.target_lang}"
            logger.debug("Phrase sub-agent prompt: %s", prompt)
            result = await phrase_agent.run(prompt)
            logger.debug("Phrase sub-agent output: %s", result.output)
            return result.output

        async def make_fallback_card(
            ctx: RunContext[Deps],
            source: str,
            reason: str | None = None,
        ) -> FallbackCard:
            if self._tool_called:
                logger.warning(
                    "Duplicate tool call prevented: fallback | source='" + str(source) + "'"
                )
                return "duplicate_tool_call_ignored"
            self._tool_called = True
            logger.info(
                "Router chose: fallback | source='%s' | deck='%s' | target='%s' | reason='%s'",
                source,
                ctx.deps.deck,
                ctx.deps.target_lang,
                reason,
            )

            prompt = f"FALLBACK SOURCE: {source}\nTARGET: {ctx.deps.target_lang}" + (
                f"\nREASON: {reason}" if reason else ""
            )
            logger.debug("Fallback sub-agent prompt: %s", prompt)
            result = await fallback_agent.run(prompt)
            logger.debug("Fallback sub-agent output: %s", result.output)
            return result.output

        # Now that tool functions are defined, create the controller Agent.
        controller = Agent(
            model=model,
            deps_type=Deps,
            system_prompt=controller_system_prompt,
            output_type=[
                make_noun_card,
                make_adj_card,
                make_verb_card,
                make_phrase_card,
                make_fallback_card,
                RouterFailure,
            ],
            instrument=True,
        )

        self.agent = controller

    async def add_word_async(self, word: str, deck: str, target_lang: str) -> list[str]:
        logger.info("Adding word: '%s' to deck='%s' target='%s'", word, deck, target_lang)
        self._tool_called = False
        deps = Deps(deck=deck, target_lang=target_lang)
        user_message = f"Word: {word}\nTarget language: {target_lang}"
        logger.debug("Controller agent user_message: %s", user_message)
        logger.debug("Controller agent deps: deck=%s target=%s", deps.deck, deps.target_lang)
        result = await self.agent.run(user_message, deps=deps)
        logger.debug("Controller agent messages: %s", result.all_messages)

        output = result.output
        if isinstance(output, FlashcardType):
            logger.debug(
                "Posting flashcard via anki.add_flashcard | deck=%s word=%s type=%s",
                deck,
                word,
                type(output).__name__,
            )
            note_id = anki.add_flashcard(deck, word, output)
            if note_id == anki.DUPLICATE_NOTE:
                logger.info("Flashcard duplicate; not created")
            else:
                logger.info("Note created: id=%s", note_id)
        elif isinstance(output, RouterFailure):
            logger.error("RouterFailure: %s", output.explanation)
        else:
            logger.error("Unexpected router output type: %s", type(output))

        # Return full trace for observability / tests
        return result.all_messages

    def add_word(self, word: str, deck: str, target_lang: str) -> list[str]:
        logger.info("Adding word: '%s' to deck='%s' target='%s'", word, deck, target_lang)
        self._tool_called = False
        deps = Deps(deck=deck, target_lang=target_lang)
        user_message = f"Word: {word}\nTarget language: {target_lang}"
        logger.debug("Controller agent user_message: %s", user_message)
        logger.debug("Controller agent deps: deck=%s target=%s", deps.deck, deps.target_lang)
        result = self.agent.run_sync(user_message, deps=deps)
        logger.debug("Controller agent messages: %s", result.all_messages)

        output = result.output
        if isinstance(output, FlashcardType):
            logger.debug(
                "Posting flashcard via anki.add_flashcard | deck=%s word=%s type=%s",
                deck,
                word,
                type(output).__name__,
            )
            note_id = anki.add_flashcard(deck, word, output)
            if note_id == anki.DUPLICATE_NOTE:
                logger.info("Flashcard duplicate; not created")
            else:
                logger.info("Note created: id=%s", note_id)
        elif isinstance(output, RouterFailure):
            logger.error("RouterFailure: %s", output.explanation)
        else:
            logger.error("Unexpected router output type: %s", type(output))

        return result.all_messages
