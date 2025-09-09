import json
import os
from collections.abc import Callable

import pytest

# pydantic-ai testing utilities
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from anki_agent.agent import AnkiAgent

pytestmark = pytest.mark.asyncio


def _ensure_prompts_env():
    # Ensure PROMPTS_PATH is set so the controller system prompt loads
    if not os.getenv("PROMPTS_PATH"):
        os.environ["PROMPTS_PATH"] = "./src/prompts/"


async def _run_with_model(monkeypatch, model_factory: Callable[[], object]):
    """Helper to construct AnkiAgent while injecting a custom pydantic-ai model.

    We patch OpenAIChatModel in the agent module so all sub-agents (controller and
    specialized noun/verb/etc.) use the provided test model instance.
    """
    _ensure_prompts_env()

    # Patch the chat model constructor used inside AnkiAgent.__init__
    from anki_agent import agent as agent_module

    monkeypatch.setattr(
        agent_module,
        "OpenAIChatModel",
        lambda *args, **kwargs: model_factory(),
    )

    # Construct the agent (will use our patched model for all internal Agents)
    a = AnkiAgent(model_name="fake", api_key="fake")
    return a


async def test_router_tool_call_duplicate_guard(monkeypatch):
    # Simulate the controller choosing a tool via FunctionModel and assert the
    # tool duplicate-call guard returns the expected sentinel string on a
    # second tool call attempt.

    # Capture note creations to avoid network calls
    captured = {"count": 0}
    from anki_agent import anki as anki_module

    def fake_add_basic_note(deck_name: str, front: str, back: str, tags=None):
        captured["count"] += 1
        return 321

    monkeypatch.setattr(anki_module, "add_basic_note", fake_add_basic_note)

    def controller_and_subagents(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        # 1) Controller chooses first tool
        if len(messages) == 1:
            return ModelResponse(parts=[ToolCallPart("make_noun_card", {"source": "hund"})])
        # 2) Sub-agent returns JSON for noun
        if len(messages) == 2:
            return ModelResponse(
                parts=[
                    TextPart(
                        json.dumps(
                            {
                                "translation": "dog",
                                "article": "en",
                                "plural": "hundar",
                                "definite_sg": "hunden",
                                "definite_pl": "hundarna",
                                "sample": "En hund springer.",
                            }
                        )
                    )
                ]
            )
        # 3) Controller tries to select a second tool — should be blocked by guard
        if len(messages) == 3:
            return ModelResponse(parts=[ToolCallPart("make_verb_card", {"source": "ata"})])
        # 4) Anything else, finish
        return ModelResponse(parts=[TextPart("done")])

    a = await _run_with_model(monkeypatch, lambda: FunctionModel(controller_and_subagents))

    msgs = await a.add_word("whatever", deck="DemoDeck", target_lang="Swedish")

    # First tool created exactly one note
    assert captured["count"] == 1
    # Second tool attempt was blocked
    assert any("duplicate_tool_call_ignored" in str(m) for m in msgs)


@pytest.mark.parametrize(
    "tool_name, source, subagent_obj, expect_in_back, expect_type_tag, expect_front_suffix",
    [
        (
            "make_noun_card",
            "hund",
            {
                "translation": "dog",
                "article": "en",
                "plural": "hundar",
                "definite_sg": "hunden",
                "definite_pl": "hundarna",
                "sample": "En hund springer.",
            },
            [
                "Translation: dog",
                "Article: en",
                "Plural: hundar",
                "Definite:",
                "Sample: En hund springer.",
            ],
            "noun",
            "",
        ),
        (
            "make_adj_card",
            "vacker",
            {
                "translation": "beautiful",
                "positive": "vacker",
                "comparative": "vackrare",
                "superlative": "vackrast",
                "sample": "En vacker dag.",
            },
            [
                "Translation: beautiful",
                "Positive: vacker",
                "Comparative: vackrare",
                "Superlative: vackrast",
                "Sample: En vacker dag.",
            ],
            "adjective",
            "",
        ),
        (
            "make_verb_card",
            "ata",
            {
                "translation": "eat",
                "infinitive": "ata",
                "present": "ater",
                "past": "at",
                "supine": "atit",
                "imperative": "at!",
                "sample_present": "Jag ater nu.",
                "sample_past": "Jag at igar.",
                "sample_supine": "Jag har atit.",
                "sample_imperative": "At!",
            },
            [
                "Translation: eat",
                "Infinitive: ata",
                "Present: ater",
                "Past: at",
                "Supine: atit",
                "Imperative: at!",
            ],
            "verb",
            " (verb)",
        ),
        (
            "make_phrase_card",
            "ta reda pa",
            {
                "text_sv": "ta reda pa",
                "translation": "find out",
                "pattern": "ta reda pa + ngt",
                "sample": "Jag vill ta reda pa det.",
            },
            [
                "Phrase: ta reda pa",
                "Translation: find out",
                "Pattern: ta reda pa + ngt",
                "Sample: Jag vill ta reda pa det.",
            ],
            "phrase",
            "",
        ),
        (
            "make_fallback_card",
            "http://foo.com",
            {
                "source": "http://foo.com",
                "translation": "",
                "sample": "",
                "notes": "non-lexical URL",
            },
            [
                "Source: http://foo.com",
                "Notes: non-lexical URL",
            ],
            "fallback",
            "",
        ),
    ],
)
async def test_routing_success_adds_note(
    monkeypatch,
    tool_name: str,
    source: str,
    subagent_obj: dict,
    expect_in_back: list[str],
    expect_type_tag: str,
    expect_front_suffix: str,
):
    """End-to-end happy-path: router selects a tool, sub-agent returns data,
    we add a Basic note via Anki payload. Minimal but checks the essentials.
    """

    payload_json = json.dumps(subagent_obj)

    def controller_and_subagents(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        # Controller decides which tool to call on the first model invocation
        if len(messages) == 1:
            return ModelResponse(parts=[ToolCallPart(tool_name, {"source": source})])
        # Sub-agent returns JSON matching its Pydantic output model
        return ModelResponse(parts=[TextPart(payload_json)])

    a = await _run_with_model(monkeypatch, lambda: FunctionModel(controller_and_subagents))

    captured = {}

    # Capture final Anki payload; return deterministic note id
    from anki_agent import anki as anki_module

    def fake_add_basic_note(deck_name: str, front: str, back: str, tags=None):
        captured.update(deck=deck_name, front=front, back=back, tags=list(tags or []))
        return 999

    monkeypatch.setattr(anki_module, "add_basic_note", fake_add_basic_note)

    msgs = await a.add_word("ignored", deck="DemoDeck", target_lang="Swedish")

    # Note id bubbles up through tool return
    assert any("note_id=999" in str(m) for m in msgs)

    # Front matches source (+ suffix for verbs)
    assert captured["front"] == f"{source}{expect_front_suffix}"

    # Back contains the expected key fields for each type
    for expected in expect_in_back:
        assert expected in captured["back"]

    # Deck propagated
    assert captured["deck"] == "DemoDeck"

    # Tags include the type tag and 'ai'
    assert "ai" in set(captured["tags"]) and expect_type_tag in set(captured["tags"])
