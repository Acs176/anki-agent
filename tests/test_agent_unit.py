import pytest

from anki_agent import orchestrator


@pytest.mark.asyncio
async def test_add_word_calls_controller_with_expected_args(monkeypatch):
    # Construct with fake model/api key; we'll stub out .agent.run below
    a = orchestrator.AnkiAgentOrchestrator(model_name="fake", api_key="fake")

    captured = {}

    async def fake_run(user_message, deps=None):
        captured["user_message"] = user_message
        captured["deps"] = deps

        class FakeResult:
            output = "note_id=123"
            all_messages = ["tool: make_nonverb_card", "note_id=123"]

        return FakeResult()

    # Replace the controller agent with a simple stub exposing async .run
    class FakeController:
        async def run(self, user_message, deps=None):
            return await fake_run(user_message, deps=deps)

    a.agent = FakeController()

    result = await a.add_word_async("immediately", "test", "svenska")

    assert "Word: immediately" in captured["user_message"]
    assert "Target language: svenska" in captured["user_message"]
    assert captured["deps"].deck == "test"
    assert captured["deps"].target_lang == "svenska"
    assert result == ["tool: make_nonverb_card", "note_id=123"]
