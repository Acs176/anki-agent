import io
import json

import anki


class FakeResponse(io.BytesIO):
    """
    We want to stub request.urlopen - That function is a context manager in Python
    (used with 'with'). This means that it must implement __enter__ and __exit__.

    Also, the response must support to be called by json.load(resp). This means that
    it must be 'readable'. That's why we make it io.BytesIO (it implements .read())
    """

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


def make_resp(result=None, error=None):
    payload = json.dumps({"result": result, "error": error}).encode("utf-8")
    return FakeResponse(payload)


def test_invoke_success(monkeypatch):
    def fake_urlopen(req):
        return make_resp(result=42, error=None)

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    assert anki.invoke("testAction", foo="bar") == 42


def test_invoke_error(monkeypatch):
    def fake_urlopen(req):
        return make_resp(result=None, error="oops")

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    try:
        anki.invoke("x")
        raise RuntimeError("should raise")
    except RuntimeError as e:
        assert "oops" in str(e)


def test_add_basic_note_builds_payload(monkeypatch):
    captured = {}

    def fake_urlopen(req):
        # urllib.request.Request stores data on the object
        body = json.loads(req.data.decode("utf-8"))
        captured["body"] = body
        ## capture the body (it will have all the payload passed inside invoke)
        return make_resp(result=123, error=None)

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    note_id = anki.add_basic_note("MyDeck", "Front", "Back", tags=["ai"])
    assert note_id == 123

    b = captured["body"]
    assert b["action"] == "addNote"
    assert b["params"]["note"]["deckName"] == "MyDeck"
    assert b["params"]["note"]["fields"] == {"Front": "Front", "Back": "Back"}
    assert b["params"]["note"]["tags"] == ["ai"]
