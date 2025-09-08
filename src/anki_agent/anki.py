import json
import urllib.request

ANKI_CONNECT_URL = "http://127.0.0.1:8765"
API_VERSION = 6
API_KEY = None


def _payload(action, params=None):
    body = {"action": action, "version": API_VERSION}
    if params:
        body["params"] = params

    if API_KEY is not None:
        # AnkiConnect accepts a top-level 'key' in the JSON body
        body["key"] = API_KEY
    return json.dumps(body).encode("utf-8")


def invoke(action, **params):
    req = urllib.request.Request(ANKI_CONNECT_URL, _payload(action, params))
    with urllib.request.urlopen(req) as resp:
        response = json.load(resp)
    if "error" not in response or "result" not in response:
        raise RuntimeError("Invalid AnkiConnect response")
    if response["error"] is not None:
        raise RuntimeError(response["error"])
    return response["result"]


def ensure_deck(deck_name: str):
    # createDeck won't overwrite, it just ensures it exists
    return invoke("createDeck", deck=deck_name)  # returns deck ID if it created one


def add_basic_note(deck_name: str, front: str, back: str, tags=None):
    note = {
        "deckName": deck_name,
        "modelName": "Basic",
        "fields": {"Front": front, "Back": back},
        "options": {
            "allowDuplicate": False,
            "duplicateScope": "deck",
            "duplicateScopeOptions": {
                "deckName": deck_name,
                "checkChildren": False,
                "checkAllModels": False,
            },
        },
        "tags": tags or [],
    }
    return invoke("addNote", note=note)  # returns note id on success


if __name__ == "__main__":
    deck = "test"  # any deck path you like; subdecks use '::'
    ensure_deck(deck)
    note_id = add_basic_note(deck, front="hablar", back="to speak", tags=["auto"])
    print("Created note id:", note_id)
