import json
import urllib.request

from pydantic import BaseModel

from .model import NonVerbOut, VerbOut

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


def add_flashcard(
    deck_name: str,
    source_word: str,
    data: BaseModel,
    tags: list[str] | None = None,
):
    match data:
        case NonVerbOut():
            return add_nonverb_flashcard(deck_name, source_word, data, tags)
        case VerbOut():
            return add_verb_flashcard(deck_name, source_word, data, tags)
        case _:
            raise TypeError(
                f"Unsupported data type {type(data).__name__}. Expected NonVerbOut or VerbOut."
            )


def add_nonverb_flashcard(
    deck_name: str,
    source_word: str,
    data: NonVerbOut,
    tags: list[str] | None = None,
):
    """
    Create an Anki flashcard for a non-verb word using NonVerbOut data.
    """
    front = source_word
    back = f"Translation: {data.translation}\nSample: {data.sample}"
    return add_basic_note(deck_name, front, back, tags=(tags or []) + ["ai", "nonverb"])


def add_verb_flashcard(
    deck_name: str,
    source_word: str,
    data: VerbOut,
    tags: list[str] | None = None,
):
    """
    Create an Anki flashcard for a verb word using VerbOut data.
    """
    front = f"{source_word} — (verb)"
    back = (
        f"Translation: {data.translation}\n"
        f"Forms:\n"
        f"- Infinitive: {data.infinitive}\n"
        f"- Present: {data.present} — {data.sample_present}\n"
        f"- Past: {data.past} — {data.sample_past}\n"
        f"- Supine: {data.supine} — {data.sample_supine}\n"
        f"- Imperative: {data.imperative} — {data.sample_imperative}"
    )
    return add_basic_note(deck_name, front, back, tags=(tags or []) + ["ai", "verb"])


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
