import json
import urllib.error
import urllib.request
from typing import Any

from .logging_utils import logs_handler
from .model import AdjCard, FallbackCard, FlashcardType, NounCard, PhraseCard, VerbCard

ANKI_CONNECT_URL = "http://127.0.0.1:8765"
API_VERSION = 6
API_KEY = None
# Special return for duplicate note attempts
DUPLICATE_NOTE = -2

logger = logs_handler.get_logger()


def _payload(action: str, params: dict[str, Any] | None = None):
    body = {"action": action, "version": API_VERSION}
    if params:
        body["params"] = params

    if API_KEY is not None:
        # AnkiConnect accepts a top-level 'key' in the JSON body
        body["key"] = API_KEY
    return json.dumps(body).encode("utf-8")


def invoke(action: str, **params):
    logger.debug("Invoking AnkiConnect action=%s params=%s", action, list(params.keys()))
    req = urllib.request.Request(ANKI_CONNECT_URL, _payload(action, params))
    try:
        with urllib.request.urlopen(req) as resp:
            response = json.load(resp)
    except (
        OSError,
        urllib.error.URLError,
        urllib.error.HTTPError,
        ConnectionError,
        TimeoutError,
    ) as e:
        logger.error(
            "Failed to reach AnkiConnect at %s for action=%s: %s",
            ANKI_CONNECT_URL,
            action,
            e,
        )
        raise RuntimeError(
            f"Cannot connect to AnkiConnect at {ANKI_CONNECT_URL}. Is Anki running and AnkiConnect enabled?"
        ) from e
    except json.JSONDecodeError as e:
        logger.error("Invalid JSON from AnkiConnect for action=%s: %s", action, e)
        raise RuntimeError("Invalid JSON response from AnkiConnect") from e

    if "error" not in response or "result" not in response:
        logger.error("Invalid AnkiConnect response: keys=%s", list(response.keys()))
        raise RuntimeError("Invalid AnkiConnect response")
    if response["error"] is not None:
        logger.error("AnkiConnect error for action=%s: %s", action, response["error"])
        raise RuntimeError(response["error"])
    logger.debug("AnkiConnect action=%s result=%s", action, response["result"])
    return response["result"]


def ensure_deck(deck_name: str):
    logger.info("Ensuring deck exists: %s", deck_name)
    return invoke("createDeck", deck=deck_name)  # returns deck ID if it created one


def add_flashcard(
    deck_name: str,
    source_word: str,
    data: FlashcardType,
    tags: list[str] = None,
):
    if isinstance(data, NounCard):
        return add_noun_flashcard(deck_name, source_word, data, tags)
    if isinstance(data, AdjCard):
        return add_adj_flashcard(deck_name, source_word, data, tags)
    if isinstance(data, VerbCard):
        return add_verb_flashcard(deck_name, source_word, data, tags)
    if isinstance(data, PhraseCard):
        return add_phrase_flashcard(deck_name, source_word, data, tags)
    if isinstance(data, FallbackCard):
        return add_fallback_flashcard(deck_name, source_word, data, tags)
    raise TypeError(f"Unsupported data type {type(data).__name__}.")


def add_noun_flashcard(
    deck_name: str,
    source_word: str,
    data: NounCard,
    tags: list[str] = None,
):
    logger.info("Adding noun flashcard: deck=%s word=%s", deck_name, source_word)
    front = source_word
    back = (
        f"Translation: {data.translation}\n"
        f"Article: {data.article}\n"
        f"Plural: {data.plural}\n"
        f"Definite: {data.definite_sg} (sg), {data.definite_pl} (pl)\n"
        f"Sample: {data.sample}"
    )
    return add_basic_note(deck_name, front, back, tags=(tags or []) + ["ai", "noun"])


def add_adj_flashcard(
    deck_name: str,
    source_word: str,
    data: AdjCard,
    tags: list[str] = None,
):
    logger.info("Adding adjective flashcard: deck=%s word=%s", deck_name, source_word)
    front = source_word
    lines = [
        f"Translation: {data.translation}",
        f"Positive: {data.positive}",
    ]
    if getattr(data, "comparative", None):
        lines.append(f"Comparative: {data.comparative}")
    if getattr(data, "superlative", None):
        lines.append(f"Superlative: {data.superlative}")
    lines.append(f"Sample: {data.sample}")
    back = "\n".join(lines)
    return add_basic_note(deck_name, front, back, tags=(tags or []) + ["ai", "adjective"])


def add_verb_flashcard(
    deck_name: str,
    source_word: str,
    data: VerbCard,
    tags: list[str] = None,
):
    logger.info("Adding verb flashcard: deck=%s word=%s", deck_name, source_word)
    front = f"{source_word} (verb)"
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


def add_phrase_flashcard(
    deck_name: str,
    source_word: str,
    data: PhraseCard,
    tags: list[str] = None,
):
    logger.info("Adding phrase flashcard: deck=%s word=%s", deck_name, source_word)
    front = source_word
    lines = [
        f"Phrase: {data.text_sv}",
        f"Translation: {data.translation}",
    ]
    if getattr(data, "pattern", None):
        lines.append(f"Pattern: {data.pattern}")
    lines.append(f"Sample: {data.sample}")
    back = "\n".join(lines)
    return add_basic_note(deck_name, front, back, tags=(tags or []) + ["ai", "phrase"])


def add_fallback_flashcard(
    deck_name: str,
    source_word: str,
    data: FallbackCard,
    tags: list[str] = None,
):
    logger.info("Adding fallback flashcard: deck=%s word=%s", deck_name, source_word)
    front = source_word
    lines = [f"Source: {data.source}"]
    if getattr(data, "translation", None):
        lines.append(f"Translation: {data.translation}")
    if getattr(data, "sample", None):
        lines.append(f"Sample: {data.sample}")
    if getattr(data, "notes", None):
        lines.append(f"Notes: {data.notes}")
    back = "\n".join(lines)
    return add_basic_note(deck_name, front, back, tags=(tags or []) + ["ai", "fallback"])


def add_basic_note(deck_name: str, front: str, back: str, tags=None):
    logger.debug(
        "Submitting Basic note: deck=%s tags=%s front_len=%d back_len=%d",
        deck_name,
        (tags or []),
        len(front),
        len(back),
    )
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
    try:
        return invoke("addNote", note=note)  # returns note id on success
    except RuntimeError as e:
        msg = str(e)
        if "duplicate" in msg.lower():
            # AnkiConnect duplicate note error
            logger.info("Duplicate note detected; skipping creation")
            return DUPLICATE_NOTE
        logger.error(f"Error creating the flashcard: {e}")
        return -1
