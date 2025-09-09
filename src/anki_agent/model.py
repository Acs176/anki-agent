"""
Module for object definitions
"""

from typing import Literal

from pydantic import BaseModel


class NounCard(BaseModel):
    translation: str
    article: Literal["en", "ett"]
    plural: str
    definite_sg: str
    definite_pl: str
    sample: str


class AdjCard(BaseModel):
    translation: str
    positive: str
    comparative: str | None = None
    superlative: str | None = None
    sample: str


class VerbCard(BaseModel):
    translation: str
    infinitive: str
    present: str
    past: str
    supine: str
    imperative: str
    # keep samples very short so they fit on card backs
    sample_present: str
    sample_past: str
    sample_supine: str
    sample_imperative: str


class PhraseCard(BaseModel):
    text_sv: str  # the phrase itself in Swedish
    translation: str
    pattern: str | None = None  # optional slot, e.g. "ha **ont i** + kroppsdel"
    sample: str


class FallbackCard(BaseModel):
    source: str  # just echo back the source word/phrase
    translation: str | None = None
    sample: str | None = None
    notes: str | None = None  # for free-form data if LLM doesn't fit schema


FlashcardType = NounCard | AdjCard | VerbCard | FallbackCard | PhraseCard
