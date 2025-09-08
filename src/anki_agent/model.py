"""
Module for object definitions
"""

from pydantic import BaseModel


class NonVerbOut(BaseModel):
    translation: str
    sample_phrase: str


class VerbOut(BaseModel):
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
