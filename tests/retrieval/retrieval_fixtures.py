"""Shared synthetic corpus and a fake embedder for the retrieval tests (no model, no network)."""

from __future__ import annotations

import hashlib
import re

import numpy as np
import pandas as pd

LANGS = ("de", "fr", "it")
DIM = 64


def row(article_id, lang, title, text, **kw):
    return {
        "sr": article_id.partition(":")[0],
        "article_id": article_id,
        "lang": lang,
        "title": title,
        "text": text,
        "kind": kw.get("kind", "article"),
        "is_repealed": kw.get("repealed", False),
        "is_historical": kw.get("historical", False),
    }


def make_articles() -> pd.DataFrame:
    """Three eligible articles in three languages plus ineligible rows (repealed, historical,
    transitional, empty). Rows are deliberately shuffled."""
    rows = []
    topics = {
        "220:1": {
            "de": (
                "Mietzins",
                "1 Der Mieter zahlt den Mietzins.\n2 Die Kuendigung ist schriftlich.",
            ),
            "fr": ("Loyer", "1 Le locataire paie le loyer.\n2 Le conge est donne par ecrit."),
            "it": ("Pigione", "1 L'inquilino paga la pigione.\n2 La disdetta e scritta."),
        },
        "220:2": {
            "de": ("Kaufvertrag", "Der Kaeufer zahlt den Kaufpreis fuer die Ware."),
            "fr": ("Vente", "L'acheteur paie le prix de vente pour la marchandise."),
            "it": ("Vendita", "Il compratore paga il prezzo di vendita per la merce."),
        },
        "210:7": {
            "de": ("Ehe", "Die Ehe wird vor dem Zivilstandsamt geschlossen."),
            "fr": ("Mariage", "Le mariage est celebre devant l'officier de l'etat civil."),
            "it": (
                "Matrimonio",
                "Il matrimonio e celebrato davanti all'ufficiale dello stato civile.",
            ),
        },
    }
    for aid, per_lang in topics.items():
        for lang, (title, text) in per_lang.items():
            rows.append(row(aid, lang, title, text))
    for lang in LANGS:
        rows.append(row("220:3", lang, "x", "", repealed=True))
        rows.append(row("220:4", lang, "alt", "Frueherer Wortlaut", historical=True))
        rows.append(row("220:u1/1", lang, "Uebergang", "Uebergangsrecht", kind="transitional"))
        rows.append(row("220:5", lang, "leer", "   "))
    df = pd.DataFrame(rows)
    return df.sample(frac=1.0, random_state=0).reset_index(drop=True)


class FakeEmbedder:
    """Hashed bag of words, L2-normalised. Counts the texts it was asked to embed."""

    model_id = "fake/bow-64"
    dim = DIM

    def __init__(self) -> None:
        self.passage_calls = 0
        self.query_calls = 0

    def _vec(self, text: str) -> np.ndarray:
        v = np.zeros(DIM, dtype=np.float32)
        for tok in re.findall(r"\w+", text.lower()):
            h = int.from_bytes(hashlib.md5(tok.encode()).digest()[:4], "little")
            v[h % DIM] += 1.0
        n = np.linalg.norm(v)
        return v / n if n else v

    def embed_passages(self, texts):
        self.passage_calls += len(texts)
        return np.stack([self._vec(t) for t in texts]) if texts else np.zeros((0, DIM), np.float32)

    def embed_queries(self, texts):
        self.query_calls += len(texts)
        return np.stack([self._vec(t) for t in texts]) if texts else np.zeros((0, DIM), np.float32)
