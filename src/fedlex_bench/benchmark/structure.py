"""Where an article sits in its act, read from the raw Akoma Ntoso XML.

A brief shows the writer the act name and the position of the article (division, title, chapter,
section headings) in the language of the text, because the article alone often lacks the context
that tells what it is about (the same words occur in the GmbH and in the cooperative title of the
Code of Obligations). The position comes from the ancestors of the ``<article>`` element in the
raw file; ``articles.parquet`` is not touched. Raw files are checked against the sha256 of the
tracked manifest before they are read.

Nothing is invented: an ancestor without a heading is marked ``[heading not available]``; an
article that sits in no division, or an act without a name for the language, gives ``None`` and
the brief says "not available".
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path

from lxml import etree

from fedlex_bench.ingest import (
    NS,
    RAW_DIR,
    RawDataError,
    _article_number,
    _local,
    _read_root,
    file_path,
    load_manifest,
    text_of,
)

STRUCTURAL = {"book", "part", "division", "title", "chapter", "section", "subsection", "level"}
FEDLEX_ROLE = "{http://fedlex.admin.ch/}role"
XML_LANG = "{http://www.w3.org/XML/1998/namespace}lang"
MISSING_HEADING = "[heading not available]"


@dataclass
class Structure:
    """Structure of one act in one language."""

    act_name: str | None = None
    paths: dict[str, list[str]] = field(default_factory=dict)  # article id -> labels


def _label(element: etree._Element) -> str:
    num = element.find("a:num", NS)
    heading = element.find("a:heading", NS)
    num_text = " ".join(text_of(num).split()) if num is not None else ""
    heading_text = " ".join(text_of(heading).split()) if heading is not None else ""
    if not heading_text:
        return f"{num_text or _local(element)} {MISSING_HEADING}".strip()
    return f"{num_text} {heading_text}".strip()


def _is_structural(element: etree._Element) -> bool:
    name = _local(element)
    if name not in STRUCTURAL:
        return False
    return not (name == "level" and element.get(FEDLEX_ROLE) == "marginal")


def build_structure(root: etree._Element, sr: str, lang: str) -> Structure:
    """Act name and the structural path of every main-text article of a parsed act."""
    name = None
    for el in root.iterfind(".//a:meta//a:FRBRname", NS):
        if el.get(XML_LANG) == lang and el.get("value"):
            name = el.get("value")
            break
    paths: dict[str, list[str]] = {}
    for art in root.iterfind(".//a:article", NS):
        if (art.get("eId") or "").startswith("disp_"):
            continue  # transitional blocks have no division path and are never gold
        labels = [_label(anc) for anc in reversed(list(art.iterancestors())) if _is_structural(anc)]
        for number in _article_number(art).split(","):
            paths[f"{sr}:{number}"] = labels
    return Structure(act_name=name, paths=paths)


class RawStructure:
    """Structure provider over the raw files named in the manifest (loaded lazily, checked)."""

    def __init__(self, manifest: dict | None = None, raw_dir: Path = RAW_DIR) -> None:
        self.manifest = load_manifest() if manifest is None else manifest
        self.raw_dir = raw_dir
        self._cache: dict[tuple[str, str], Structure | None] = {}

    def _structure(self, sr: str, lang: str) -> Structure | None:
        key = (sr, lang)
        if key not in self._cache:
            entry = self.manifest.get(sr)
            if entry is None or lang not in entry.get("files", {}):
                self._cache[key] = None
            else:
                path = file_path(entry, lang, self.raw_dir)
                data = path.read_bytes()
                expected = entry["files"][lang].get("sha256")
                if expected and hashlib.sha256(data).hexdigest() != expected:
                    raise RawDataError(
                        f"SR {sr}/{lang}: sha256 of {path.name} differs from the manifest; "
                        "refusing to read the structure from it"
                    )
                self._cache[key] = build_structure(_read_root(data), sr, lang)
        return self._cache[key]

    def act_name(self, sr: str, lang: str) -> str | None:
        s = self._structure(sr, lang)
        return s.act_name if s else None

    def path(self, article_id: str, lang: str) -> list[str] | None:
        """Structural labels of the article, or ``None`` when not available (unknown article, no
        division above it, act not in the manifest)."""
        s = self._structure(article_id.partition(":")[0], lang)
        if s is None:
            return None
        labels = s.paths.get(article_id)
        return labels or None
