"""Fedlex ingestion: SPARQL discovery, raw download, Akoma Ntoso parsing.

Pipeline (all steps idempotent, inputs sorted, counts logged):

1. ``discover``  : SR number -> ELI consolidation abstract -> latest consolidation applicable on
                   ``as_of`` -> per-language expression -> XML manifest (filestore URL).
2. ``download``  : fetch the XML files to
                   ``data/raw/<sr>/<lang>_<consolidation_date>_r<revision>.xml`` (the filestore
                   revision is part of the name, so a refreshed URL can never be paired with old
                   content). Existing files are never re-downloaded. Discovery results, URLs
                   and the sha256 of every raw file are recorded in the *tracked*
                   ``data/manifest.json``; the sha256 is verified on every load and cached files
                   that are not valid Akoma Ntoso XML (e.g. an HTML error page) are rejected.
3. ``parse_akn`` : Akoma Ntoso XML -> list of :class:`ParsedArticle`.

Run ``uv run python -m fedlex_bench.ingest`` to execute discovery + download.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import time
import unicodedata
from dataclasses import asdict, dataclass
from pathlib import Path

import httpx
from lxml import etree

log = logging.getLogger("fedlex_bench.ingest")

SPARQL_ENDPOINT = "https://fedlex.data.admin.ch/sparqlendpoint"
DEFAULT_ACTS: tuple[str, ...] = (
    "210",  # Civil Code
    "220",  # Code of Obligations
    "221.214.1",  # Consumer Credit Act
    "221.229.1",  # Insurance Contract Act
    "235.1",  # Data Protection Act
    "241",  # Unfair Competition Act
    "272",  # Civil Procedure Code
    "281.1",  # Debt Enforcement and Bankruptcy Act
)
LANGS: tuple[str, ...] = ("de", "fr", "it")
# Fixed snapshot date so that a re-run reproduces the same consolidation (not "today").
DEFAULT_AS_OF = "2026-10-15"

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
MANIFEST_NAME = "manifest.json"
MANIFEST_PATH = PROJECT_ROOT / "data" / MANIFEST_NAME  # tracked in git (data/raw/* is ignored)

AKN_NS = "http://docs.oasis-open.org/legaldocml/ns/akn/3.0"
NS = {"a": AKN_NS}

JOLUX = "http://data.legilux.public.lu/resource/ontology/jolux#"


# --------------------------------------------------------------------------------------
# SPARQL discovery
# --------------------------------------------------------------------------------------


def sparql_select(query: str, client: httpx.Client | None = None, retries: int = 3) -> list[dict]:
    """Run a SELECT query and return rows as ``{var: value}`` dicts."""
    own = client is None
    client = client or httpx.Client(timeout=120)
    try:
        last: Exception | None = None
        for attempt in range(retries):
            try:
                resp = client.post(
                    SPARQL_ENDPOINT,
                    data={"query": query},
                    headers={"Accept": "application/sparql-results+json"},
                )
                resp.raise_for_status()
                bindings = resp.json()["results"]["bindings"]
                return [{k: v["value"] for k, v in b.items()} for b in bindings]
            except (httpx.HTTPError, ValueError) as exc:  # pragma: no cover - network
                last = exc
                time.sleep(2**attempt)
        raise RuntimeError(f"SPARQL query failed after {retries} attempts: {last}")
    finally:
        if own:
            client.close()


def _q_abstract(sr: str) -> str:
    # The SR number lives on the legal-taxonomy entry (skos:notation). `historicalLegalId` is not
    # reliable: the revised Data Protection Act (eli/cc/2022/491) does not carry it. Several works
    # can share an SR number over time (old DPA 1992 repealed 2023-08-31, new DPA from 2023-09-01).
    return f"""
PREFIX jolux: <{JOLUX}>
PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
SELECT DISTINCT ?work WHERE {{
  ?work a jolux:ConsolidationAbstract ; jolux:classifiedByTaxonomyEntry ?tax .
  ?tax skos:notation ?n .
  FILTER(STR(?n) = "{sr}")
}} ORDER BY ?work
"""


def _q_consolidations(work: str) -> str:
    return f"""
PREFIX jolux: <{JOLUX}>
SELECT DISTINCT ?cons ?start ?end WHERE {{
  ?cons jolux:isMemberOf <{work}> ; jolux:dateApplicability ?start .
  OPTIONAL {{ ?cons jolux:dateEndApplicability ?end }}
}} ORDER BY ?start
"""


def _q_manifestations(cons: str) -> str:
    return f"""
PREFIX jolux: <{JOLUX}>
SELECT DISTINCT ?expr ?lang ?fmt ?url WHERE {{
  <{cons}> jolux:isRealizedBy ?expr .
  ?expr jolux:isEmbodiedBy ?man .
  ?man jolux:userFormat ?fmtUri ;
       jolux:isExemplifiedBy ?url .
  BIND(STRAFTER(STR(?expr), CONCAT(STR(<{cons}>), "/")) AS ?lang)
  BIND(STRAFTER(STR(?fmtUri), "user-format/") AS ?fmt)
}} ORDER BY ?lang ?fmt ?url
"""


def pick_consolidation(rows: list[dict], as_of: str) -> dict:
    """Pick the consolidation applicable on ``as_of``: latest start <= as_of whose end (if any)
    is >= as_of. ISO dates compare lexicographically."""
    candidates = [
        r for r in rows if r["start"] <= as_of and (not r.get("end") or r["end"] >= as_of)
    ]
    if not candidates:
        raise LookupError(f"no consolidation applicable on {as_of}")
    return sorted(candidates, key=lambda r: (r["start"], r["cons"]))[-1]


_REVISION_RE = re.compile(r"-(\d+)\.[A-Za-z0-9]+$")


def revision_of(url: str) -> int:
    """Filestore revision: the trailing integer of the file name (``...-xml-3.xml`` -> 3);
    0 when the URL carries none."""
    m = _REVISION_RE.search(url)
    return int(m.group(1)) if m else 0


def revision_tag(url: str) -> str:
    """Deterministic file-name tag for a URL: ``r<revision>`` or, without a revision in the URL,
    ``h<sha1 of the URL>[:8]``."""
    m = _REVISION_RE.search(url)
    return f"r{int(m.group(1))}" if m else "h" + hashlib.sha1(url.encode("utf-8")).hexdigest()[:8]


def pick_manifestations(rows: list[dict], langs: tuple[str, ...] = LANGS) -> dict[str, dict]:
    """Per language choose XML (Akoma Ntoso) if available, else HTML. The highest filestore
    revision wins (compared as an integer, so ``-10`` beats ``-9``) when several exemplars exist."""
    out: dict[str, dict] = {}
    for lang in langs:
        by_fmt: dict[str, list[str]] = {}
        for r in rows:
            if r["lang"] == lang:
                by_fmt.setdefault(r["fmt"], []).append(r["url"])
        for fmt in ("xml", "html"):
            if fmt in by_fmt:
                url = max(by_fmt[fmt], key=lambda u: (revision_of(u), u))
                out[lang] = {"format": fmt, "url": url}
                break
    return out


def discover(sr: str, as_of: str, client: httpx.Client | None = None) -> dict:
    """SPARQL discovery for one act. Returns a JSON-serialisable record."""
    works = sparql_select(_q_abstract(sr), client)
    if not works:
        raise LookupError(f"SR {sr}: no consolidation abstract found")
    rows: list[dict] = []
    for w in works:  # several works may share one SR number over time
        found = sparql_select(_q_consolidations(w["work"]), client)
        rows.extend({**r, "work": w["work"]} for r in found)
    cons = pick_consolidation(rows, as_of)
    work = cons["work"]
    mans = pick_manifestations(sparql_select(_q_manifestations(cons["cons"]), client))
    return {
        "sr": sr,
        "work": work,
        "consolidation": cons["cons"],
        "consolidation_date": cons["start"],
        "consolidation_end": cons.get("end"),
        "as_of": as_of,
        "files": {lang: mans[lang] for lang in sorted(mans)},
    }


# --------------------------------------------------------------------------------------
# Download (idempotent, checksummed)
# --------------------------------------------------------------------------------------


class RawDataError(RuntimeError):
    """A raw file is corrupt, tampered with, or not Akoma Ntoso XML."""


def raw_path(
    sr: str,
    lang: str,
    consolidation_date: str,
    fmt: str,
    raw_dir: Path = RAW_DIR,
    url: str | None = None,
) -> Path:
    """``<raw_dir>/<sr>/<lang>_<consolidation_date>_<revision tag>.<fmt>``.

    The revision tag derives from the filestore URL (see :func:`revision_tag`), so a new URL
    can never be paired with content downloaded from an older one.
    """
    tag = f"_{revision_tag(url)}" if url else ""
    return raw_dir / sr / f"{lang}_{consolidation_date}{tag}.{fmt}"


def file_path(entry: dict, lang: str, raw_dir: Path = RAW_DIR) -> Path:
    """Raw file for ``lang`` of a manifest ``entry``."""
    f = entry["files"][lang]
    return raw_path(entry["sr"], lang, entry["consolidation_date"], f["format"], raw_dir, f["url"])


def sha256_of(path_or_bytes: Path | bytes) -> str:
    data = path_or_bytes if isinstance(path_or_bytes, bytes) else path_or_bytes.read_bytes()
    return hashlib.sha256(data).hexdigest()


def validate_akn(content: bytes, what: str = "file") -> None:
    """Raise :class:`RawDataError` unless ``content`` is an Akoma Ntoso document with an ``act``
    (rejects HTML error pages, truncated downloads and other XML dialects)."""
    try:
        root = etree.fromstring(
            content, parser=etree.XMLParser(resolve_entities=False, huge_tree=True, no_network=True)
        )
    except etree.XMLSyntaxError as exc:
        raise RawDataError(f"{what}: not valid XML ({exc})") from exc
    if root.tag != f"{{{AKN_NS}}}akomaNtoso" or root.find("a:act", NS) is None:
        raise RawDataError(f"{what}: not an Akoma Ntoso <act> document (root {root.tag!r})")


def load_manifest(manifest_path: Path = MANIFEST_PATH) -> dict:
    if manifest_path.exists():
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    return {}


def save_manifest(manifest: dict, manifest_path: Path = MANIFEST_PATH) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    manifest_path.write_text(text, encoding="utf-8", newline="\n")


def verify_cached(path: Path, f: dict, what: str) -> bool:
    """Check a cached raw file. True: usable. False: unusable and removed (will be re-fetched).
    Raises :class:`RawDataError` when the bytes differ from the recorded sha256."""
    if not path.exists():
        return False
    content = path.read_bytes()
    expected = f.get("sha256")
    if expected and sha256_of(content) != expected:
        raise RawDataError(
            f"{what}: sha256 mismatch for {path.name} (manifest {expected[:12]}..., "
            f"file {sha256_of(content)[:12]}...). Delete the file or re-run with --refresh."
        )
    try:
        validate_akn(content, what)
    except RawDataError as exc:
        log.warning("%s: discarding cached file %s (%s)", what, path.name, exc)
        path.unlink()
        return False
    if not expected:
        f["sha256"] = sha256_of(content)
        f["bytes"] = len(content)
    return True


def ensure_raw(
    acts: tuple[str, ...] = DEFAULT_ACTS,
    as_of: str = DEFAULT_AS_OF,
    raw_dir: Path = RAW_DIR,
    refresh: bool = False,
    manifest_path: Path | None = None,
) -> dict:
    """Discover (recorded in the tracked manifest) and download raw files; returns the manifest.

    Network is only touched when a manifest entry is missing/stale or a file is missing/invalid.
    Every raw file is checked against its recorded sha256 and must be valid Akoma Ntoso XML.
    """
    manifest_path = manifest_path or raw_dir.parent / MANIFEST_NAME
    manifest = load_manifest(manifest_path)
    clients: list[httpx.Client] = []  # created lazily: a fully cached run is offline

    def get_client() -> httpx.Client:
        if not clients:
            clients.append(httpx.Client(timeout=120, follow_redirects=True))
        return clients[0]

    try:
        for sr in sorted(acts):
            entry = manifest.get(sr)
            if refresh or entry is None or entry.get("as_of") != as_of:
                log.info("SR %s: SPARQL discovery (as_of=%s)", sr, as_of)
                old = (entry or {}).get("files", {})
                entry = discover(sr, as_of, get_client())
                for lang, f in entry["files"].items():  # same URL -> same content: keep the sha
                    if lang in old and old[lang].get("url") == f["url"] and "sha256" in old[lang]:
                        f["sha256"], f["bytes"] = old[lang]["sha256"], old[lang]["bytes"]
                manifest[sr] = entry
            missing = [lang for lang in LANGS if lang not in entry["files"]]
            if missing:
                raise LookupError(f"SR {sr}: no file for language(s) {missing}")
            for lang in sorted(entry["files"]):
                f = entry["files"][lang]
                path = file_path(entry, lang, raw_dir)
                what = f"SR {sr}/{lang}"
                if f["format"] == "xml" and verify_cached(path, f, what):
                    log.info("%s: cached %s (sha256 ok)", what, path.name)
                    continue
                log.info("%s: downloading %s", what, f["url"])
                resp = get_client().get(f["url"])
                resp.raise_for_status()
                if f["format"] == "xml":
                    validate_akn(resp.content, what)
                digest = sha256_of(resp.content)
                if f.get("sha256") and f["sha256"] != digest:
                    raise RawDataError(f"{what}: downloaded content differs from recorded sha256")
                path.parent.mkdir(parents=True, exist_ok=True)
                tmp = path.with_suffix(path.suffix + ".part")
                tmp.write_bytes(resp.content)
                tmp.replace(path)
                f["sha256"], f["bytes"] = digest, len(resp.content)
                log.info("%s: %d bytes, sha256 %s", what, len(resp.content), digest[:12])
        save_manifest(manifest, manifest_path)
    finally:
        for c in clients:
            c.close()
    return manifest


# --------------------------------------------------------------------------------------
# Parsing (Akoma Ntoso)
# --------------------------------------------------------------------------------------

FEDLEX_NS = "http://fedlex.admin.ch/"


@dataclass(frozen=True)
class ParsedArticle:
    """One chunk-to-be.

    ``number`` is the alignment-key part of the id (``<SR>:<number>``). Transitional/final
    provisions live in their own namespace: ``u<N>/<number>`` for an article of block ``N`` and
    ``u<N>`` for a block without ``<article>`` elements that is chunked as a whole.
    **``u<N>`` is positional per snapshot**: ``N`` is the 1-based index of the ``disp_uN``
    block in that consolidation, so it can shift when a later consolidation adds or removes a
    block. Compare ids across snapshots through ``block_heading``, not through ``N``.
    """

    number: str  # normalised article number, e.g. "257d"; transitional: "u12/2-4"
    kind: str  # "article" (main body) | "transitional" (disp_uN blocks: Schluss-/Uebergangsbest.)
    eid: str  # raw eId, e.g. "art_257_d" or "disp_u12/art_2_4"
    title: str  # marginal note, may be empty
    text: str  # body text without footnotes; empty for repealed articles
    is_repealed: bool  # no text in force: repealed, obsolete or elided
    repeal_note: str  # footnote text that declared the repeal ("" when not repealed)
    is_historical: bool = False  # block quoting superseded wording (ZGB "fruehere Bestimmungen")
    block_heading: str = ""  # heading of the enclosing disp_ block ("" for body articles)
    displayed_number: str = ""  # number as printed in <num>, normalised (cross-check of eId)
    n_paragraphs: int = 0  # structural fingerprint for cross-language comparison
    n_items: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


ELISION = "…"  # Fedlex marks omitted/repealed text with a bare ellipsis

# A line that is nothing but a repeal marker (DE/FR/IT).
_REPEAL_WORD = (
    r"(?:aufgehoben|abgeschafft|abrog[eéè]e?s?|abrogat[oaie]|abolit[oaie]|"
    r"(?:ausser|hors de) kraft|\[aufgehoben\])"
)
_REPEALED_STUB_RE = re.compile(
    rf"^\W*(?:\d+[a-z]*\W+)?(?:{_REPEAL_WORD}|{ELISION}|\.\.\.)\W*$", re.IGNORECASE
)
# A footnote that mentions a repeal anywhere. Body-less articles usually carry a history such as
# "Eingefuegt durch ... Aufgehoben durch ..." ("Gegenstandslos" = obsolete, treated alike).
_REPEAL_NOTE_RE = re.compile(
    rf"{_REPEAL_WORD}|gegenstandslos|sans objet|priv[oa] d(?:i |['’])\s*oggetto|caduc",
    re.IGNORECASE,
)
# Blocks that merely quote superseded wording (ZGB: "Wortlaut der früheren Bestimmungen des
# sechsten Titels" / "Teneur des anciennes dispositions" / "Testo delle disposizioni previgenti").
_HISTORICAL_RE = re.compile(
    r"früheren\s+bestimmungen|anciennes\s+dispositions|disposizioni\s+previgenti"
    r"|precedenti\s+disposizioni",
    re.IGNORECASE,
)

# Latin multiplicative suffixes used in Swiss article numbering, in ordinal order
# (bis = 2nd, ter = 3rd, quater = 4th, ...).
_SUFFIX_WORDS = (
    "bis ter quater quinquies sexies septies octies novies decies undecies duodecies "
    "terdecies quaterdecies quindecies sexdecies septdecies octodecies novodecies vicies"
).split()
LATIN_ORDINAL = {w: i + 2 for i, w in enumerate(_SUFFIX_WORDS)}
_SUFFIX_RE = re.compile("(?:" + "|".join(sorted(_SUFFIX_WORDS, key=len, reverse=True)) + r"|[a-z])")

# Connectors between two article numbers in "Art. 50 und 51" / "135 à 149" / "135 a 149".
_RANGE_WORDS = {"und", "et", "e", "à", "a", "au", "bis"}

_SPACE_CHARS = "".join(chr(c) for c in (0xA0, 0x202F, 0x2009, 0x2007, 0x2002, 0x2003, 0xFEFF))
_SPACE_RE = re.compile(f"[{_SPACE_CHARS}]")
# hyphen-like characters: U+2010..U+2015 (hyphen, non-breaking hyphen, figure/en/em dash, bar)
# and U+2212 (minus sign)
_DASHES = "".join(chr(c) for c in (*range(0x2010, 0x2016), 0x2212))
_DASH_RE = re.compile(f"\\s*[-{_DASHES}]\\s*")
_WS_RE = re.compile(r"\s+")
_ART_PREFIX_RE = re.compile(r"^\s*(?:art(?:icle|icolo|ikel)?s?\.?|§)\s*", re.IGNORECASE)


def normalize_article_number(raw: str) -> str:
    """Normalise a displayed article number to the id form.

    ``"Art. 257 d"`` -> ``"257d"``, ``"Art. 12bis"`` -> ``"12bis"``, ``"Art. 8 bis"`` ->
    ``"8bis"``, ``"art. 6a bis"`` -> ``"6abis"``, ``"Art. 135-149"`` -> ``"135-149"``,
    ``"Art. 2 4"`` -> ``"2-4"`` (other whitespace-separated parts are joined with ``-``).
    Hyphen, non-breaking hyphen (U+2011), en/em dash and minus all count as ``-``. The result is
    lowercase with no whitespace.
    """
    s = unicodedata.normalize("NFKC", raw or "")
    s = _SPACE_RE.sub(" ", s)
    s = _ART_PREFIX_RE.sub("", s).strip().lower()
    s = _DASH_RE.sub("-", s)
    s = re.sub(r"(?<=[0-9a-z])à(?=\d)", " à ", s)  # glued FR range "663dà663h"
    s = s.rstrip(".").strip()
    toks = s.split()
    out = ""
    for i, tok in enumerate(toks):
        nxt = toks[i + 1] if i + 1 < len(toks) else ""
        if not out:
            out = tok
        elif tok in _RANGE_WORDS and nxt[:1].isdigit():
            out += "-"  # "135 a 149", "50 und 51", "28d a 28f" -> "135-149", ...
        elif _SUFFIX_RE.fullmatch(tok):
            out += tok  # "257 d" -> "257d", "8 bis" -> "8bis"
        elif out.endswith("-"):
            out += tok
        else:
            out += "-" + tok  # "2 4" -> "2-4"
    return out


def number_from_eid(eid: str) -> str:
    """Derive the normalised number from an AKN eId.

    ``art_257_d`` -> ``257d``; ``art_6_bis`` -> ``6bis``; ``art_268_a_bis`` -> ``268abis``;
    ``art_135_149`` (repealed range) -> ``135-149``; ``disp_u12/art_2_4`` -> ``2-4``.
    """
    last = eid.rsplit("/", 1)[-1]
    last = re.sub(r"^art_", "", last)
    out = ""
    for part in last.split("_"):
        out += ("-" + part) if (out and part.isdigit()) else part
    return out.lower()


def _local(el: etree._Element) -> str:
    return etree.QName(el).localname


def _clean(text: str) -> str:
    return _WS_RE.sub(" ", _SPACE_RE.sub(" ", text)).strip()


_BLOCKS = {"paragraph", "item", "p", "listIntroduction", "blockList", "heading"}
# Containers that carry no line break of their own: a node that is their first child is "at the
# start" of the container, so a ``num`` before the container also precedes the node.
_TRANSPARENT = {"content", "blockList", "listIntroduction"}


def _prev_element(node: etree._Element) -> etree._Element | None:
    prev = node.getprevious()
    while prev is not None and not isinstance(prev.tag, str):
        prev = prev.getprevious()
    return prev


def _follows_num(node: etree._Element) -> bool:
    """True when ``node`` starts right after a ``num`` (so the number and text share a line).

    Climbs while the node is the first element child of a transparent container, e.g.
    ``paragraph > num, content > blockList > listIntroduction``: the introduction follows the
    paragraph number even though it is nested two levels below ``content``.
    """
    probe = node
    while True:
        prev = _prev_element(probe)
        if prev is not None:
            return _local(prev) == "num"
        parent = probe.getparent()
        if parent is None or _local(parent) not in _TRANSPARENT:
            return False
        probe = parent


def text_of(el: etree._Element) -> str:
    """Flatten an element to text: footnotes (authorialNote) dropped, block elements on their own
    line, list/paragraph numbers kept on the same line as their text."""
    parts: list[str] = []

    def walk(node: etree._Element) -> None:
        name = _local(node)
        if name == "authorialNote":
            return
        if name in _BLOCKS and not _follows_num(node):
            parts.append("\n")
        if name == "br":
            parts.append("\n")
        if node.text:
            parts.append(node.text)
        for child in node:
            if isinstance(child.tag, str):
                walk(child)
            if child.tail:
                parts.append(child.tail)
        if name == "num":
            parts.append(" ")
        elif name in _BLOCKS:
            parts.append("\n")

    walk(el)
    lines = [_clean(line) for line in "".join(parts).split("\n")]
    return "\n".join(line for line in lines if line)


def _marginal_title(article: etree._Element, has_body: bool = True) -> str:
    """Marginal note of an article: its own ``heading`` if present (only trusted when the article
    has a body; body-less articles can carry a mis-split number there), else the heading of the
    nearest enclosing ``level`` when that level carries ``fedlex:role="marginal"``."""
    own = article.find("a:heading", NS)
    if own is not None and has_body:
        return text_of(own).replace("\n", " ")
    parent = article.getparent()
    if parent is not None and _local(parent) == "level":
        if parent.get(f"{{{FEDLEX_NS}}}role") == "marginal":
            h = parent.find("a:heading", NS)
            if h is not None:
                return text_of(h).replace("\n", " ")
    return ""


def _repeal_note(article: etree._Element) -> str:
    """Text of a footnote attached to the article number that declares a repeal."""
    # Usually in <num>; in a few Italian files the number is split and the note lands in <heading>.
    for tag in ("num", "heading"):
        for holder in article.findall(f"a:{tag}", NS):
            for note in holder.iterfind(".//a:authorialNote", NS):
                t = _clean("".join(note.itertext()))
                if _REPEAL_NOTE_RE.search(t):
                    return t
    return ""


def _is_repealed_stub(text: str) -> bool:
    lines = [ln for ln in text.split("\n") if ln.strip()]
    return bool(lines) and all(_REPEALED_STUB_RE.match(ln) for ln in lines)


_NUMBERED_LINE_RE = re.compile(r"^\d+[a-z]*\s")


def _is_amendment_stub(text: str) -> bool:
    """An amendment-of-another-act block: an unnumbered introduction ending in ``:`` followed by
    Fedlex's elision mark (``Sono modificate:`` + ``...``). The amended wording is not part of
    this act, and some languages keep the introduction while others reduce the block to the
    elision mark alone."""
    lines = [ln for ln in text.split("\n") if ln.strip()]
    return (
        len(lines) >= 2
        and lines[-1].strip() == ELISION
        and lines[-2].rstrip().endswith(":")
        and not any(_NUMBERED_LINE_RE.match(ln) for ln in lines)
    )


def _article_number(article: etree._Element) -> str:
    """Prefer the eId (machine key, identical across languages); fall back to ``num``."""
    eid = article.get("eId") or ""
    if re.search(r"(^|/)art_", eid):
        return number_from_eid(eid)
    num = article.find("a:num", NS)
    return normalize_article_number(text_of(num)) if num is not None else ""


_UNIT_EID_RE = re.compile(r"^disp_(u\d+)$")


def _units(root: etree._Element) -> list[etree._Element]:
    """Top-level transitional/final-provision blocks (``disp_uN``) in document order."""
    return [
        u
        for u in root.xpath("//a:transitional | //a:proviso", namespaces=NS)
        if _UNIT_EID_RE.match(u.get("eId") or "")
    ]


def _unit_heading(unit: etree._Element) -> str:
    h = unit.find("a:heading", NS)
    return text_of(h).replace("\n", " ") if h is not None else ""


def _read_root(xml: bytes | str | Path) -> etree._Element:
    if isinstance(xml, Path):
        xml = xml.read_bytes()
    if isinstance(xml, str):
        xml = xml.encode("utf-8")
    parser = etree.XMLParser(resolve_entities=False, huge_tree=True, no_network=True)
    return etree.fromstring(xml, parser=parser)


def _strip(el: etree._Element, tags: tuple[str, ...]) -> etree._Element:
    clone = etree.fromstring(etree.tostring(el))
    for tag in tags:
        for child in clone.findall(f"a:{tag}", NS):
            clone.remove(child)
    return clone


def parse_akn(xml: bytes | str | Path) -> list[ParsedArticle]:
    """Parse an Akoma Ntoso act into articles, in document order (articles, then block units).

    - Main-body articles get ``kind='article'`` and number e.g. ``257d``.
    - Articles inside ``transitional``/``proviso`` blocks (eId prefix ``disp_uN/``, e.g. the
      Schlusstitel) get ``kind='transitional'`` and number ``uN/<n>`` so that they never collide
      with body articles that carry the same number. ``uN`` is positional per snapshot (see
      :class:`ParsedArticle`).
    - A block without any ``<article>`` but with substantive text is chunked as one
      pseudo-article ``uN`` (``kind='transitional'``).
    - ``is_repealed``: no substantive text in force. True when the article has no body (a
      footnote usually says "aufgehoben / abroge / abrogato" or "gegenstandslos" = obsolete, but
      some files only carry an "introduced by" note), or its
      entire text is such a stub or Fedlex's elision mark, or (transitional only) it is an
      amendment-of-another-act introduction followed by the elision mark. Such articles keep an
      empty text.
    - ``is_historical``: the enclosing block quotes superseded wording (see ``_HISTORICAL_RE``).
    """
    root = _read_root(xml)
    out: list[ParsedArticle] = []
    for art in root.iterfind(".//a:article", NS):
        eid = art.get("eId") or ""
        number = _article_number(art)
        kind, block_heading = "article", ""
        m = re.match(r"^disp_(u\d+)/", eid)
        if m:
            kind = "transitional"
            number = f"{m.group(1)}/{number}"
            units = art.xpath("ancestor::a:transitional | ancestor::a:proviso", namespaces=NS)
            block_heading = _unit_heading(units[0]) if units else ""
        text = text_of(_strip(art, ("num", "heading")))
        note = _repeal_note(art)
        # An article with no body at all has no text in force, even when its footnote does not
        # say so (IT VVG Art. 48-49 only carries "Introdotto dal ...").
        repealed = (
            not text
            or _is_repealed_stub(text)
            or (kind == "transitional" and _is_amendment_stub(text))
        )
        if repealed and not note:
            note = text
        num = art.find("a:num", NS)
        out.append(
            ParsedArticle(
                number=number,
                kind=kind,
                eid=eid,
                title=_marginal_title(art, has_body=bool(text)),
                text="" if repealed else text,
                is_repealed=repealed,
                repeal_note=note if repealed else "",
                is_historical=kind == "transitional" and bool(_HISTORICAL_RE.search(block_heading)),
                block_heading=block_heading,
                displayed_number=normalize_article_number(text_of(num)) if num is not None else "",
                n_paragraphs=len(art.findall(".//a:paragraph", NS)),
                n_items=len(art.findall(".//a:item", NS)),
            )
        )
    for unit in _units(root):
        if unit.find(".//a:article", NS) is not None:
            continue
        heading = _unit_heading(unit)
        text = text_of(_strip(unit, ("num", "heading")))
        if not text:
            continue  # nothing to chunk; reported through unarticled_units()
        repealed = _is_repealed_stub(text) or _is_amendment_stub(text)
        match = _UNIT_EID_RE.match(unit.get("eId") or "")
        assert match is not None  # _units() only yields matching eIds
        out.append(
            ParsedArticle(
                number=match.group(1),
                kind="transitional",
                eid=unit.get("eId") or "",
                title=heading,
                text="" if repealed else text,
                is_repealed=repealed,
                repeal_note=text if repealed else "",
                is_historical=bool(_HISTORICAL_RE.search(heading)),
                block_heading=heading,
                displayed_number="",
                n_paragraphs=len(unit.findall(".//a:paragraph", NS)),
                n_items=len(unit.findall(".//a:item", NS)),
            )
        )
    return out


def unarticled_units(xml: bytes | str | Path) -> list[tuple[str, str, int]]:
    """Transitional/final-provision blocks without any ``<article>`` (plain paragraphs only).

    Returns ``[(eId, heading, n_chars), ...]`` in document order. Blocks with text are chunked
    as pseudo-articles by :func:`parse_akn`; this lists them anyway so that reports can show
    what was chunked and what (``n_chars == 0``) had nothing to chunk.
    """
    out: list[tuple[str, str, int]] = []
    for unit in _units(_read_root(xml)):
        if unit.find(".//a:article", NS) is None:
            text = text_of(_strip(unit, ("num", "heading")))
            out.append((unit.get("eId") or "", _unit_heading(unit), len(text)))
    return out


def unchunked_components(xml: bytes | str | Path) -> list[tuple[str, int]]:
    """Annexes (``<components>/<component>/<doc>``) are not chunked. Returns
    ``[(name, n_chars)]`` so that reports can show how much text is left out."""
    out: list[tuple[str, int]] = []
    for i, comp in enumerate(_read_root(xml).iterfind(".//a:components/a:component", NS), 1):
        doc = comp.find("a:doc", NS)
        body = _strip(doc if doc is not None else comp, ("meta",))
        name = (doc.get("name") if doc is not None else None) or "component"
        out.append((f"{name} #{i}", len(text_of(body))))
    return out


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--acts", nargs="*", default=list(DEFAULT_ACTS), help="SR numbers")
    ap.add_argument("--as-of", default=DEFAULT_AS_OF, help="consolidation applicable on this date")
    ap.add_argument("--refresh", action="store_true", help="redo SPARQL discovery")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    manifest = ensure_raw(tuple(args.acts), args.as_of, refresh=args.refresh)
    for sr in sorted(manifest):
        e = manifest[sr]
        log.info(
            "SR %s: consolidation %s (%s .. %s), languages=%s",
            sr,
            e["consolidation_date"],
            e["consolidation_date"],
            e.get("consolidation_end") or "open",
            ",".join(sorted(e["files"])),
        )


if __name__ == "__main__":
    main()
