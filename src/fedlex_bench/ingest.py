"""Fedlex ingestion: SPARQL discovery, raw download, Akoma Ntoso parsing.

Pipeline (all steps idempotent, inputs sorted, counts logged):

1. ``discover``  : SR number -> ELI consolidation abstract -> latest consolidation applicable on
                   ``as_of`` -> per-language expression -> XML manifest (filestore URL).
2. ``download``  : fetch the XML files to ``data/raw/<sr>/<lang>_<consolidation_date>.xml``;
                   existing files are never re-downloaded. Discovery results are cached in
                   ``data/raw/manifest.json``.
3. ``parse_akn`` : Akoma Ntoso XML -> list of :class:`ParsedArticle`.

Run ``uv run python -m fedlex_bench.ingest`` to execute discovery + download.
"""

from __future__ import annotations

import argparse
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
DEFAULT_ACTS: tuple[str, ...] = ("210", "220", "235.1")
LANGS: tuple[str, ...] = ("de", "fr", "it")
# Fixed snapshot date so that a re-run reproduces the same consolidation (not "today").
DEFAULT_AS_OF = "2026-10-15"

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
MANIFEST_NAME = "manifest.json"

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


def pick_manifestations(rows: list[dict], langs: tuple[str, ...] = LANGS) -> dict[str, dict]:
    """Per language choose XML (Akoma Ntoso) if available, else HTML. Highest filestore revision
    (lexicographically last URL) wins when several exemplars exist."""
    out: dict[str, dict] = {}
    for lang in langs:
        by_fmt: dict[str, list[str]] = {}
        for r in rows:
            if r["lang"] == lang:
                by_fmt.setdefault(r["fmt"], []).append(r["url"])
        for fmt in ("xml", "html"):
            if fmt in by_fmt:
                out[lang] = {"format": fmt, "url": sorted(by_fmt[fmt])[-1]}
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
# Download (idempotent)
# --------------------------------------------------------------------------------------


def raw_path(
    sr: str, lang: str, consolidation_date: str, fmt: str, raw_dir: Path = RAW_DIR
) -> Path:
    return raw_dir / sr / f"{lang}_{consolidation_date}.{fmt}"


def load_manifest(raw_dir: Path = RAW_DIR) -> dict:
    path = raw_dir / MANIFEST_NAME
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def save_manifest(manifest: dict, raw_dir: Path = RAW_DIR) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    text = json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    (raw_dir / MANIFEST_NAME).write_text(text, encoding="utf-8")


def ensure_raw(
    acts: tuple[str, ...] = DEFAULT_ACTS,
    as_of: str = DEFAULT_AS_OF,
    raw_dir: Path = RAW_DIR,
    refresh: bool = False,
) -> dict:
    """Discover (cached in manifest.json) and download raw files. Returns the manifest.

    Network is only touched when a manifest entry is missing/stale or a file is missing.
    """
    manifest = load_manifest(raw_dir)
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
                entry = discover(sr, as_of, get_client())
                manifest[sr] = entry
            missing = [lang for lang in LANGS if lang not in entry["files"]]
            if missing:
                raise LookupError(f"SR {sr}: no file for language(s) {missing}")
            for lang in sorted(entry["files"]):
                f = entry["files"][lang]
                path = raw_path(sr, lang, entry["consolidation_date"], f["format"], raw_dir)
                if path.exists() and path.stat().st_size > 0:
                    log.info("SR %s/%s: cached %s", sr, lang, path.name)
                    continue
                log.info("SR %s/%s: downloading %s", sr, lang, f["url"])
                resp = get_client().get(f["url"])
                resp.raise_for_status()
                path.parent.mkdir(parents=True, exist_ok=True)
                tmp = path.with_suffix(path.suffix + ".part")
                tmp.write_bytes(resp.content)
                tmp.replace(path)
                log.info("SR %s/%s: %d bytes", sr, lang, len(resp.content))
        save_manifest(manifest, raw_dir)
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
    number: str  # normalised article number, e.g. "257d"; transitional: "u12/2-4"
    kind: str  # "article" (main body) | "transitional" (disp_uN blocks: Schluss-/Übergangsbest.)
    eid: str  # raw eId, e.g. "art_257_d" or "disp_u12/art_2_4"
    title: str  # marginal note, may be empty
    text: str  # body text without footnotes; empty for repealed articles
    is_repealed: bool
    repeal_note: str  # footnote text that declared the repeal ("" when not repealed)

    def to_dict(self) -> dict:
        return asdict(self)


# A line that is nothing but a repeal marker (DE/FR/IT).
_REPEAL_WORD = (
    r"(?:aufgehoben|abgeschafft|abrog[eéè]e?s?|abrogat[oaie]|abolit[oaie]|"
    r"(?:ausser|hors de) kraft|\[aufgehoben\])"
)
_REPEALED_STUB_RE = re.compile(
    rf"^\W*(?:\d+[a-z]*\W+)?(?:{_REPEAL_WORD}|…|\.\.\.)\W*$", re.IGNORECASE
)
# A footnote that mentions a repeal anywhere. Body-less articles usually carry a history such as
# "Eingefügt durch ... Aufgehoben durch ..." ("Gegenstandslos" = obsolete, treated alike).
_REPEAL_NOTE_RE = re.compile(
    rf"{_REPEAL_WORD}|gegenstandslos|sans objet|priv[oa] d(?:i |[’'])\s*oggetto|caduc",
    re.IGNORECASE,
)

# Latin multiplicative suffixes used in Swiss article numbering (257bis, 12quater, ...).
_SUFFIX_WORDS = (
    "bis ter quater quinquies sexies septies octies novies decies undecies duodecies "
    "terdecies quaterdecies quindecies sexdecies septdecies octodecies novodecies vicies"
).split()
_SUFFIX_RE = re.compile("(?:" + "|".join(sorted(_SUFFIX_WORDS, key=len, reverse=True)) + r"|[a-z])")

# Connectors between two article numbers in "Art. 50 und 51" / "135 à 149" / "135 a 149".
_RANGE_WORDS = {"und", "et", "e", "à", "a", "au", "bis"}

_SPACE_CHARS = "".join(chr(c) for c in (0xA0, 0x202F, 0x2009, 0x2007, 0x2002, 0x2003, 0xFEFF))
_SPACE_RE = re.compile(f"[{_SPACE_CHARS}]")
_WS_RE = re.compile(r"\s+")
_ART_PREFIX_RE = re.compile(r"^\s*(?:art(?:icle|icolo|ikel)?s?\.?|§)\s*", re.IGNORECASE)


def normalize_article_number(raw: str) -> str:
    """Normalise a displayed article number to the id form.

    ``"Art. 257 d"`` -> ``"257d"``, ``"Art. 12bis"`` -> ``"12bis"``, ``"Art. 8 bis"`` ->
    ``"8bis"``, ``"art. 6a bis"`` -> ``"6abis"``, ``"Art. 135-149"`` -> ``"135-149"``,
    ``"Art. 2 4"`` -> ``"2-4"`` (other whitespace-separated parts are joined with ``-``).
    The result is lowercase with no whitespace.
    """
    s = unicodedata.normalize("NFKC", raw or "")
    s = _SPACE_RE.sub(" ", s)
    s = _ART_PREFIX_RE.sub("", s).strip().lower()
    s = re.sub(r"\s*[-–—]\s*", "-", s)  # hyphen / en dash / em dash ranges
    s = s.rstrip(".").strip()
    toks = s.split()
    out = ""
    for i, tok in enumerate(toks):
        nxt = toks[i + 1] if i + 1 < len(toks) else ""
        if not out:
            out = tok
        elif tok in _RANGE_WORDS and nxt[:1].isdigit():
            out += "-"  # "135 a 149", "50 und 51", "28d à 28f" -> "135-149", ...
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


def _follows_num(node: etree._Element) -> bool:
    """True when ``node`` directly follows a ``num`` (so the number and text share a line)."""
    probe = node
    parent = node.getparent()
    if parent is not None and node.getprevious() is None and _local(parent) == "content":
        probe = parent
    prev = probe.getprevious()
    while prev is not None and not isinstance(prev.tag, str):
        prev = prev.getprevious()
    return prev is not None and _local(prev) == "num"


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


def _article_number(article: etree._Element) -> str:
    """Prefer the eId (machine key, identical across languages); fall back to ``num``."""
    eid = article.get("eId") or ""
    if re.search(r"(^|/)art_", eid):
        return number_from_eid(eid)
    num = article.find("a:num", NS)
    return normalize_article_number(text_of(num)) if num is not None else ""


def parse_akn(xml: bytes | str | Path) -> list[ParsedArticle]:
    """Parse an Akoma Ntoso act into articles, in document order.

    - Main-body articles get ``kind='article'`` and number e.g. ``257d``.
    - Articles inside ``transitional``/``proviso`` blocks (eId prefix ``disp_uN/``, e.g. the
      Schlusstitel) get ``kind='transitional'`` and number ``uN/<n>`` so that they never collide
      with body articles that carry the same number.
    - ``is_repealed``: no substantive text in force. True when the article has no body and a
      footnote says "aufgehoben / abrogé / abrogato" (or "gegenstandslos" = obsolete), or its
      entire text is such a stub or Fedlex's elision mark "…". Such articles keep an empty text.
    """
    if isinstance(xml, Path):
        xml = xml.read_bytes()
    if isinstance(xml, str):
        xml = xml.encode("utf-8")
    parser = etree.XMLParser(resolve_entities=False, huge_tree=True, no_network=True)
    root = etree.fromstring(xml, parser=parser)
    out: list[ParsedArticle] = []
    for art in root.iterfind(".//a:article", NS):
        eid = art.get("eId") or ""
        number = _article_number(art)
        kind = "article"
        m = re.match(r"^disp_(u\d+)/", eid)
        if m:
            kind = "transitional"
            number = f"{m.group(1)}/{number}"
        body = etree.fromstring(etree.tostring(art))
        for tag in ("num", "heading"):
            for el in body.findall(f"a:{tag}", NS):
                body.remove(el)
        text = text_of(body)
        note = _repeal_note(art)
        repealed = (not text and bool(note)) or _is_repealed_stub(text)
        if repealed and not note:
            note = text
        out.append(
            ParsedArticle(
                number=number,
                kind=kind,
                eid=eid,
                title=_marginal_title(art, has_body=bool(text)),
                text="" if repealed else text,
                is_repealed=repealed,
                repeal_note=note if repealed else "",
            )
        )
    return out


def unarticled_units(xml: bytes | str | Path) -> list[tuple[str, str]]:
    """Transitional/final-provision blocks without any ``<article>`` (plain paragraphs only).

    These are not chunked (one chunk per *article*); they are reported so that nothing is dropped
    silently. Returns ``[(eId, heading), ...]`` in document order.
    """
    if isinstance(xml, Path):
        xml = xml.read_bytes()
    if isinstance(xml, str):
        xml = xml.encode("utf-8")
    root = etree.fromstring(xml, parser=etree.XMLParser(resolve_entities=False, huge_tree=True))
    out: list[tuple[str, str]] = []
    for unit in root.xpath("//a:body/a:transitional | //a:body/a:proviso", namespaces=NS):
        if not unit.findall(".//a:article", NS):
            h = unit.find("a:heading", NS)
            out.append(
                (unit.get("eId") or "", text_of(h).replace("\n", " ") if h is not None else "")
            )
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
