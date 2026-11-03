"""The Markdown subset of the post template, parsed to blocks and rendered to HTML.

Supported: headings (# to ######), paragraphs, bullet and numbered lists (nested by indentation),
fenced code blocks, pipe tables, blockquotes, horizontal rules, and generated blocks
(``{{table NAME}}``, ``{{figure NAME}}``). Inline: ``**bold**``, ``*italic*``, ``code``,
``[text](url)`` and backslash escapes. Raw HTML in the template is escaped, never passed through.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from html import escape
from urllib.parse import urlparse

from fedlex_bench.writeup.placeholders import (
    BLOCK_KINDS,
    CLOSE,
    CODE_SPAN,
    FENCE,
    OPEN,
    TOKEN,
    BuildError,
    Item,
    Tokenized,
)
from fedlex_bench.writeup.tables import TITLES, Table, table_markdown

e = escape

HEADING = re.compile(r"^\s{0,3}(#{1,6})\s+(.*?)(?:\s+#+)?\s*$")
RULE = re.compile(r"^\s{0,3}([-*_])(?:\s*\1){2,}\s*$")
LIST_RE = re.compile(r"^(\s*)([-*+]|\d+[.)])\s+(.*)$")
SEPARATOR = re.compile(r"^\s*\|?\s*:?-+:?\s*(\|\s*:?-+:?\s*)*\|?\s*$")
BLOCK_TOKEN = re.compile(f"^\\s*{OPEN}(\\d+){CLOSE}\\s*$")
ESCAPES = re.compile(r"\\([\\`*_\[\]{}()#+\-.!|<>])")
LINK = re.compile(r"\[([^\]]+)\]\(([^)\s]+)\)")
BOLD = re.compile(r"\*\*(?=\S)(.+?)(?<=\S)\*\*")
ITALIC = re.compile(r"(?<![*\w])\*(?=\S)(.+?)(?<=\S)\*(?![*\w])")
ALLOWED_SCHEMES = {"http", "https", "mailto"}

# ---------------------------------------------------------------- blocks


@dataclass
class Heading:
    level: int
    text: str
    line: int = 0


@dataclass
class Para:
    text: str


@dataclass
class ListItem:
    text: str
    children: list = field(default_factory=list)


@dataclass
class ListBlock:
    ordered: bool
    items: list[ListItem]


@dataclass
class Code:
    lang: str
    text: str


@dataclass
class TableBlock:
    header: list[str]
    aligns: list[str]
    rows: list[list[str]]


@dataclass
class Quote:
    blocks: list


@dataclass
class Rule:
    pass


@dataclass
class Generated:
    item: Item


def indent_of(line: str) -> int:
    return len(line) - len(line.lstrip(" "))


def split_row(line: str) -> list[str]:
    s = line.strip()
    if s.startswith("|"):
        s = s[1:]
    if s.endswith("|") and not s.endswith("\\|"):
        s = s[:-1]
    return [c.replace("\\|", "|").strip() for c in re.split(r"(?<!\\)\|", s)]


def align_of(cell: str) -> str:
    cell = cell.strip()
    if cell.startswith(":") and cell.endswith(":"):
        return "center"
    return "right" if cell.endswith(":") else "left"


class Parser:
    def __init__(self, items: list[Item]):
        self.items = items

    def block_item(self, line: str) -> Item | None:
        m = BLOCK_TOKEN.match(line)
        if m and self.items[int(m.group(1))].kind in BLOCK_KINDS:
            return self.items[int(m.group(1))]
        return None

    def table_start(self, lines: list[str], i: int) -> bool:
        return (i + 1 < len(lines) and "|" in lines[i] and "|" in lines[i + 1]
                and bool(SEPARATOR.match(lines[i + 1])))

    def starts_block(self, lines: list[str], i: int) -> bool:
        line = lines[i]
        return bool(
            FENCE.match(line) or HEADING.match(line) or RULE.match(line)
            or line.lstrip().startswith(">") or LIST_RE.match(line) or self.block_item(line)
            or self.table_start(lines, i)
        )

    def parse(self, lines: list[str], first_line: int = 1) -> list:
        blocks: list = []
        i, n = 0, len(lines)
        while i < n:
            line = lines[i]
            if not line.strip():
                i += 1
                continue
            fence = FENCE.match(line)
            if fence:
                mark = fence.group(1)
                lang = line.strip()[len(mark):].strip()
                body, i = [], i + 1
                while i < n:
                    close = FENCE.match(lines[i])
                    if close and close.group(1)[0] == mark[0] and len(close.group(1)) >= len(mark):
                        break
                    body.append(lines[i])
                    i += 1
                blocks.append(Code(lang, "\n".join(body)))
                i += 1
                continue
            m = HEADING.match(line)
            if m:
                blocks.append(Heading(len(m.group(1)), m.group(2), first_line + i))
                i += 1
                continue
            if RULE.match(line):
                blocks.append(Rule())
                i += 1
                continue
            if line.lstrip().startswith(">"):
                body = []
                while i < n and lines[i].lstrip().startswith(">"):
                    body.append(re.sub(r"^\s*>\s?", "", lines[i]))
                    i += 1
                blocks.append(Quote(self.parse(body, first_line + i - len(body))))
                continue
            item = self.block_item(line)
            if item:
                blocks.append(Generated(item))
                i += 1
                continue
            if LIST_RE.match(line):
                i, block = self.parse_list(lines, i, first_line)
                blocks.append(block)
                continue
            if self.table_start(lines, i):
                header = split_row(line)
                aligns = [align_of(c) for c in split_row(lines[i + 1])]
                rows, i = [], i + 2
                while i < n and lines[i].strip() and "|" in lines[i]:
                    cells = split_row(lines[i])
                    cells += [""] * (len(header) - len(cells))
                    rows.append(cells[: len(header)])
                    i += 1
                aligns += ["left"] * (len(header) - len(aligns))
                blocks.append(TableBlock(header, aligns[: len(header)], rows))
                continue
            para = [line.strip()]
            i += 1
            while i < n and lines[i].strip() and not self.starts_block(lines, i):
                para.append(lines[i].strip())
                i += 1
            blocks.append(Para(" ".join(para)))
        return blocks

    def parse_list(self, lines: list[str], i: int, first_line: int) -> tuple[int, ListBlock]:
        n = len(lines)
        first = LIST_RE.match(lines[i])
        assert first is not None
        base = indent_of(lines[i])
        ordered = first.group(2)[0].isdigit()
        items: list[ListItem] = []
        while i < n:
            m = LIST_RE.match(lines[i])
            if not m or indent_of(lines[i]) != base or m.group(2)[0].isdigit() != ordered:
                break
            text = m.group(3).strip()
            j, sub = i + 1, []
            while j < n:
                if not lines[j].strip():
                    k = j
                    while k < n and not lines[k].strip():
                        k += 1
                    if k < n and indent_of(lines[k]) > base:
                        sub.extend([""] * (k - j))
                        j = k
                        continue
                    if k < n:
                        nxt = LIST_RE.match(lines[k])
                        if (nxt and indent_of(lines[k]) == base
                                and nxt.group(2)[0].isdigit() == ordered):
                            j = k
                    break
                if indent_of(lines[j]) > base:
                    sub.append(lines[j])
                    j += 1
                    continue
                break
            children: list = []
            if sub:
                cut = min(indent_of(s) for s in sub if s.strip())
                children = self.parse([s[cut:] for s in sub], first_line + i + 1)
                if children and isinstance(children[0], Para):
                    text += " " + children[0].text
                    children = children[1:]
            items.append(ListItem(text, children))
            i = j
        return i, ListBlock(ordered, items)


def parse_blocks(tok: Tokenized) -> list:
    return Parser(tok.items).parse(tok.lines)


# ---------------------------------------------------------------- plain text helpers


def value_of(item: Item) -> str:
    return item.value


def plain(text: str, items: list[Item]) -> str:
    """Text without markup, tokens replaced by their values (titles, ids, descriptions)."""
    text = TOKEN.sub(lambda m: value_of(items[int(m.group(1))]), text)
    text = CODE_SPAN.sub(lambda m: m.group(2), text)
    text = LINK.sub(lambda m: m.group(1), text)
    text = ESCAPES.sub(lambda m: m.group(1), text)
    return re.sub(r"[*]{1,2}", "", text).strip()


def slugify(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug or "section"


# ---------------------------------------------------------------- inline HTML


def value_html(item: Item) -> str:
    if item.kind in ("v", "d"):
        head, _, tail = item.value.partition(" ")
        return (f'<span class="v">{e(head)}</span>'
                + (f' <span class="ci">{e(tail)}</span>' if tail else ""))
    if item.kind in ("m", "cost", "lat", "n", "traps", "pct_cheaper", "x_faster"):
        return f'<span class="v">{e(item.value)}</span>'
    return e(item.value)


def check_url(url: str, label: str) -> None:
    scheme = urlparse(url).scheme.lower()
    if scheme and scheme not in ALLOWED_SCHEMES:
        raise BuildError([f"link '{label}' has a '{scheme}:' target; only http, https, mailto "
                          "and relative links are allowed"])


def inline_html(text: str, items: list[Item]) -> str:
    store: list[str] = []

    def keep(html: str) -> str:
        store.append(html)
        return f"{len(store) - 1}"

    text = ESCAPES.sub(lambda m: keep(e(m.group(1))), text)
    text = CODE_SPAN.sub(lambda m: keep(f"<code>{e(m.group(2).strip())}</code>"), text)

    def link(m: re.Match) -> str:
        url = TOKEN.sub(lambda t: value_of(items[int(t.group(1))]), m.group(2))
        check_url(url, m.group(1))
        return keep(f'<a href="{e(url, quote=True)}">{inline_html(m.group(1), items)}</a>')

    text = LINK.sub(link, text)
    text = TOKEN.sub(lambda m: keep(value_html(items[int(m.group(1))])), text)
    text = e(text, quote=False)
    text = BOLD.sub(r"<strong>\1</strong>", text)
    text = ITALIC.sub(r"<em>\1</em>", text)
    return re.sub("(\\d+)", lambda m: store[int(m.group(1))], text)


# ---------------------------------------------------------------- HTML blocks


def generated_table_html(table: Table) -> str:
    out = [f'<div class="tablewrap" tabindex="0"><table class="data" '
           f'aria-label="{e(TITLES.get(table.name, table.name), quote=True)}"><thead><tr>'
           f'<th scope="col">{e(table.first_header)}</th>']
    out += [f'<th scope="col" class="num">{e(h)}</th>' for h in table.headers]
    out.append("</tr></thead><tbody>")
    for row in table.rows:
        out.append(f'<tr><th scope="row"><span class="lab">{e(row.label)}</span> '
                   f'<span class="nm">{e(row.name)}</span></th>')
        for cell in row.cells:
            head, _, tail = cell.partition(" [")
            inner = (f'<span class="v">{e(head)}</span> <span class="ci">[{e(tail)}</span>'
                     if tail else f'<span class="v">{e(head)}</span>')
            out.append(f'<td class="num">{inner}</td>')
        out.append("</tr>")
    out.append("</tbody></table></div>")
    legend = "".join(f"<li><strong>{e(r.label)}</strong>: {e(r.name)}</li>" for r in table.rows)
    out.append(f'<ul class="legend">{legend}</ul>')
    return "".join(out)


class Renderer:
    def __init__(self, items: list[Item]):
        self.items = items
        self.ids: dict[str, int] = {}

    def inline(self, text: str) -> str:
        return inline_html(text, self.items)

    def unique_id(self, text: str) -> str:
        slug = slugify(plain(text, self.items))
        count = self.ids.get(slug, 0)
        self.ids[slug] = count + 1
        return slug if count == 0 else f"{slug}-{count + 1}"

    def block(self, b) -> str:
        if isinstance(b, Heading):
            level = max(2, b.level)
            return f"<h{level}>{self.inline(b.text)}</h{level}>"
        if isinstance(b, Para):
            return f"<p>{self.inline(b.text)}</p>"
        if isinstance(b, ListBlock):
            tag = "ol" if b.ordered else "ul"
            lis = "".join(
                f"<li>{self.inline(i.text)}{''.join(self.block(c) for c in i.children)}</li>"
                for i in b.items)
            return f"<{tag}>{lis}</{tag}>"
        if isinstance(b, Code):
            cls = f' class="language-{e(b.lang, quote=True)}"' if b.lang else ""
            return f"<pre><code{cls}>{e(b.text, quote=False)}</code></pre>"
        if isinstance(b, TableBlock):
            head = "".join(
                f'<th scope="col" class="al-{a}">{self.inline(h)}</th>'
                for h, a in zip(b.header, b.aligns, strict=True))
            rows = "".join(
                "<tr>" + "".join(
                    f'<td class="al-{a}">{self.inline(c)}</td>'
                    for c, a in zip(r, b.aligns, strict=True)) + "</tr>"
                for r in b.rows)
            return (f'<div class="tablewrap" tabindex="0"><table class="data"><thead><tr>{head}'
                    f"</tr></thead><tbody>{rows}</tbody></table></div>")
        if isinstance(b, Quote):
            return "<blockquote>" + "".join(self.block(c) for c in b.blocks) + "</blockquote>"
        if isinstance(b, Rule):
            return "<hr>"
        if isinstance(b, Generated):
            if b.item.kind == "table":
                assert b.item.table is not None
                return generated_table_html(b.item.table)
            assert b.item.figure is not None
            return f'<figure class="wf-fig">{b.item.figure.inline}</figure>'
        raise TypeError(b)


def render_html(blocks: list, items: list[Item]) -> tuple[str, str, list[tuple[str, str]]]:
    """Title (first H1), the main HTML as sections split at H2, and nav entries for the H2s."""
    r = Renderer(items)
    title = ""
    sections: list[tuple[str, str, list[str]]] = []  # id, nav text, html parts
    current: list[str] = []
    cur_id, cur_nav = "intro", ""
    first = True
    for b in blocks:
        if first and isinstance(b, Heading) and b.level == 1:
            title = plain(b.text, items)
            first = False
            continue
        first = False
        if isinstance(b, Heading) and b.level <= 2:
            if current:
                sections.append((cur_id, cur_nav, current))
            cur_id, cur_nav, current = r.unique_id(b.text), plain(b.text, items), []
        if isinstance(b, Heading) and b.level >= 3:
            hid = r.unique_id(b.text)
            level = b.level
            current.append(f'<h{level} id="{e(hid, quote=True)}">{r.inline(b.text)}</h{level}>')
        else:
            current.append(r.block(b))
    if current:
        sections.append((cur_id, cur_nav, current))
    main = []
    nav: list[tuple[str, str]] = []
    for sid, text, parts in sections:
        if text:
            nav.append((sid, text))
        main.append(f'<section id="{e(sid, quote=True)}">' + "".join(parts) + "</section>")
    return title, '<article class="post">' + "".join(main) + "</article>", nav


# ---------------------------------------------------------------- filled Markdown


def fill_markdown(tok: Tokenized) -> str:
    """The template with every placeholder replaced by its value (post.md)."""
    out: list[str] = []
    for line in tok.lines:
        m = BLOCK_TOKEN.match(line)
        item = tok.items[int(m.group(1))] if m else None
        if item is not None and item.kind == "table":
            assert item.table is not None
            out.append(table_markdown(item.table))
        elif item is not None and item.kind == "figure":
            fig = item.figure
            assert fig is not None
            alt = fig.desc.replace("[", "(").replace("]", ")")
            out.append(f"![{alt}](figures/{fig.name}.svg)")
            out.append("")
            out.append(f"*{fig.title}.*")
        else:
            out.append(TOKEN.sub(lambda t: tok.items[int(t.group(1))].value, line))
    return "\n".join(out) + "\n"
