"""The Markdown subset: blocks and inline rendering of the post template."""

from __future__ import annotations

import pytest
from writeup_fixtures import make_project

from fedlex_bench.site.build import build_report
from fedlex_bench.writeup.build import build_post
from fedlex_bench.writeup.placeholders import BuildError


@pytest.fixture(scope="module")
def project(tmp_path_factory):
    return make_project(tmp_path_factory.mktemp("md"))


def render(project, text: str) -> str:
    root, manifest = project
    post = build_post(build_report(root, manifest), "# Title\n\n" + text, root)
    main = post.html.split("<main", 1)[1].split("</main>", 1)[0]
    return main.split(">", 1)[1]


def test_headings_sections_and_nav(project):
    root, manifest = project
    post = build_post(build_report(root, manifest),
                      "# The title\n\nIntro.\n\n## First part\n\ntext\n\n### Sub *part*\n\n"
                      "## Second part\n\n## First part\n", root)
    assert "<h1>The title</h1>" in post.html
    assert post.html.count("<h1>") == 1
    assert '<section id="intro"><p>Intro.</p></section>' in post.html
    assert '<section id="first-part"><h2>First part</h2><p>text</p>' in post.html
    assert '<h3 id="sub-part">Sub <em>part</em></h3>' in post.html
    assert '<section id="first-part-2">' in post.html  # ids stay unique
    nav = post.html.split("<nav", 1)[1].split("</nav>", 1)[0]
    assert nav.index("#first-part\"") < nav.index("#second-part") < nav.index("index.html")


def test_inline_markup(project):
    html = render(project, "A **bold** and *italic* and `co*de` and "
                           "[link](https://x.org/a?b=1&c=2) and \\*literal\\* and <b>tag</b>.")
    assert ("<p>A <strong>bold</strong> and <em>italic</em> and <code>co*de</code> and "
            '<a href="https://x.org/a?b=1&amp;c=2">link</a> and *literal* and '
            "&lt;b&gt;tag&lt;/b&gt;.</p>") in html


def test_lists_nested_and_numbered(project):
    html = render(project, "- one\n- two with `x`\n  - inner a\n  - inner b\n- three\n\n"
                           "1. first\n2. second\n   continued\n")
    assert ("<ul><li>one</li><li>two with <code>x</code><ul><li>inner a</li><li>inner b</li>"
            "</ul></li><li>three</li></ul>") in html
    assert "<ol><li>first</li><li>second continued</li></ol>" in html


def test_code_block_is_escaped_and_literal(project):
    html = render(project, "```python\nx = 1 < 2  # {{v T1 recall@10}} *not bold*\n```")
    assert ('<pre><code class="language-python">x = 1 &lt; 2  # {{v T1 recall@10}} *not bold*'
            "</code></pre>") in html


def test_pipe_table_with_alignment(project):
    html = render(project, "| Name | Value |\n|:--|--:|\n| a \\| b | **1** |\n| c | 2 |\n")
    assert '<th scope="col" class="al-left">Name</th><th scope="col" class="al-right">Value</th>' \
        in html
    assert '<td class="al-left">a | b</td><td class="al-right"><strong>1</strong></td>' in html
    assert html.count("<tr>") == 3 and 'class="tablewrap"' in html


def test_quote_rule_and_paragraph_joining(project):
    html = render(project, "> quoted *text*\n> second line\n\n---\n\nline one\nline two\n")
    assert "<blockquote><p>quoted <em>text</em> second line</p></blockquote><hr>" in html
    assert "<p>line one line two</p>" in html


def test_placeholder_values_are_styled_spans_in_html(project):
    html = render(project, "T1: {{v T1 recall@10}}; mean {{m T1 recall@10}}; {{lat T1}}.")
    assert '<span class="v">0.81</span> <span class="ci">[' in html
    assert 'mean <span class="v">0.81</span>;' in html


def test_unclosed_fence_is_an_error(project):
    root, manifest = project
    with pytest.raises(BuildError):
        build_post(build_report(root, manifest), "# T\n\n```\nnever closed\n", root)


def test_figure_and_table_blocks(project):
    html = render(project, "{{figure fig_english}}\n\n{{table by_language}}\n")
    assert '<figure class="wf-fig"><svg class="wf"' in html
    assert 'aria-label="Recall@10 by language of the question"' in html
    assert '<ul class="legend">' in html
