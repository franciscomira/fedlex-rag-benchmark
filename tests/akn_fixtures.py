"""Small inline Akoma Ntoso fixtures mimicking the Fedlex XML structure (no network)."""

from __future__ import annotations

AKN_HEAD = (
    '<akomaNtoso xmlns="http://docs.oasis-open.org/legaldocml/ns/akn/3.0" '
    'xmlns:fedlex="http://fedlex.admin.ch/"><act name="publicLaw"><meta/><body>'
)
AKN_TAIL = "</body></act></akomaNtoso>"


def note(text: str) -> str:
    return f"<authorialNote><p> {text}</p></authorialNote>"


def wrap(inner: str) -> bytes:
    return (AKN_HEAD + inner + AKN_TAIL).encode("utf-8")


# One body article with two numbered paragraphs, a footnote in the text and a marginal level.
ART_1 = (
    '<level eId="lvl_a" fedlex:role="marginal"><num>a. </num><heading>Abschluss</heading>'
    '<article eId="art_1"><num><b>Art. 1</b></num>'
    '<paragraph eId="art_1/para_1"><num>1</num><content>'
    f"<p> Zum Abschluss ist Willenseinigung erforderlich.{note('Fassung gemaess X.')}</p>"
    "</content></paragraph>"
    '<paragraph eId="art_1/para_2"><num>2</num><content>'
    "<p> Sie kann ausdruecklich sein.</p></content></paragraph>"
    "</article></level>"
)

# Suffix numbering: 257d (letter), 12bis, 6a+bis, 40ter, 9quater.
ART_257_D = (
    '<article eId="art_257_d"><num><b>Art. 257</b><i>d</i></num>'
    '<paragraph eId="art_257_d/para"><content><p>Text 257d.</p></content></paragraph></article>'
)
ART_12_BIS = (
    '<article eId="art_12_bis"><num><b>Art. 12</b><sup>bis</sup></num>'
    '<paragraph eId="art_12_bis/para"><content><p>Text 12bis.</p></content></paragraph></article>'
)
ART_6_A_BIS = (
    '<article eId="art_6_a_bis"><num><b>Art. 6</b><i>a</i><sup>bis</sup></num>'
    '<paragraph eId="art_6_a_bis/para"><content><p>Text 6abis.</p></content></paragraph></article>'
)
ART_40_TER = (
    '<article eId="art_40_ter"><num><b>Art. 40</b><sup>ter</sup></num>'
    '<paragraph eId="art_40_ter/para"><content><p>Text 40ter.</p></content></paragraph></article>'
)
ART_9_QUATER = (
    '<article eId="art_9_quater"><num><b>Art. 9</b><sup>quater</sup></num>'
    '<paragraph eId="art_9_quater/para"><content><p>Text 9quater.</p></content></paragraph>'
    "</article>"
)

# Repealed: no body at all, repeal only declared in a footnote on the number.
ART_10_REPEALED = (
    '<article eId="art_10"><num><b>Art. 10</b>'
    f"{note('Aufgehoben durch Anhang 1 Ziff. II 3 der ZPO, mit Wirkung seit 1. Jan. 2011.')}"
    "</num></article>"
)
# Repealed after having been inserted (history in the note, repeal word not at the start).
ART_40_G_REPEALED = (
    '<article eId="art_40_g"><num><b>Art. 40</b><i>g</i>'
    f"{note('Eingefügt durch Ziff. I des BG vom 5. Okt. 1990. Aufgehoben durch Ziff. I des BG.')}"
    "</num></article>"
)
# Repealed range of articles.
ART_135_149_REPEALED = (
    '<article eId="art_135_149"><num><b>Art. 135</b>–<b>149</b>'
    f"{note('Aufgehoben durch die ZPO.')}</num></article>"
)
# French wording and an empty <content/> body ("Gegenstandslos" equivalent in IT).
ART_OBSOLETE = (
    '<article eId="art_77"><num><b>Art. 77</b>'
    f"{note('Privo d’oggetto.')}</num>"
    '<content><paragraph eId="art_77/para"><content/></paragraph></content></article>'
)
# Body is just an explicit stub paragraph.
ART_STUB_BODY = (
    '<article eId="art_78"><num><b>Art. 78</b></num>'
    '<paragraph eId="art_78/para"><content><p>Abrogé</p></content></paragraph></article>'
)
# Only one paragraph is repealed: the article itself is NOT repealed.
ART_PARTIAL_REPEAL = (
    '<article eId="art_56"><num><b>Art. 56</b></num>'
    '<paragraph eId="art_56/para_1"><num>1</num><content><p>Haftung des Tierhalters.</p>'
    "</content></paragraph>"
    '<paragraph eId="art_56/para_2"><num>2</num><content><p>…</p></content></paragraph>'
    "</article>"
)
# Italian markup quirk: number split, footnote ends up in <heading>.
ART_IT_SPLIT = (
    '<article eId="art_697_l_697_m"><num><b>Art. 697</b><i>l </i></num>'
    f"<heading>e <b>Art. 697</b><i>m</i>{note('Abrogati dall’all. n. 2 della L del 26 set. 2025.')}"
    "</heading></article>"
)

# Transitional / final provisions: ids collide with body articles (art_1) and must stay apart.
TRANSITIONAL = (
    '<proviso eId="disp_u1"><heading>Schlusstitel</heading>'
    '<article eId="disp_u1/art_1"><num><b>Art. 1</b></num>'
    '<paragraph eId="disp_u1/art_1/para"><content><p>Uebergangsrecht.</p></content></paragraph>'
    "</article>"
    '<article eId="disp_u1/art_12_c_bis"><num><b>Art. 12</b><i>c</i><sup>bis</sup></num>'
    '<paragraph eId="disp_u1/art_12_c_bis/para"><content><p>Uebergang cbis.</p></content>'
    "</paragraph></article></proviso>"
)
# Transitional block holding amendments in the form "Art. 2 4" (number parts, eId art_2_4).
TRANSITIONAL_2_4 = (
    '<transitional eId="disp_u12"><heading>Schlussbestimmungen</heading>'
    '<article eId="disp_u12/art_2_4"><num><b>Art. 2</b> <b>4</b></num>'
    '<content><paragraph eId="disp_u12/art_2_4/para"><content><p>…</p></content>'
    "</paragraph></content></article></transitional>"
)
# Transitional unit with no <article> inside (plain paragraphs only).
TRANSITIONAL_NO_ARTICLE = (
    '<transitional eId="disp_u4"><heading>Uebergangsbestimmung zur Aenderung</heading>'
    '<paragraph eId="disp_u4/para"><content><p>Gilt sinngemaess.</p></content></paragraph>'
    "</transitional>"
)


# Paragraph number followed by a list: paragraph > num, content > blockList > listIntroduction.
# The "1" must stay on the same line as the introduction (it used to be left orphaned).
ART_LIST = (
    '<article eId="art_20"><num><b>Art. 20</b></num>'
    '<paragraph eId="art_20/para_1"><num>1</num><content><blockList>'
    '<listIntroduction eId="art_20/para_1/listintro">Der Kunde kann widerrufen, wenn:'
    "</listIntroduction>"
    '<item eId="art_20/para_1/lbl_a"><num>a. </num><p>am Arbeitsplatz;</p></item>'
    '<item eId="art_20/para_1/lbl_b"><num>b. </num><p>am Telefon.</p></item>'
    "</blockList></content></paragraph>"
    '<paragraph eId="art_20/para_2"><num>2</num><content><p>Zweiter Absatz.</p></content>'
    "</paragraph></article>"
)

# Block quoting superseded wording (ZGB "Wortlaut der fruheren Bestimmungen").
HISTORICAL = (
    '<proviso eId="disp_u5"><heading>Wortlaut der früheren Bestimmungen des sechsten Titels'
    "</heading>"
    '<article eId="disp_u5/art_178"><num><b>Art. 178</b></num>'
    '<paragraph eId="disp_u5/art_178/para"><content><p>Altes Recht.</p></content></paragraph>'
    "</article></proviso>"
)
# Amendment of another act: introduction ending in ":" followed by the elision mark.
AMENDMENT = (
    '<transitional eId="disp_u6"><heading>Änderung von Bundesgesetzen</heading>'
    '<article eId="disp_u6/art_2"><num><b>Art. 2</b></num>'
    '<paragraph eId="disp_u6/art_2/para"><content><p>Sono modificate:</p>'
    "<p>…</p></content></paragraph></article></transitional>"
)


def full_act() -> bytes:
    return wrap(
        ART_1
        + ART_257_D
        + ART_12_BIS
        + ART_6_A_BIS
        + ART_40_TER
        + ART_9_QUATER
        + ART_10_REPEALED
        + ART_40_G_REPEALED
        + ART_135_149_REPEALED
        + ART_OBSOLETE
        + ART_STUB_BODY
        + ART_PARTIAL_REPEAL
        + ART_IT_SPLIT
        + TRANSITIONAL
        + TRANSITIONAL_2_4
        + TRANSITIONAL_NO_ARTICLE
        + ART_LIST
        + HISTORICAL
        + AMENDMENT
    )
