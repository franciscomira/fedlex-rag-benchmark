// Demo page script. No libraries, no external requests. All text from the server is put into the
// page with textContent, never as HTML.
(function () {
  "use strict";

  var LANGS = [["de", "DE"], ["fr", "FR"], ["it", "IT"]];

  function el(tag, cls, text) {
    var node = document.createElement(tag);
    if (cls) node.className = cls;
    if (text !== undefined && text !== null) node.textContent = text;
    return node;
  }

  function safeHttps(url) {
    return typeof url === "string" && url.indexOf("https://") === 0 ? url : null;
  }

  function firstTitle(titles) {
    for (var i = 0; i < LANGS.length; i++) {
      if (titles && titles[LANGS[i][0]]) return titles[LANGS[i][0]];
    }
    return "";
  }

  // The three language columns for one article.
  function languageColumns(hit) {
    var grid = el("div", "langs");
    LANGS.forEach(function (pair) {
      var lang = pair[0];
      var col = el("div", "lang-col");
      col.setAttribute("lang", lang);
      col.appendChild(el("h3", null, pair[1]));
      var text = hit.texts && hit.texts[lang];
      if (hit.titles && hit.titles[lang]) col.appendChild(el("p", "lang-title", hit.titles[lang]));
      if (text) {
        col.appendChild(el("p", "text", text));
      } else {
        col.appendChild(el("p", "missing", hit.is_repealed ? "Repealed: no current text." : "No text in this language."));
      }
      var url = safeHttps(hit.source_urls && hit.source_urls[lang]);
      if (url) {
        var a = el("a", null, "Fedlex source (" + pair[1] + ")");
        a.href = url;
        a.target = "_blank";
        a.rel = "noopener noreferrer";
        col.appendChild(a);
      }
      grid.appendChild(col);
    });
    return grid;
  }

  function metaLine(hit) {
    var bits = [];
    if (hit.consolidation_date) bits.push("Consolidation of " + hit.consolidation_date);
    bits.push("Texts as published by Fedlex, unaltered");
    return el("p", "meta", bits.join(" · "));
  }

  function hitNode(hit) {
    var item = el("li");
    var details = el("details", "hit");
    var summary = el("summary");
    summary.appendChild(el("span", "rank", "#" + hit.rank));
    summary.appendChild(el("span", "ids", "SR " + hit.sr + " · " + hit.article_id));
    summary.appendChild(el("span", "title", firstTitle(hit.titles)));
    if (hit.is_repealed) summary.appendChild(el("span", "badge repealed", "repealed"));
    var score = typeof hit.score === "number" ? hit.score.toFixed(3) : "n/a";
    var s = el("span", "score", "similarity " + score);
    s.title = "Cosine similarity between your question and the best-matching passage. Not a probability.";
    summary.appendChild(s);
    details.appendChild(summary);
    var body = el("div", "hit-body");
    body.appendChild(metaLine(hit));
    body.appendChild(languageColumns(hit));
    details.appendChild(body);
    item.appendChild(details);
    return item;
  }

  function errorMessage(response, data) {
    if (response.status === 429) {
      var wait = response.headers.get("Retry-After");
      return "Too many requests. Please wait" + (wait ? " about " + wait + " seconds" : " a moment") + " and try again.";
    }
    if (response.status === 503) return "The demo is switched off at the moment.";
    if (data && data.error) return data.error;
    return "Something went wrong (HTTP " + response.status + ").";
  }

  // ----- search -----
  var form = document.getElementById("search-form");
  var box = document.getElementById("q");
  var counter = document.getElementById("counter");
  var statusLine = document.getElementById("status");
  var results = document.getElementById("results");
  var button = document.getElementById("go");
  var maxChars = box ? parseInt(box.getAttribute("maxlength"), 10) : 1000;

  function setStatus(message, isError) {
    statusLine.textContent = message;
    statusLine.className = "status" + (isError ? " error" : "");
  }

  if (box) {
    box.addEventListener("input", function () {
      counter.textContent = box.value.length + " / " + maxChars;
    });
  }

  if (form) {
    form.addEventListener("submit", function (event) {
      event.preventDefault();
      var q = box.value.trim();
      if (!q) {
        setStatus("Please type a question.", true);
        return;
      }
      button.disabled = true;
      setStatus("Searching...", false);
      results.textContent = "";
      fetch("/api/search", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ q: q, k: 10 })
      })
        .then(function (response) {
          return response.json().catch(function () { return null; }).then(function (data) {
            return { response: response, data: data };
          });
        })
        .then(function (r) {
          if (!r.response.ok || !r.data) {
            setStatus(errorMessage(r.response, r.data), true);
            return;
          }
          var hits = r.data.hits || [];
          if (!hits.length) {
            setStatus("No articles found.", false);
            return;
          }
          setStatus(hits.length + " articles, best first. Click an article to read it in German, French and Italian.", false);
          hits.forEach(function (hit, i) {
            var node = hitNode(hit);
            if (i === 0) node.firstChild.open = true;
            results.appendChild(node);
          });
        })
        .catch(function () {
          setStatus("The server could not be reached. Please try again.", true);
        })
        .then(function () {
          button.disabled = false;
        });
    });
  }

  // ----- cited articles in the recorded examples -----
  document.addEventListener("click", function (event) {
    var target = event.target.closest ? event.target.closest("button.cite") : null;
    if (!target) return;
    var panel = target.closest(".example-body").querySelector(".cite-panel");
    var id = target.getAttribute("data-article");
    if (target.getAttribute("aria-expanded") === "true") {
      target.setAttribute("aria-expanded", "false");
      panel.textContent = "";
      return;
    }
    panel.parentNode.querySelectorAll("button.cite").forEach(function (b) {
      b.setAttribute("aria-expanded", "false");
    });
    target.setAttribute("aria-expanded", "true");
    panel.textContent = "Loading " + id + "...";
    fetch("/api/article/" + encodeURIComponent(id))
      .then(function (response) {
        return response.json().catch(function () { return null; }).then(function (data) {
          return { response: response, data: data };
        });
      })
      .then(function (r) {
        panel.textContent = "";
        if (!r.response.ok || !r.data || !r.data.article) {
          panel.textContent = errorMessage(r.response, r.data);
          return;
        }
        var hit = r.data.article;
        panel.appendChild(el("p", "ids", "SR " + hit.sr + " · " + hit.article_id));
        panel.appendChild(metaLine(hit));
        panel.appendChild(languageColumns(hit));
      })
      .catch(function () {
        panel.textContent = "The server could not be reached. Please try again.";
      });
  });
})();
