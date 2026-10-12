"""
pipeline.py - extract structured RFP information from a bid folder with LLM agents.

Steps
  0. Ingest    every PDF (page by page) and HTML (one page) in the bid folder into memory.
  1. Summarise give the LLM the pages in <= 20k-char batches and collect a brief summary plus
               per-page "signals" (entities / content) that tell the agents where to look.
  2. Extract   run one LangChain agent per field group defined in GROUPS below. Each agent has
               bm25_search (document-scoped, optional page filter), grep_lines, read_page and
               register_record tools. Every passage the retrieval tools return carries an evidence id
               ([b#] bm25, [g#] grep, [r#] read_page); register_record only accepts values that cite
               ids shown in that agent's conversation and derives the source page from them, so the
               summary/signals in the prompt can never be the source. The loop ends as soon as the
               group's fields are registered, or after --max-turns when the agent is told to register
               whatever it has.
  3. Write     output/<Bid>.json                  field -> value
               output/<Bid>/<Bid>.evidence.json   how each value was obtained (source pages, cited snippets,
                                                  key_evidence, verified flag, confidence, turn)

Usage
  0. put the bid's documents in ./task/<BidName>/
  1. python -m venv .venv && .venv\\Scripts\\pip install -r requirements.txt
  2. fill in the AZURE_OPENAI_* settings in the CONFIG section below
  3. .venv\\Scripts\\python pipeline.py task\\Bid1 [--max-turns 24] [--max-tokens-before-compression 60000] [--no-cache]

Sections: CONFIG | prompt | doc_ingestion | tools | agent | main
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from itertools import groupby
from pathlib import Path
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler

# =====================================================================================
# CONFIG
# =====================================================================================

# Azure AI Foundry -> Azure OpenAI deployment. Endpoint is the resource URL (not the /api/projects/... URL); the
# API version is a REST version, not the model version; the deployment must support tool calling.
# All tests are run on gpt-4.1-mini.
AZURE_OPENAI_ENDPOINT = ""
AZURE_OPENAI_API_KEY = ""
AZURE_OPENAI_API_VERSION = ""
AZURE_OPENAI_DEPLOYMENT = "gpt-4.1-mini"
AZURE_OPENAI_SUMMARY_DEPLOYMENT = ""  # optional cheaper deployment for the summary step and history compression

MAX_CHARS_PER_BATCH = 20_000        # summary & signals: text handed to the LLM per call
CHUNK_CHARS, CHUNK_OVERLAP = 1000, 200  # BM25 passage size
MAX_TOP_K = 10
SNIPPET_KEEP_CHARS = 600  # evidence.json keeps at most this much of each cited snippet

@dataclass
class Settings:
    bid_dir: Path
    out_path: Path
    max_turns: int
    max_tokens_before_compression: int
    no_cache: bool

    @property
    def bid_out_dir(self) -> Path:
        """output/<BidName>/ - the summary cache and the evidence file live here, next to <BidName>.json."""
        return self.out_path.parent / self.bid_dir.name

    @property
    def evidence_path(self) -> Path:
        return self.bid_out_dir / f"{self.bid_dir.name}.evidence.json"


def build_llm(summary: bool = False):
    """Chat model on the Azure OpenAI deployment configured above (the summary deployment when set and asked for)."""
    from langchain_openai import AzureChatOpenAI

    return AzureChatOpenAI(azure_endpoint=AZURE_OPENAI_ENDPOINT, api_key=AZURE_OPENAI_API_KEY, api_version=AZURE_OPENAI_API_VERSION,
                           azure_deployment=(AZURE_OPENAI_SUMMARY_DEPLOYMENT if summary else "") or AZURE_OPENAI_DEPLOYMENT,
                           stream_usage=True)


class TokenTracker(BaseCallbackHandler):
    """Counts input/output tokens of every LLM call made under a run, including calls made inside middleware.

    Pass it as `config={"callbacks": [tracker]}` to llm.invoke / agent.stream; LangChain then calls `on_llm_end`
    with each response, whose usage_metadata is accumulated. `since(mark)` reports the usage of one stage or turn.
    """

    def __init__(self):
        super().__init__()
        self.calls = self.input = self.output = 0

    def on_llm_end(self, response, **kwargs) -> None:
        for generation in response.generations:
            for g in generation:
                usage = getattr(getattr(g, "message", None), "usage_metadata", None)
                if usage:
                    self.calls += 1
                    self.input += usage.get("input_tokens", 0)
                    self.output += usage.get("output_tokens", 0)

    def mark(self) -> tuple[int, int, int]:
        return self.calls, self.input, self.output

    def since(self, mark: tuple[int, int, int]) -> str:
        calls, inp, out = self.calls - mark[0], self.input - mark[1], self.output - mark[2]
        return f"tokens in {inp:,} / out {out:,}" + (f" ({calls} calls)" if calls != 1 else "")

    def total(self) -> str:
        return f"in {self.input:,} / out {self.output:,} / total {self.input + self.output:,} over {self.calls} LLM calls"


# =====================================================================================
# prompt
# =====================================================================================

SUMMARY_SIGNALS_SYSTEM = """You are an analyst preparing retrieval aids for a procurement (RFP) document set.
You receive the text of one or more pages of a single document. Answer in exactly this markdown layout:

## Summary
A very brief summary (3-6 sentences) of what these pages contain and what kind of document this is
(base RFP, Q&A addendum, date-change addendum, bid-portal listing, ...).

## Page Signals
For EACH page that carries useful information:

### Page <N>
#### Entities
- concrete entities: bid numbers, organizations, people, emails, phones, URLs, dates/times (and what they are for),
  amounts, form names, section titles, product/tier names, quantities, spec keywords, contract terms, bond/insurance terms.
#### Content
- 1-4 bullets on what the page covers and which output fields it could answer (due date, submission method, term,
  pre-bid meeting, delivery, payment, bond, required documents, cooperative, installation, products, specs, manufacturers).

Skip pages that are only boilerplate, blank forms or repeated headers. Use the page labels exactly as given.
Be terse and factual; never invent information."""

MERGE_SUMMARIES_PROMPT = """Merge these partial summaries of the same {scope} into ONE brief markdown summary (4-8 sentences)
that keeps every concrete fact and removes repetition. Respond with the merged summary only.

{summaries}"""

FIELD_AGENT_SYSTEM = """You are a procurement analyst agent working on request for proposal (RFP) documents. 
The task is to understand the documents and scavenge information for filling "{group}" for a bid,
working using from the documents below.

# Documents (use the id or the exact name as the `document` argument of the tools)
{doc_index}

# Fields to fill (name, required datatype, what to extract)
{fields_spec}

# How to work
- Use the summary and page signals at the end to decide where to look, then:
  * bm25_search(document, query, top_k, page_no): ranked ~1000-char passages from ONE document (optionally one
    page) with a 0-1 confidence. Try several phrasings; search the addenda as well as the main RFP.
  * grep_lines(pattern, document): exact/regex hits as doc:page:line - best for numbers, IDs, dates, "must submit".
  * read_page(document, page_no): the full page - use for tables (price schedules, spec lists, SKU lists).
- Every passage a tool returns carries an evidence id in square brackets: [b1], [b2] ... for bm25_search passages,
  [g1], [g2] ... for grep_lines hits, [r1], [r2] ... for read_page pages. Note the ids of the passages that state
  the values you extract - they are the proof that register_record requires.
- The summary and page signals below are hints about where to look, never evidence: they have no ids, so nothing
  can be registered from them until the text has been retrieved with a tool.
- Precedence when sources disagree: latest addendum > earlier addendum > base RFP > portal listing.
- Record results with register_record(field, value_json, evidence, key_evidence, confidence). A registration is
  REJECTED (nothing stored) when it cites no evidence id, an id that was never shown to you, or invalid JSON:
  * value_json is JSON whose root type matches the datatype (str -> string, list -> array, dict -> object with the
    keys named in the spec). Datetimes as ISO-8601 with offset, dates as YYYY-MM-DD, unknown scalars as null.
  * evidence: the ids of the passages that state the value, e.g. ["g3", "r1"]. The source pages are derived from them.
  * key_evidence (dict and list values): {{"key or index": "id"}} naming which passage backs each key / item, so
    that a value with several parts is traceable. Keys named notes / *_notes hold your own wording.
  * The tool answers with a WARNING when a value (or list item) does not appear in the cited passages - re-check
    the passage and correct the value or cite the passage that really contains it.
  * Calling register_record again for the same field MERGES (dict keys updated, list items appended, scalars
    replaced), so you may register early and refine.
  * If the documents do not state a value: dict -> null keys plus a "notes" key; list -> []; str -> a short
    "not specified" string. Cite the passage you checked where the value would have appeared (schedule table,
    submission instructions, ...); evidence may be empty only for a fully-null value. Never leave a field unregistered.
- Keep values close to the source wording and mention the addendum number when an addendum changed something.
- Stop as soon as every field of this group is registered. Keep chat text minimal; the records are the output.
{prior}
# Summary and page signals of all documents
Below are a brief on the documents provided for gathering the information. Use the summary and signals to gather any information required.
{summary}
"""

PRIOR_RESULTS_BLOCK = """
# Results already registered by the other field groups (primary source for your field)
```json
{results}
```
"""

FINAL_TURN_MESSAGE = ("MAX TURNS REACHED. Do not call search/grep/read tools any more. Using only what is already in this "
                      "conversation, call register_record now for each of: {fields}, citing the evidence ids already shown; "
                      "where nothing was retrieved, register the fully-null form (null / [] / null keys plus notes) with "
                      "empty evidence, then reply DONE.")

COMPRESSION_PROMPT = """Compress the working memory of an RFP extraction agent. Replace the conversation with bullet lists:
## Fields already registered (field -> value and source)
## Candidate facts found but not yet registered (exact values, each with the evidence id(s) [b#]/[g#]/[r#] and
   docN:page of the passage that states it - register_record needs those ids)
## Places already inspected (documents/pages read, queries run, what was NOT found)
## Remaining work
Respond only with that context.

<messages>
{messages}
</messages>"""


# =====================================================================================
# doc_ingestion
# =====================================================================================

@dataclass
class Page:
    page_no: int
    text: str


@dataclass
class Document:
    id: str          # short handle for the agent, e.g. "doc2"
    name: str        # file name
    kind: str        # "pdf" | "html"
    pages: list[Page]


@dataclass
class Chunk:
    """A ~CHUNK_CHARS piece of one page: the unit BM25 scores and returns."""
    doc: Document
    page_no: int
    text: str


class Corpus:
    """All documents of one bid, held in memory, plus one BM25 index over every page chunk.

    The index is corpus-wide (IDF statistics need the whole bid; a one-page addendum on its own has none) and
    searches are filtered down to the requested document / page."""

    def __init__(self, documents: list[Document]):
        import spacy
        from rank_bm25 import BM25Okapi

        self.documents = documents
        self.chunks = [Chunk(d, p.page_no, t) for d in documents for p in d.pages for t in _chunks(p.text)]
        self._nlp = spacy.blank("en")  # tokenizer + English stop words only; no statistical model needed
        self._bm25 = BM25Okapi([self.tokens(c.text) for c in self.chunks])

    def tokens(self, text: str) -> list[str]:
        return [t.lower_ for t in self._nlp.tokenizer(text) if not (t.is_stop or t.is_punct or t.is_space)]

    def find(self, ref: str) -> Document | None:
        """Resolve a document by id, exact file name, or unique case-insensitive substring."""
        for d in self.documents:
            if ref in (d.id, d.name):
                return d
        hits = [d for d in self.documents if ref.lower() in d.name.lower()]
        return hits[0] if len(hits) == 1 else None

    def index_markdown(self) -> str:
        return "\n".join(f"- {d.id}: `{d.name}` ({d.kind}, {len(d.pages)} pages)" for d in self.documents)

    def search(self, doc: Document, query: str, top_k: int, page_no: int | None) -> list[tuple[Chunk, float, float]]:
        """Top-k chunks of one document (optionally one page) as (chunk, bm25 score, confidence 0-1)."""
        tokens = self.tokens(query)
        if not tokens:
            return []
        scored = [(c, float(s)) for c, s in zip(self.chunks, self._bm25.get_scores(tokens))
                  if s > 0 and c.doc is doc and (page_no is None or c.page_no == page_no)]
        scored.sort(key=lambda x: x[1], reverse=True)
        best = scored[0][1] if scored else 1.0
        return [(c, s, s / best) for c, s in scored[:top_k]]


def _norm(text: str) -> str:
    """Comparison form for the soft evidence check: letters and digits only. PDF extraction yields "9 0 days" and
    "md\n.us" where the model writes "90 days" and "md.us"; dropping spaces and punctuation makes them equal."""
    return re.sub(r"[^a-z0-9]", "", str(text).lower())


def _clean(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\x00", "")
    return re.sub(r"\n{3,}", "\n\n", "\n".join(l.rstrip() for l in text.split("\n"))).strip()


def _chunks(text: str) -> list[str]:
    """Cut page text at line breaks into ~CHUNK_CHARS pieces; each piece repeats the previous one's last
    CHUNK_OVERLAP chars so a fact on a boundary is still found whole."""
    pieces, cur = [], ""
    for line in text.splitlines(keepends=True):
        if cur.strip() and len(cur) + len(line) > CHUNK_CHARS:
            pieces.append(cur.strip())
            cur = cur[-CHUNK_OVERLAP:]
        cur += line
    return pieces + ([cur.strip()] if cur.strip() else [])


def _pdf_pages(path: Path) -> list[Page]:
    from pypdf import PdfReader

    pages = []
    for i, page in enumerate(PdfReader(str(path)).pages, 1):
        text = page.extract_text() or ""
        if len(text.strip()) < 20:  # image-only or odd encoding: try the second extractor
            import pypdfium2 as pdfium

            text = pdfium.PdfDocument(str(path))[i - 1].get_textpage().get_text_range()
        pages.append(Page(i, _clean(text)))
    return pages


def _html_pages(path: Path) -> list[Page]:
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(path.read_text(encoding="utf-8", errors="ignore"), "lxml")
    for tag in soup(["script", "style", "noscript", "svg", "nav"]):
        tag.decompose()
    lines = (re.sub(r"\s+", " ", l).strip() for l in soup.get_text("\n").splitlines())
    return [Page(1, _clean("\n".join(k for k, _ in groupby(lines) if k)))]  # groupby drops menu items rendered twice


def load_bid_folder(bid_dir: Path) -> Corpus:
    files = sorted(p for p in bid_dir.iterdir() if p.suffix.lower() in (".pdf", ".html", ".htm"))
    if not files:
        raise FileNotFoundError(f"No PDF/HTML documents in {bid_dir}")
    docs = []
    for i, path in enumerate(files, 1):
        print(f"  reading {path.name}")
        is_pdf = path.suffix.lower() == ".pdf"
        docs.append(Document(f"doc{i}", path.name, "pdf" if is_pdf else "html", _pdf_pages(path) if is_pdf else _html_pages(path)))
    return Corpus(docs)


# =====================================================================================
# tools
# =====================================================================================

@dataclass
class FieldSpec:
    name: str
    datatype: str
    extracted: str
    guidance: str

    @property
    def root_type(self) -> type | None:
        """Python type the registered JSON value should have: the outer name of the datatype annotation."""
        outer = re.match(r"\w*", self.datatype.lower())[0]
        return {"str": str, "list": list, "dict": dict, "bool": bool, "int": int, "float": float}.get(outer)


@dataclass
class FieldGroup:
    name: str
    fields: list[FieldSpec] = field(default_factory=list)
    run_last: bool = False  # the Synthesis group is built from the other groups' results


@dataclass
class Snippet:
    """One passage a retrieval tool showed the agent; its id (b#, g#, r#) is the evidence register_record accepts."""
    id: str
    tool: str
    doc: str
    page: int
    text: str


def _merge(old: Any, new: Any) -> Any:
    """Repeat registrations: dicts merge key-wise, lists append without duplicates, scalars are replaced."""
    if isinstance(old, dict) and isinstance(new, dict):
        return {**old, **{k: _merge(old[k], v) if k in old else v for k, v in new.items()}}
    if isinstance(old, list):
        seen = {json.dumps(x, sort_keys=True, default=str) for x in old}
        return old + [v for v in (new if isinstance(new, list) else [new]) if json.dumps(v, sort_keys=True, default=str) not in seen]
    return new


class RecordStore:
    """Registered field values plus their evidence history; also tells each agent loop when its group is complete."""

    def __init__(self, groups: list[FieldGroup]):
        self.groups = groups
        self.specs = {f.name: f for g in groups for f in g.fields}
        self.values: dict[str, Any] = {}
        self.evidence: dict[str, list[dict]] = defaultdict(list)
        self.turn = 0

    def canonical(self, name: str) -> str | None:
        norm = lambda s: re.sub(r"[\s_\-]+", "", s).lower()  # noqa: E731 - accept "bid_number" for "Bid Number"
        return next((n for n in self.specs if norm(n) == norm(name)), None)

    def register(self, name: str, value: Any, source: str, confidence: float, **provenance) -> str:
        action = "merged" if name in self.values else "registered"
        self.values[name] = _merge(self.values[name], value) if name in self.values else value
        self.evidence[name].append({"turn": self.turn, "action": action, "value": value, "source": source, "confidence": confidence,
                                    **provenance, "timestamp": dt.datetime.now().isoformat(timespec="seconds")})
        return action

    def unfilled(self, group: FieldGroup) -> list[str]:
        return [f.name for f in group.fields if f.name not in self.values]

    def fields_json(self) -> dict[str, Any]:
        return {f.name: self.values.get(f.name) for g in self.groups for f in g.fields}


def _scalars(value: Any, path: str = "") -> list[tuple[str, str]]:
    """(path, text) of every document-derived scalar in a value. Skips nulls, empties, booleans, 'not specified'
    strings and keys the model words itself (notes, *_notes, source) - what is left must be backed by evidence."""
    if isinstance(value, dict):
        return [x for k, v in value.items() if not re.search(r"(^|_)(notes?|source)$", str(k), re.I) for x in _scalars(v, f"{path}.{k}".lstrip("."))]
    if isinstance(value, list):
        return [x for i, v in enumerate(value) for x in _scalars(v, f"{path}[{i}]")]
    if value in (None, "") or isinstance(value, bool) or str(value).lower().startswith(("not specified", "not stated", "n/a")):
        return []
    return [(path or "value", str(value))]


def _spellings(text: str) -> list[str]:
    """Normalised forms a value may take in the source. ISO dates (the spec's output format) are also tried as
    06/01/2024, 6/1/2024, 06/01/24, June 01, 2024 and 1 June 2024, since documents never print ISO."""
    forms = [text]
    try:
        d = dt.datetime.fromisoformat(text.strip())
        forms += [d.strftime(f) for f in ("%m/%d/%Y", "%m/%d/%y", "%B %d, %Y", "%d %B %Y")]
        forms += [re.sub(r"\b0(\d)", r"\1", f) for f in forms[1:]]
    except ValueError:
        pass
    return [_norm(f) for f in forms]


def make_tools(corpus: Corpus, store: RecordStore, group: FieldGroup) -> list:
    """The four tools of a field agent, closed over the corpus, the shared store and the agent's group."""
    from langchain_core.tools import tool

    allowed = [f.name for f in group.fields]
    unknown_doc = "Unknown document. Available:\n" + corpus.index_markdown()
    shown: dict[str, Snippet] = {}  # every passage handed to this agent, by evidence id; lives outside the chat history
    counts: Counter[str] = Counter()

    def show(tool_name: str, doc: str, page: int, text: str) -> str:
        """Keep a passage under the next id of its tool (b#, g#, r# = first letter of the tool name) and return the id."""
        counts[tool_name] += 1
        sid = f"{tool_name[0]}{counts[tool_name]}"
        shown[sid] = Snippet(sid, tool_name, doc, page, text)
        return sid

    @tool
    def bm25_search(document: str, query: str, top_k: int = 5, page_no: int | None = None) -> str:
        """BM25 search inside ONE document (id like "doc1" or its name), optionally restricted to one page.
        Returns up to top_k (max 10) passages, each with an evidence id [b#], page number, BM25 score and 0-1 confidence
        relative to the best hit."""
        doc = corpus.find(document)
        if doc is None:
            return unknown_doc
        hits = corpus.search(doc, query, max(1, min(int(top_k), MAX_TOP_K)), page_no)
        if not hits:
            return f"QUESTION: {query}\nNo matching passage in {doc.id}{f' page {page_no}' if page_no else ''}. Try other keywords."
        out = [f"QUESTION: {query}", f"DOCUMENT: {doc.id} {doc.name}"]
        for rank, (c, score, conf) in enumerate(hits, 1):
            sid = show("bm25_search", doc.id, c.page_no, c.text)
            out += [f"\n[{sid}] #{rank} | {doc.id} page {c.page_no} | bm25={score:.2f} | confidence={conf:.2f}", c.text]
        return "\n".join(out)

    @tool
    def grep_lines(pattern: str, document: str | None = None, context: int = 1, max_hits: int = 30) -> str:
        """Case-insensitive regex search over every line of every page (or of one document). Returns hits with an
        evidence id [g#], doc:page:line and `context` surrounding lines. Best for exact tokens: dates, IDs, amounts, form names."""
        try:
            rx = re.compile(pattern, re.IGNORECASE)
        except re.error as e:
            return f"Invalid regex: {e}"
        docs = [corpus.find(document)] if document else corpus.documents
        if docs == [None]:
            return unknown_doc
        ctx, out = max(0, min(int(context), 5)), []
        for d in docs:
            for p in d.pages:
                lines = p.text.split("\n")
                for i in (i for i, line in enumerate(lines) if rx.search(line)):
                    lo, hi = max(0, i - ctx), min(len(lines), i + ctx + 1)
                    sid = show("grep_lines", d.id, p.page_no, "\n".join(lines[lo:hi]))
                    out.append(f"[{sid}] {d.id} | page {p.page_no} | line {i + 1}\n" + "\n".join(("> " if j == i else "  ") + lines[j] for j in range(lo, hi)))
                    if len(out) >= max_hits:
                        return "\n\n".join(out) + f"\n... stopped at {max_hits} hits; narrow the pattern."
        return "\n\n".join(out) or f"No lines match /{pattern}/."

    @tool
    def read_page(document: str, page_no: int) -> str:
        """Full text of one page of a document (id or name; 1-based page), with an evidence id [r#]. Use for tables and spec lists."""
        doc = corpus.find(document)
        if doc is None:
            return unknown_doc
        if not 1 <= int(page_no) <= len(doc.pages):
            return f"{doc.id} has pages 1..{len(doc.pages)}."
        text = doc.pages[int(page_no) - 1].text
        sid = show("read_page", doc.id, int(page_no), text[:12_000])
        return f"[{sid}] {doc.id} | page {page_no}\n" + (text if len(text) <= 12_000 else text[:12_000] + f"\n... [truncated, {len(text)} chars]")

    @tool
    def register_record(field: str, value_json: str | list | dict, evidence: str | list | None = None,
                        key_evidence: str | dict | None = None, confidence: float = 0.5) -> str:
        """Record an output field of this group. REJECTED (nothing stored) without valid evidence ids.
        value_json: JSON whose root type matches the field datatype (a JSON string, or the list/object itself).
        evidence: ids of the tool passages that state the value, e.g. ["g3", "r1"] - only ids shown in this
        conversation are accepted; may be empty only for a fully-null value (nothing found after searching).
        key_evidence: for dict / list values, {"key or index": "id"} naming the passage behind each key or item.
        confidence: 0-1. Repeat calls merge (dict keys updated, list items appended).
        Answers with a WARNING when a value or item does not appear in the cited passages."""
        name = store.canonical(field)
        if name not in allowed:
            return f"Rejected: '{field}' is not a field of this group. Allowed: {allowed}"
        spec = store.specs[name]
        try:
            value = json.loads(value_json) if isinstance(value_json, str) else value_json
        except json.JSONDecodeError as e:
            if spec.root_type is not str:
                return f"Rejected: value_json is not valid JSON ({e}). Send it again as a JSON document."
            value = value_json  # a bare string for a str field is the value itself

        ids = dict.fromkeys(i.lower() for i in re.findall(r"\b[bgr]\d+\b", f"{evidence} {key_evidence}", re.I))
        if unknown := [i for i in ids if i not in shown]:
            return f"Rejected: unknown evidence id(s) {unknown}. Ids shown so far: {', '.join(shown) or 'none - call bm25_search / grep_lines / read_page first'}."
        cited = [shown[i] for i in ids]
        if not cited and not group.run_last and _scalars(value):
            return ("Rejected: no evidence ids. Cite the [b#]/[g#]/[r#] passages that state the value; empty evidence "
                    "is only allowed for a fully-null value.")
        source = ", ".join(dict.fromkeys(f"{s.doc}:{s.page}" for s in cited)) or ("synthesis" if group.run_last else "not found")

        warn = "" if value is None or spec.root_type is None or isinstance(value, spec.root_type) else \
            f" WARNING: datatype is `{spec.datatype}` but you sent a JSON {type(value).__name__}; stored anyway."
        haystack = "".join(_norm(s.text) for s in cited)  # soft check: warn (never reject) when a value is not in the cited text
        if unsupported := [f"{path}={text[:40]!r}" for path, text in _scalars(value) if cited and not any(f in haystack for f in _spellings(text))]:
            warn += f" WARNING: not found in the cited passages: {unsupported[:8]}{' ...' if len(unsupported) > 8 else ''} - re-check them or cite the passage that contains each."

        action = store.register(name, value, source, max(0.0, min(1.0, float(confidence))), verified=bool(cited),
                                evidence=[{**asdict(s), "text": s.text[:SNIPPET_KEEP_CHARS]} for s in cited], key_evidence=key_evidence)
        remaining = store.unfilled(group)
        return f"{action} '{name}' (source {source}).{warn} " + (f"Remaining: {remaining}" if remaining else "ALL FIELDS OF THIS GROUP ARE REGISTERED - stop.")

    return [bm25_search, grep_lines, read_page, register_record]


# =====================================================================================
# agent
# =====================================================================================

# predefined data related to the fields
GROUPS = [
  {
    "name": "Identification fields",
    "fields": [
      {
        "name": "Bid Number",
        "datatype": "str",
        "extracted": "The primary solicitation identifier exactly as printed by the issuing agency. Secondary IDs (internal sourcing number, portal reference number) are kept in a companion `list[str]` or appended in parentheses; the primary ID always comes first.",
        "guidance": "The agency's unique solicitation identifier. Found on the cover page, page headers/footers, addendum titles, and the portal listing. Agencies often have two numbers \u2014 a procurement number and a system/sourcing number \u2014 and portals add a third; capture all of them, with the primary one first."
      },
      {
        "name": "Title",
        "datatype": "str",
        "extracted": "The official solicitation name from the cover page, with portal-added decorations (sourcing numbers, asterisks, category tags) removed.",
        "guidance": "The official name of the solicitation, as printed on the cover page. Strip portal decorations like **SOURCING #168884**."
      },
      {
        "name": "company_name",
        "datatype": "str",
        "extracted": "The name of the issuing organization / buyer, including its common short form and jurisdiction.",
        "guidance": "The issuing organization (the buyer), not the vendor. Cover page letterhead, header block, or portal \"Issuing Organization\". Include the common short name and jurisdiction."
      },
      {
        "name": "contact_info",
        "datatype": "dict[str, str]",
        "extracted": "Key/value pairs for every contact point a vendor may need: buyer name, buyer email, buyer phone, procurement office phone/email, mailing or delivery address, portal URL, solicitation URL, and any specialised office (M/WBE, IT). Keys are short snake_case labels; values are the raw strings from the documents.",
        "guidance": "Everyone a vendor might need to reach: the named buyer/procurement officer (with email \u2014 usually the only permitted channel for questions), the general procurement office phone/email, mailing/delivery address for hard-copy submissions, portal URLs, and any specialized office (e.g., M/WBE, IT). Pull from the header block, submittal instructions, and portal page; note any \"no contact with other staff\" rule."
      }
    ],
    "run_last": False
  },
  {
    "name": "Schedule fields",
    "fields": [
      {
        "name": "Due Date",
        "datatype": "dict[str, datetime | str | None]",
        "extracted": "The controlling proposal deadline as a timezone-aware `datetime` under key `due`, the pre-amendment deadline under `original_due` (None if never changed), the published timezone label under `timezone`, and the document that set the final date under `source`.",
        "guidance": "The final date and time proposals must be received, with timezone. Start from the base RFP's schedule table, then check every addendum for an extension, then cross-check the portal's closing date. If the timezone differs between sources, confirm they are the same instant."
      },
      {
        "name": "Pre Bid Meeting",
        "datatype": "dict[str, datetime | bool | str | None]",
        "extracted": "Details of the pre-proposal conference and its Q&A window: `datetime` of the meeting, `mandatory` (True/False/None when not stated), `location_or_link`, `questions_due` datetime, and `answers_posted` datetime. All values None when no meeting is held.",
        "guidance": "Date, time, location or virtual link, and \u2014 critically \u2014 whether attendance is mandatory (non-attendance can disqualify). Also capture the related question deadline and when answers will be posted, since those form one \"Q&A window\". Found in the schedule table and any \"Pre-Proposal Conference\" paragraph."
      },
      {
        "name": "Delivery Date",
        "datatype": "dict[str, date | int | str | None]",
        "extracted": "The required delivery/completion timing in whichever form the documents give it: a hard `date`, a lead-time in `days_after_order` (int), or a free-text `notes` string for per-purchase-order / IDIQ arrangements, anticipated start of ordering, and related service-level clocks (repair turnaround, loaner devices).",
        "guidance": "When goods/services must be delivered or work completed. Look for: a hard date (\"deliver by\u2026\"), a duration (\"within 30 days ARO/after PO\"), or \u2014 for term/IDIQ contracts \u2014 a statement that deliveries are per-order. Q&A addenda frequently give the only real answer (\"when doyou need these?\"). Also capture service-level clocks tied to delivery (repair turnaround, hot-swap)."
      }
    ],
    "run_last": False
  },
  {
    "name": "Commercial / contractual fields",
    "fields": [
      {
        "name": "Bid Submission Type",
        "datatype": "dict[str, str | list[str] | int | None]",
        "extracted": "How the response must be delivered: `methods` (list of accepted channels such as portal, sealed hard copy, email), `portal_url`, `hard_copy_count` (int), `delivery_address`, `prohibited_methods` (list), and `procurement_method` (the solicitation type as written, e.g. \"RFP\", \"IFB\", \"CSP\").",
        "guidance": "How the response must be delivered: electronic portal (which one, registration lead time), sealed hard copy (number of copies, address,\"Attn:\" line), email, or a mix; explicitly note prohibited methods (no fax/email). Also record the procurement method (RFP, IFB, RFQ, CSP) since it governs whether negotiation is allowed. Found under \"Instructions to Offerors / Submittal\"."
      },
      {
        "name": "Term of Bid",
        "datatype": "dict[str, int | float | str | None]",
        "extracted": "The contract duration and renewal mechanics: `initial_term_years`, `renewal_count`, `renewal_term_years`, `max_total_years`, and a `notes` string for non-renewal notice periods, price-adjustment windows, firm-fixed-price periods, effective-date trigger, and any stated bid validity period.",
        "guidance": "Duration of the resulting contract: initial term, number and length of renewal options, maximum total, and any related mechanics (non-renewal notice period, price-adjustment windows, firm-fixed-price period, effective date trigger). Found in \"Term / Length of Contract\" and pricing sections. Distinguish this from the bid validity period (how long the offer must stay open) if the RFP states one."
      },
      {
        "name": "Payment Terms",
        "datatype": "dict[str, str | int | None]",
        "extracted": "Stated payment conditions: `terms` (raw text such as \"Net 30\" or None when deferred to general T&Cs), `net_days` (int when parseable), `funding_source`, and `notes` for pricing/payment mechanics that are stated (fixed price, no hidden fees, escalation rules, subcontractor pay-through, P-card acceptance).",
        "guidance": "How and when the vendor is paid: Net-30/45, milestone or progress payments, retainage, invoicing requirements, prompt-pay statutes, funding source, P-card acceptance. Often absent from the RFP body and deferred to the agency's general terms and conditions \u2014 say so, and capture whatever pricing/payment mechanics are stated (fixed price, no hidden fees, escalation rules, subcontractor pay-through)."
      },
      {
        "name": "Bid Bond Requirement",
        "datatype": "dict[str, bool | str | list[str] | None]",
        "extracted": "Whether bid security is required (`required`: True/False/None), its `amount` as written (percentage or fixed sum), any post-award `other_bonds` (performance, payment) as a list, and `insurance_notes` covering certificate-of-insurance requirements and their deadlines.",
        "guidance": "Whether a bid security (bond, cashier's check, letter of credit) must accompany the proposal, its amount (% of bid or fixed sum), and any post-award bonds (performance, payment) or insurance certificates and their deadlines. Search for \"bond\", \"surety\", \"security\", \"insurance\". Construction and large-service bids usually have them; commodity IT bids usually don't. If absent, say \"none specified\" and note related insurance obligations."
      },
      {
        "name": "Contract or Cooperative to use",
        "datatype": "dict[str, str | bool | list[str] | None]",
        "extracted": "The contract vehicle: `vehicle` (direct agency contract vs. purchase through an existing cooperative/piggyback contract), `cooperative_name` if one is named (BuyBoard, TIPS, Sourcewell, state contract, etc.), `others_may_ride` (bool) for interlocal/cooperative clauses, and `rider_programs` listing the named programs.",
        "guidance": "The contract vehicle: a direct agency contract, or purchase via an existing cooperative/piggyback contract (BuyBoard, TIPS, Sourcewell,OMNIA, NASPO, state contracts, GSA). Also note whether other entities may ride the resulting contract (interlocal/cooperative clauses) \u2014 that expands the vendor's market. Found in \"Cooperative Purchasing / Interlocal Agreement\" attributes and the portal's \"Piggyback\" flag."
      },
      {
        "name": "Installation",
        "datatype": "dict[str, bool | list[str] | str | None]",
        "extracted": "Whether work beyond shipping is in scope: `required` (True/False/None), `services` as a list of the enumerated tasks (install, configure, image/enroll, mount, asset-tag, etch, train, disposal, white-glove delivery), and `notes` for scheduling/coordination rules and addendum clarifications.",
        "guidance": "Whether the vendor must do more than ship: install, configure, image/enroll, mount, asset-tag, train, dispose of old equipment, or provide \"white-glove\" deployment; and who coordinates scheduling. Found in scope/statement-of-work, deployment sections, and Q&A. State \"not required\" explicitly when the bid is deliver-only."
      }
    ],
    "run_last": False
  },
  {
    "name": "Compliance field",
    "fields": [
      {
        "name": "Any Additional Documentation Required",
        "datatype": "list[dict[str, str | bool]]",
        "extracted": "One entry per form or attachment that must be returned besides the price sheet, each with `name`, `mandatory` (bool), `timing` (\"withproposal\", \"post award\", or \"conditional\"), and `source` (the section or addendum that requires it).",
        "guidance": "Everything that must be returned besides the price sheet. Assemble it from several places: the attachments list, every \"must be submitted with\u2026\" sentence, signature/certification pages, addenda acknowledgment instructions, reference and subcontractor forms, and Q&A answers that move aform's timing (e.g., Form 1295 \"with the response\"). Separate submit-with-proposal items from post-award items (insurance certificates) and mark optional ones. Missing any mandatory item typically renders a bid non-responsive, so err on the side of completeness."
      }
    ],
    "run_last": False
  },
  {
    "name": "Product fields",
    "fields": [
      {
        "name": "Product",
        "datatype": "list[dict[str, str | int | None]]",
        "extracted": "One entry per price-schedule line item: `line` (line number as written), `name`, `unit`, `estimated_quantity` (int or None), and `notes` (e.g. category, whether it is a service/warranty/discount line rather than goods). Accessories mentioned only in scope text are added as entries with `line` = None.",
        "guidance": "The list of what is being bought, ideally mirroring the price schedule: line number, item name, unit, and estimated quantity, plus ancillary lines (warranty, services, subscriptions, discount %) and any accessories mentioned in scope. Quantities may appear only in a later \"evaluation\" section or attached spreadsheet \u2014 read the whole price schedule."
      },
      {
        "name": "Product Specification",
        "datatype": "dict[str, list[str]]",
        "extracted": "Mapping from product/tier name to an ordered list of its minimum requirement statements, kept close to the source wording (CPU, display, memory, storage, wireless, ports, OS, battery, warranty...). A \"General Requirements\" key holds rules that apply to all items (new/current-model, OEM parts, warranty handling). Addendum clarifications are appended inline with attribution.",
        "guidance": "The minimum technical requirements per product, kept as close to the source wording as possible, grouped by line/tier. Fold in Q&A clarifications inline with attribution (e.g., \"USB 3.1 minimum; 3.2 permissible per Addendum 1\"). Include any general equipment rules (new, current-model, OEM parts, warranty length, environmental/compliance standards)."
      },
      {
        "name": "MFG for Registration",
        "datatype": "dict[str, bool | list[str] | str | None]",
        "extracted": "Manufacturer expectations: `manufacturers` (named brands, empty list when none), `brand_agnostic` (bool), `reseller_requirements` (authorized-reseller, OEM warranty, deal-registration conditions), `platforms_referenced` (chipsets/OS named in specs when no brand is mandated), and `registration_notes` for the buyer's own vendor-registration requirement.",
        "guidance": "The manufacturer(s) the buyer expects \u2014 either a named brand/\"or equal\", a brand-restricted list, or brand-agnostic requirements\u2014 and any manufacturer relationship the vendor must hold (authorized reseller, OEM warranty, deal registration). When no brand is named, record the referenced platforms (chipsets, OS) and the reseller/OEM conditions, and note the buyer's own vendor-registration requirement."
      },
      {
        "name": "Model_no",
        "datatype": "list[str]  (one list each for Model_no and Part_no)",
        "extracted": "Buyer-specified manufacturer model numbers and part numbers/SKUs, one string per value. An empty list means the buyer named none; the accompanying notes should state explicitly when the vendor is instead required to propose make/model per line.",
        "guidance": "Specific manufacturer model numbers or SKUs, if the buyer names them (common in \"or equal\" bids or replacement-part purchases). Many RFPs instead ask the vendor to propose make/model \u2014 record that explicitly rather than leaving the field empty."
      },
      {
        "name": "Part_no",
        "datatype": "list[str]  (one list each for Model_no and Part_no)",
        "extracted": "Buyer-specified manufacturer model numbers and part numbers/SKUs, one string per value. An empty list means the buyer named none; the accompanying notes should state explicitly when the vendor is instead required to propose make/model per line.",
        "guidance": "Specific manufacturer model numbers or SKUs, if the buyer names them (common in \"or equal\" bids or replacement-part purchases). Many RFPs instead ask the vendor to propose make/model \u2014 record that explicitly rather than leaving the field empty."
      }
    ],
    "run_last": False
  },
  {
    "name": "Synthesis field",
    "fields": [
      {
        "name": "Bid Summary",
        "datatype": "str",
        "extracted": "A single-paragraph executive overview composed from the other fields: who is buying what, scale, contract length and funding, award approach, key service obligations, final due date (with any extension), evaluation weights, anticipated award date, and the most material addendum changes.",
        "guidance": "A one-paragraph executive overview a bid manager can read in 30 seconds: who is buying what, scale/quantities, contract length and funding, award approach (single/multiple, by line/package), key service obligations, due date (with any extension), evaluation weights, anticipated award date, and the most material addendum changes. Written last, after all other fields are settled, and only from facts already captured above."
      }
    ],
    "run_last": True
  }
]

# initializes the predefined Groups in a dataclass
def field_groups() -> list[FieldGroup]:
    """The field groups from GROUPS as dataclasses (FieldSpec per field; the Synthesis group runs last)."""
    return [FieldGroup(g["name"], [FieldSpec(**f) for f in g["fields"]], g["run_last"]) for g in GROUPS]


# return the field information in markdown format.
def _fields_spec_md(fields: list[FieldSpec]) -> str:
    return "\n\n".join(f"### {f.name}\n- Datatype: `{f.datatype}`\n- What is extracted: {f.extracted}\n- Guidance: {f.guidance}" for f in fields)


# Batches of pages for long PDFs
def _batches(doc: Document) -> list[list[tuple[str, str]]]:
    """Group a document's pages into batches of <= MAX_CHARS_PER_BATCH chars; oversize pages are split into parts."""
    parts = [(f"Page {p.page_no}" + (f" (part {k})" if len(p.text) > MAX_CHARS_PER_BATCH else ""), p.text[i:i + MAX_CHARS_PER_BATCH])
             for p in doc.pages for k, i in enumerate(range(0, len(p.text), MAX_CHARS_PER_BATCH), 1)]
    batches: list[list[tuple[str, str]]] = [[]]
    for part in parts:
        if batches[-1] and sum(len(t) for _, t in batches[-1]) + len(part[1]) > MAX_CHARS_PER_BATCH:
            batches.append([])
        batches[-1].append(part)
    return [b for b in batches if b]


# Generate summary & signal for the Corpus
# For each document:
    # For each batch:
        # generate summary and signal
        # summaries.append(gen_summary)
        # signals.extend(gen_summary)
    # merge_summary(summaries)
# return summary and signals
def build_summary_and_signals(corpus: Corpus, llm, settings: Settings, tracker: TokenTracker) -> str:
    """Step 1: per-batch LLM calls -> per-document summary + page signals -> one markdown for all agents (cached).

    Every LLM call (batch, per-document merge, bid-level merge) is printed with its token usage."""
    from langchain_core.messages import HumanMessage, SystemMessage

    # cache name carries a hash of the document list, so adding, removing or renaming a file (which renumbers docN ids) rebuilds it
    cache = settings.bid_out_dir / f"summary_signals-{hashlib.md5(corpus.index_markdown().encode()).hexdigest()[:8]}.md"
    if cache.exists() and not settings.no_cache:
        print(f"  using cached {cache}")
        return cache.read_text(encoding="utf-8")
    stage = tracker.mark()

    def ask(label: str, *msgs) -> str:
        call = tracker.mark()
        reply = llm.invoke(list(msgs), config={"callbacks": [tracker]}).content.strip()
        print(f"  {label}: {tracker.since(call)}")
        return reply

    def merge(summaries: list[str], scope: str) -> str:
        if len(summaries) == 1:
            return summaries[0]
        return ask(f"merge {scope}", HumanMessage(MERGE_SUMMARIES_PROMPT.format(scope=scope, summaries="\n\n---\n\n".join(summaries))))

    def _split_reply(reply: str) -> tuple[str, str]:
        """Split the LLM answer into (summary, page signals)."""
        parts = re.split(r"^##\s*Page Signals\s*$", reply, maxsplit=1, flags=re.M | re.I)
        summary = re.sub(r"^##\s*Summary\s*$", "", parts[0], flags=re.M | re.I).strip() if len(parts) == 2 else ""
        return summary, parts[-1].strip()

    sections, doc_summaries = [], []
    for doc in corpus.documents:
        batches = _batches(doc)
        summaries, signals = [], []
        for i, batch in enumerate(batches, 1):
            body = f"DOCUMENT: {doc.name}\n\n" + "\n\n".join(f"===== {label} =====\n{text}" for label, text in batch)
            reply = ask(f"{doc.id} batch {i}/{len(batches)} ({batch[0][0]} .. {batch[-1][0]})",
                        SystemMessage(SUMMARY_SIGNALS_SYSTEM), HumanMessage(body))
            summary, page_signals = _split_reply(reply)
            summaries.append(summary)
            signals.append(page_signals)
        doc_summary = merge([s for s in summaries if s] or ["(no summary)"], doc.id)
        doc_summaries.append(f"{doc.name}: {doc_summary}")
        sections.append(f"# {doc.id}: {doc.name}\n## Summary\n{doc_summary}\n\n## Page signals\n" + "\n\n".join(s for s in signals if s))

    md = f"# Bid summary\n{merge(doc_summaries, 'bid')}\n\n" + "\n\n".join(sections)
    settings.bid_out_dir.mkdir(parents=True, exist_ok=True)
    cache.write_text(md, encoding="utf-8")
    print(f"  summary & signals total: {tracker.since(stage)}")
    return md


# Langchain middleware for SummarizationMiddleware, CompletionMiddleware, MaxTurnsMiddleware
# SummarizationMiddleware - compaction,
# CompletionMiddleware - successful completion,
# MaxTurnsMiddleware - max turn complete
def _middlewares(store: RecordStore, group: FieldGroup, settings: Settings, summary_llm) -> list:
    """History compression, stop-when-complete, and the max-turn 'register what you have' behaviour."""
    from langchain.agents.middleware import AgentMiddleware, SummarizationMiddleware, hook_config
    from langchain_core.messages import HumanMessage

    class CompletionMiddleware(AgentMiddleware):
        @hook_config(can_jump_to=["end"])
        def before_model(self, state, runtime):
            return None if store.unfilled(group) else {"jump_to": "end"}

    class MaxTurnsMiddleware(AgentMiddleware):
        injected = False

        @hook_config(can_jump_to=["end"])
        def before_model(self, state, runtime):
            if store.turn >= settings.max_turns and not self.injected:
                self.injected = True
                print(f"  max turns reached; asking for remaining fields {store.unfilled(group)}")
                return {"messages": [HumanMessage(FINAL_TURN_MESSAGE.format(fields=store.unfilled(group)))]}
            return {"jump_to": "end"} if store.turn > settings.max_turns else None

        def after_model(self, state, runtime):
            store.turn += 1

    return [SummarizationMiddleware(model=summary_llm, trigger=("tokens", settings.max_tokens_before_compression),
                                    keep=("messages", 6), summary_prompt=COMPRESSION_PROMPT),
            CompletionMiddleware(), MaxTurnsMiddleware()]


# Field agent extracts all the fields in a particular fieldGroup and saves the results and evidence in RecordStore.
def run_field_agent(group: FieldGroup, llm, summary_llm, corpus: Corpus, store: RecordStore, summary_md: str,
                    settings: Settings, tracker: TokenTracker) -> None:
    """Step 2 for ONE field group: build its agent, run the tool loop, and guarantee every field is registered.

    Values are written only through the register_record tool. The loop stops when CompletionMiddleware sees the
    group complete, when MaxTurnsMiddleware hits the turn limit, or when the model stops calling tools. Anything
    still unfilled afterwards is registered as None so the output JSON always contains every field.
    """
    from langchain.agents import create_agent
    from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

    print(f"\n== {group.name}: {[f.name for f in group.fields]}")
    store.turn = 0  # per-group turn counter, advanced by MaxTurnsMiddleware.after_model

    # The Synthesis group writes its field from the other groups' results, so they go into its system prompt.
    prior = PRIOR_RESULTS_BLOCK.format(results=json.dumps(store.fields_json(), indent=2, ensure_ascii=False)) if group.run_last else ""
    system = FIELD_AGENT_SYSTEM.format(group=group.name, doc_index=corpus.index_markdown(),
                                       fields_spec=_fields_spec_md(group.fields), prior=prior, summary=summary_md)
    agent = create_agent(model=llm, tools=make_tools(corpus, store, group), system_prompt=system,
                         middleware=_middlewares(store, group, settings, summary_llm))
    start = HumanMessage(f"Extract the fields for '{group.name}'. Fields to be filled: {store.unfilled(group)}. Begin.")


    # Print Model turn as it happens.
    # Every turn costs several graph steps (middleware + model + tools), hence the generous recursion_limit. 
    # The tracker callback propagates to every LLM call inside the graph, so compression calls are counted too; `last` marks where the previous print stopped.
    group_mark = last = tracker.mark()
    try:
        for update in agent.stream({"messages": [start]}, stream_mode="updates",
                                   config={"recursion_limit": settings.max_turns * 6 + 30, "callbacks": [tracker]}):
            for node, payload in update.items():
                messages = payload.get("messages") or [] if isinstance(payload, dict) else []
                if node.startswith("SummarizationMiddleware") and messages:  # the node only emits messages when it compressed
                    print(f"  (conversation history compressed: {tracker.since(last)})")
                    last = tracker.mark()
                elif node == "model":  # one line per model turn: its token usage and the tool calls it made
                    for m in (m for m in messages if isinstance(m, AIMessage)):
                        calls = ", ".join(f"{c['name']}({json.dumps(c['args'], ensure_ascii=False)[:300 if c['name'] == 'register_record' else 150]})" for c in m.tool_calls)
                        print(f"  turn {store.turn + 1} [{tracker.since(last)}]: {calls or str(m.content)[:150]}")
                        last = tracker.mark()
                elif node == "tools":  # show what got recorded
                    for m in (m for m in messages if isinstance(m, ToolMessage) and m.name == "register_record"):
                        print(f"    {str(m.content)[:220]}")
    except Exception as e:  # auth / rate-limit / model errors: report and let the remaining groups run
        print(f"  ERROR in group '{group.name}': {type(e).__name__}: {e}")

    for name in store.unfilled(group):  # never leave a hole in the output
        store.register(name, None, "not found (agent did not register)", 0.0, verified=False, evidence=[], key_evidence=None)
    print(f"  done in {store.turn} turns | {group.name} {tracker.since(group_mark)}")


# Corpus = loadDocs()
# corpSummary = build_summary_and_signals()
# groups = loadGroups()
# storage = RecordStore(groups)
# For group in groups:
    # run_field_agent(storage, group)
# return store.fields, store.evidence
def run_pipeline(settings: Settings) -> None:
    """Run the whole extraction for one bid folder and write the two output files.

    Order: ingest documents -> parse field groups -> summary & signals (one LLM pass over all pages, cached)
    -> one agent per group with the Synthesis group last (it needs the others' results) -> write
    <out>.json (field -> value only) and <out dir>/<Bid>/<Bid>.evidence.json (per-field registration history + run metadata).
    """
    print(f"Processing documents in {settings.bid_dir}")
    corpus = load_bid_folder(settings.bid_dir)
    groups = field_groups()
    llm, summary_llm = build_llm(), build_llm(summary=True)  # same deployment unless AZURE_OPENAI_SUMMARY_DEPLOYMENT is filled in
    tracker = TokenTracker()  # one counter for the whole run; stages report their share via mark()/since()

    print("Generating summary and signals")
    summary_md = build_summary_and_signals(corpus, summary_llm, settings, tracker)

    print("Extracting fields")
    store = RecordStore(groups)  # one store shared by all agents; each agent may only register its own group's fields
    for group in sorted(groups, key=lambda g: g.run_last):  # stable sort: original order, Synthesis group last
        run_field_agent(group, llm, summary_llm, corpus, store, summary_md, settings, tracker)

    # Values and provenance are written to separate files so the main JSON stays a clean field -> value map.
    settings.bid_out_dir.mkdir(parents=True, exist_ok=True)  # also creates output/ for <BidName>.json
    settings.out_path.write_text(json.dumps(store.fields_json(), indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    evidence = {"bid": settings.bid_dir.name, "generated_at": dt.datetime.now().isoformat(timespec="seconds"),
                "model": AZURE_OPENAI_DEPLOYMENT,
                "documents": [{"id": d.id, "name": d.name, "kind": d.kind, "pages": len(d.pages)} for d in corpus.documents],
                "token_usage": {"input": tracker.input, "output": tracker.output, "llm_calls": tracker.calls},
                "fields": dict(store.evidence)}  # field -> [{turn, action, value, source, confidence, timestamp}, ...]
    settings.evidence_path.write_text(json.dumps(evidence, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"\nTotal token usage: {tracker.total()}")
    print(f"Wrote {settings.out_path}\nWrote {settings.evidence_path}")


# =====================================================================================
# main
# =====================================================================================

def main(argv: list[str] | None = None) -> int:
    """Command-line entry point: parse arguments into Settings and run the pipeline.

    Returns a process exit code: 0 on success, 2 for input errors (missing or empty bid folder).
    """
    ap = argparse.ArgumentParser(description="Extract structured RFP information from a bid folder with LLM agents.")
    ap.add_argument("bid_dir", type=Path, help="folder with the bid's PDF/HTML documents, e.g. task/Bid1")
    ap.add_argument("--out", type=Path, help="output JSON path (default output/<BidName>.json)")
    ap.add_argument("--max-turns", type=int, default=24, help="model turns before the agent is told to register what it has")
    ap.add_argument("--max-tokens-before-compression", type=int, default=60_000, help="history size that triggers compression")
    ap.add_argument("--no-cache", action="store_true", help="rebuild summary & signals even if cached")
    a = ap.parse_args(argv)

    bid_dir = a.bid_dir.resolve()  # resolved so the folder name (used for output file names) is reliable
    if not bid_dir.is_dir():
        ap.error(f"{bid_dir} is not a directory")
    settings = Settings(bid_dir, a.out or Path("output") / f"{bid_dir.name}.json", a.max_turns,
                        a.max_tokens_before_compression, a.no_cache)
    try:
        run_pipeline(settings)
    except (RuntimeError, FileNotFoundError) as e:  # expected user-facing failures; no traceback
        print(f"error: {e}")
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
