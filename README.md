# infoExtract — RFP information extraction with LLM agents

`pipeline.py` reads every document of one bid (RFP PDFs, addenda, bid-portal HTML pages) and fills a fixed set of
structured fields — due date, submission method, contract term, bond requirements, products, specifications, and so on.
It uses Azure AI Foundry (Azure OpenAI) models through LangChain: one pass summarises the documents, then one agent per
field group searches the documents with BM25 / grep / page-read tools and registers the values it finds.

```
infoExtract/
├── pipeline.py          the whole pipeline (CONFIG | prompt | doc_ingestion | tools | agent | main)
├── task/<BidName>/      the bid's PDF / HTML documents                                   <- you provide
├── output/<BidName>.json           extracted fields (field -> value)
├── output/<BidName>/<BidName>.evidence.json   how each value was obtained (source pages, cited snippets, confidence, turn, token usage)
├── output/<BidName>/summary_signals-<hash>.md   cached summary & page signals (hash of the document list)
└── requirements.txt
```

## Getting started

### Step 0 — put the inputs in place

1. Create a folder for the bid under `task/` and paste all of its documents into it, e.g. `task/Bid1/`.
   PDFs and HTML files are picked up; anything else is ignored. The folder name becomes the output name.

### Step 1 — Python environment

```powershell
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
```

### Step 2 — Azure OpenAI credentials

Fill in the `AZURE_OPENAI_*` constants at the top of `pipeline.py` (CONFIG section) from Azure AI Foundry → your
project → **Deployments** → the deployment:

| Constant | Value |
|---|---|
| `AZURE_OPENAI_ENDPOINT` | the resource's Azure OpenAI endpoint, `https://<resource>.openai.azure.com/` (not the `/api/projects/...` URL) |
| `AZURE_OPENAI_API_KEY` | key from the resource's *Keys and Endpoint* |
| `AZURE_OPENAI_API_VERSION` | a REST API version such as `2024-12-01-preview` (not the model version) |
| `AZURE_OPENAI_DEPLOYMENT` | the deployment **name** (e.g. `gpt-4.1-mini`); it must support tool calling |
| `AZURE_OPENAI_SUMMARY_DEPLOYMENT` | optional cheaper deployment for the summary step and history compression |

### Step 3 — run

```powershell
.venv\Scripts\python pipeline.py task\Bid1
```

| Option | Default | Meaning |
|---|---|---|
| `--out PATH` | `output/<BidName>.json` | where to write the fields; the evidence file and cache go in `<out dir>/<BidName>/` |
| `--max-turns N` | 24 | model turns per agent before it is told to register whatever it has |
| `--max-tokens-before-compression N` | 60000 | conversation size that triggers history compression |
| `--no-cache` | off | rebuild the summary & signals instead of reusing the cached `summary_signals-<hash>.md` (it is rebuilt anyway when files are added, removed or renamed) |

The console shows each stage, one line per LLM call in the summary stage, one line per agent turn with its token
usage, a token total per field group, and the total input/output tokens at the end.

## How the fields are filled

`GROUPS` in `pipeline.py` defines six **field groups**. Each group is handled by its own agent, which sees the document list, the
summary & page signals of all documents, and the specification of only its own fields. The agent searches with
`bm25_search` (one document, optional page; documents are tokenised with spaCy's blank English tokenizer and its
stop-word list, no language model needed), `grep_lines` and `read_page`. Every passage these tools return is
labelled with an **evidence id** - `[b1]`, `[b2]`, ... for bm25 passages, `[g1]`, ... for grep hits, `[r1]`, ... for
pages read - and `register_record` requires those ids: a value is only stored when it cites ids that were shown to
that agent (`evidence`, plus an optional per-key `key_evidence` map for dict/list values); the source pages are
derived from the cited passages. Calls that cite no id, or an id that was never shown, are rejected, so the
summary & page signals in the prompt can never serve as the source. When a value (or list item) does not appear in
the cited passages the tool stores it but answers with a warning naming the unsupported parts. Repeated
registrations merge (dict keys are updated, list items appended, scalars replaced). A group's loop ends as soon as
all its fields are registered. The *Synthesis* group runs last, receives every other group's results and needs no ids.

Where a value is not stated in the documents the agent still registers the field: `null` sub-fields plus a `notes`
key for dicts, `[]` for lists, and a short "not specified" string for `str` fields; only such fully-null values may
be registered with empty evidence (source `not found`). Source pages, the cited snippets, `key_evidence`, a
`verified` flag and confidence never appear in `output/<BidName>.json`; they are kept in
`output/<BidName>/<BidName>.evidence.json`.

### Identification fields

Who issued the bid and how it is identified — the values used to file, reference and contact. Mostly single strings
taken from the cover page, header block and portal listing.

| Field | Datatype | Sub-fields |
|---|---|---|
| `Bid Number` | `str` | — primary solicitation ID first; secondary IDs (sourcing / portal reference numbers) appended in parentheses |
| `Title` | `str` | — |
| `company_name` | `str` | — issuing organization with short name and jurisdiction |
| `contact_info` | `dict[str, str]` | free keys such as `buyer_name`, `buyer_email`, `buyer_phone`, `procurement_office_phone`, `procurement_office_email`, `mailing_address`, `portal_url`, `specialized_office` |

### Schedule fields

Every date a vendor must act on. Grouped because they come from the same schedule table and are all changed by
addenda; the agent is told to check addenda before the base RFP.

| Field | Datatype | Sub-fields |
|---|---|---|
| `Due Date` | `dict[str, datetime \| str \| None]` | `due` (controlling deadline, timezone-aware), `original_due` (pre-amendment deadline or `null`), `timezone`, `source` (document that set the final date) |
| `Pre Bid Meeting` | `dict[str, datetime \| bool \| str \| None]` | `datetime`, `mandatory`, `location_or_link`, `questions_due`, `answers_posted` |
| `Delivery Date` | `dict[str, date \| int \| str \| None]` | `date` (hard date) or `days_after_order` (lead time), `notes` (per-PO / IDIQ arrangements, service-level clocks) |

### Commercial / contractual fields

The terms of doing business: how to submit, how long the contract runs, how the vendor is paid, what security is
required, which contract vehicle applies, and whether work beyond delivery is in scope. Each is a dict so that the
stated parts and the "not stated" parts can be told apart.

| Field | Datatype | Sub-fields |
|---|---|---|
| `Bid Submission Type` | `dict[str, str \| list[str] \| int \| None]` | `methods` (list), `portal_url`, `hard_copy_count`, `delivery_address`, `prohibited_methods` (list), `procurement_method` (RFP / IFB / RFQ / CSP …) |
| `Term of Bid` | `dict[str, int \| float \| str \| None]` | `initial_term_years`, `renewal_count`, `renewal_term_years`, `max_total_years`, `notes` (notice periods, price-adjustment windows, effective-date trigger) |
| `Payment Terms` | `dict[str, str \| int \| None]` | `terms` (raw text or `null`), `net_days`, `funding_source`, `notes` (fixed price, escalation, pay-through, P-card) |
| `Bid Bond Requirement` | `dict[str, bool \| str \| list[str] \| None]` | `required`, `amount`, `other_bonds` (list: performance / payment), `insurance_notes` |
| `Contract or Cooperative to use` | `dict[str, str \| bool \| list[str] \| None]` | `vehicle`, `cooperative_name`, `others_may_ride`, `rider_programs` (list) |
| `Installation` | `dict[str, bool \| list[str] \| str \| None]` | `required`, `services` (list: install, configure, image/enroll, mount, asset-tag, etch, train, disposal …), `notes` |

### Compliance field

Everything that has to be returned with (or after) the proposal besides the price sheet. Kept as its own group
because the items are scattered across attachment lists, certification pages, addendum instructions and Q&A
answers, and missing one usually makes a bid non-responsive.

| Field | Datatype | Sub-fields (per list item) |
|---|---|---|
| `Any Additional Documentation Required` | `list[dict[str, str \| bool]]` | `name`, `mandatory`, `timing` (`with proposal` / `post award` / `conditional`), `source` |

### Product fields

What is being bought and to what specification. Grouped because they all come from the scope / specification
sections and the price schedule, and because manufacturer, model and part information are read together.

| Field | Datatype | Sub-fields |
|---|---|---|
| `Product` | `list[dict[str, str \| int \| None]]` | per line item: `line`, `name`, `unit`, `estimated_quantity`, `notes` (category, service/warranty/discount line) |
| `Product Specification` | `dict[str, list[str]]` | product/tier name → ordered list of minimum requirement statements; a `General Requirements` key holds rules for all items |
| `MFG for Registration` | `dict[str, bool \| list[str] \| str \| None]` | `manufacturers` (list, empty if none named), `brand_agnostic`, `reseller_requirements` (list), `platforms_referenced` (list), `registration_notes` |
| `Model_no` | `list[str]` | buyer-specified model numbers; `[]` when the vendor proposes make/model |
| `Part_no` | `list[str]` | buyer-specified part numbers / SKUs; `[]` when none |

### Synthesis field

Written last, from the other groups' registered values rather than from fresh searching, so it is consistent with
the rest of the output.

| Field | Datatype | Sub-fields |
|---|---|---|
| `Bid Summary` | `str` | — one paragraph: who is buying what, scale, term and funding, award approach, key service obligations, final due date, evaluation weights, anticipated award, material addendum changes |

## Changing the fields

Edit `GROUPS` in `pipeline.py` (section *agent*). Each entry is a group with a `name`, a list of `fields` and a `run_last` flag; each field has `name`, `datatype` (Python annotation form, its outer name decides the JSON root type), `extracted` (what to record) and `guidance` (where to look). The group with `run_last: True` runs after the others and receives their results.
