# Company Brain mini-POC

A small, runnable prototype of a **"Company Brain"** context layer. The pipeline
turns email, SharePoint documents, CRM rows and meeting transcripts into **time-stamped relationships**
between companies, contacts, deals and lenders. It then answers deal questions where **every line of
the answer carries a source reference**.

It runs locally for $0 on synthetic data. Each piece maps to a Databricks component (see
[Mapping to Databricks](#mapping-to-databricks)).

## Findings in brief

1. **The five deal questions can be answered with traceable evidence** once relationships are extracted
   into a simple edge table. For example, the financing request is tracked as $12M → $15M → $14M, and each
   step cites the email or document it came from.
2. **Search alone is not enough for relationship questions.** *"Which bank said no to Northwind?"* misses
   in keyword, vector and hybrid search alike, because the email says *"we will pass"*. The extracted
   `Larkspur Bank → REVIEWED → declined` edge answers it directly.

## Run it

```bash
pip install -r requirements.txt     # sentence-transformers + numpy, for hybrid search only
python brain.py                     # answers the five questions for deal D-101 (no model needed)
python brain.py search "Dayton warehouse appraisal"
python brain.py eval                # retrieval back-test: keyword vs vector vs hybrid
python test_brain.py                # 6 checks
```

The first `search`/`eval` run downloads `BAAI/bge-small-en-v1.5` (~130 MB), which runs on CPU.

## The sample

Fictional data (companies, people and `.example` email domains) about an advisory firm running a
debt raise:

| Source | Stands in for | Contents |
|---|---|---|
| `data/emails.jsonl` | Outlook via Lakeflow Connect | 9 emails, including one unrelated deal as noise |
| `data/docs/*.md` | SharePoint files | CIM v1 ($12M ask) and term sheet request v2 ($15M ask) |
| `data/transcripts/*.txt` | Transcripts in GCS | One lender call |
| `data/crm.json` | CRM export | Companies, lenders, contacts, deals, past lender outcomes |

## The five questions, answered

| Question | Answered from | Result |
|---|---|---|
| What is the latest on the deal? | Most recent deal-linked sources + current state | Request lowered to $14M (Oct 1), Quillfeather indication of interest (Sep 29), Basaltline engaged (Sep 22) |
| Who has been involved? | `PARTICIPATED_IN` edges | 6 people with org, title, first/last seen and sources. The unrelated Fabrikam email does not leak in |
| What is the current financing request, and how has it changed? | `FINANCING_REQUEST` edges over time | $12M → $15M → $14M. A document restating the same amount counts as more evidence, not a change |
| Which lenders have reviewed the opportunity? | `REVIEWED {status}` edges, latest per lender | Larkspur declined · Quillfeather indication of interest · Basaltline reviewing, each with full status history |
| Which lenders appear relevant based on prior activity? | CRM lender outcomes on same-industry deals, minus lenders already engaged | Tidewren (funded a logistics deal). Larkspur is flagged because it declined one |

Example output (`python brain.py`, abridged):

```json
"history": [
  {"amount_musd": 12.0, "as_of": "2026-08-04T09:12:00Z",
   "sources": ["outlook://msg-001", "sharepoint://sites/Deals/Northwind/Northwind_CIM_v1.docx"]},
  {"amount_musd": 15.0, "as_of": "2026-09-02T16:20:00Z",
   "sources": ["outlook://msg-004", "sharepoint://sites/Deals/Northwind/Northwind_Term_Sheet_Request_v2.docx"]},
  {"amount_musd": 14.0, "as_of": "2026-10-01T09:02:00Z", "sources": ["outlook://msg-009"]}
]
```

## How it works

```
sources ──► normalized.documents ──► graph.edges ───────────────┐
            (doc_id, source,          (src, rel, dst, value,     ├──► answers, each line with source_ref
             source_ref, ts, text)     ts, source_ref)           │
                    └──────────► hybrid index (BM25 + vectors) ──┘    (search supplies evidence passages)
```

- **Normalize:** every row keeps `source_ref` and `ts`, so any answer can point back to an exact message or file.
- **Extract:** produces edges such as `PARTICIPATED_IN`, `FINANCING_REQUEST {amount}`, `REVIEWED {status}` and `MENTIONS`, each with a timestamp and source. "Changes over time" means sorting edges by `ts`; the current state is the latest edge.
- **Resolve entities:** names, aliases ("Quillfeather" → Quillfeather Credit) and email domains map to one CRM id, so the graph doesn't split.
- **Retrieve:** BM25 and `bge-small` embeddings, fused with Reciprocal Rank Fusion.

## Retrieval back-test

10 queries over 12 documents: exact tokens, keywords, entity names and paraphrases.

| | hit@1 | hit@3 | MRR |
|---|---|---|---|
| Keyword (BM25) | 0.6 | 0.9 | 0.742 |
| Vector (bge-small) | 0.4 | 0.5 | 0.546 |
| Hybrid (RRF) | 0.7 | 0.9 | 0.808 |

**Read with care:** 12 documents is far too few to rank retrievers. Renaming the fictional companies
alone moved the vector scores. What stays stable is the qualitative result: paraphrased relationship
questions ("which bank said no") fail in every mode, and the structured edge answers them. The harness
(`brain.py eval`) is the part that carries over to a larger corpus.

## Mapping to Databricks

Port sketch; not yet run on Databricks.

| Here | On Databricks |
|---|---|
| `data/` sources | Lakeflow Connect (SharePoint, Outlook); GCS sample in a UC volume; CRM CSV in a volume |
| `load()` → documents | `company_brain.normalized.documents` Delta table; `ai_parse_document` for files |
| `extract()` | `ai_query(model, prompt, responseFormat => <edge schema>)` → `company_brain.graph.edges` |
| `BM25` + `Vectors` + `rrf()` | AI Search Delta Sync index, `query_type="HYBRID"` on a standard endpoint |
| Answer functions | SQL views over `graph.edges`, exposed through a Genie Agent; AI Search for evidence passages |

## Known limits of this prototype

- The extractor is regex and alias rules, proven only on this sample. On Databricks it becomes an LLM call with a fixed JSON schema; the edge table stays the same.
- One chunk per document. Real email threads need chunking and quoted-reply stripping.
- No stemming in BM25, so "passed" does not match "pass".
- Entity resolution is aliases plus email domains only.

## Files

```
brain.py        pipeline, five answer functions, retrieval back-test
test_brain.py   6 checks: change-over-time, latest lender status, relevant lenders, no cross-deal leak,
                every edge traces to a real source, retrieval regression guard
data/           synthetic sample
```
