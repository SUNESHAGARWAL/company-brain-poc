"""Company Brain mini-POC.

raw sources -> normalized documents -> entity/relationship edges with timestamps
-> hybrid retrieval (BM25 + vectors, RRF) -> answers where every claim carries a source_ref.

The three in-memory tables mirror the Unity Catalog layout proposed for Databricks:
  normalized.documents   (doc_id, source, source_ref, ts, title, text, people)
  graph.edges            (src, rel, dst, value, ts, source_ref)
  crm.*                  (companies, lenders, contacts, deals, deal_lender_history)

Usage:
  python brain.py                 # answer the five client questions for deal D-101
  python brain.py search "query"  # hybrid search with per-retriever ranks
  python brain.py eval            # retrieval back-test: BM25 vs vector vs hybrid
"""
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path

DATA = Path(__file__).parent / "data"
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


# ---------- normalize: every row keeps source_ref + ts ----------

def load():
    crm = json.loads((DATA / "crm.json").read_text())
    docs = []
    for line in (DATA / "emails.jsonl").read_text().splitlines():
        m = json.loads(line)
        docs.append(dict(doc_id=m["id"], source="outlook", source_ref=f"outlook://{m['id']}", ts=m["ts"],
                         title=m["subject"], people=[m["from"], *m["to"], *m["cc"]], text=m["body"]))
    for p in sorted((DATA / "docs").glob("*.md")):
        _, meta, body = p.read_text().split("---\n", 2)
        meta = dict(line.split(": ", 1) for line in meta.strip().splitlines())
        docs.append(dict(doc_id=p.stem, source="sharepoint", source_ref=f"sharepoint:/{meta['sharepoint_path']}",
                         ts=meta["modified"], title=body.strip().splitlines()[0].lstrip("# "), people=[], text=body.strip()))
    for p in sorted((DATA / "transcripts").glob("*.txt")):
        head, _, body = p.read_text().partition("\n\n")
        meta = dict(line.split(": ", 1) for line in head.splitlines())
        docs.append(dict(doc_id=p.stem, source="transcript", source_ref=f"gs://company-brain/transcripts/{p.name}",
                         ts=meta["Date"], title=meta["Meeting"],
                         people=[a.strip() for a in meta["Attendees"].split(",")], text=body.strip()))
    return crm, sorted(docs, key=lambda d: d["ts"])


# ---------- extract: entities + temporal edges ----------
# Known ceiling: regex/gazetteer extractor, only proven on this synthetic sample. On Databricks this step is
# ai_query() with a JSON response schema over normalized.documents; the edge schema below stays the same.

REQUEST = re.compile(r"\b(raise|request|seeking)\b.*?\$(\d+(?:\.\d+)?) million", re.I)
LENDER_STATUS = [  # first match wins, so the strongest signal goes first
    ("declined", re.compile(r"\b(pass|declin)", re.I)),
    ("indication_of_interest", re.compile(r"indication of interest|term sheet issued", re.I)),
    ("credit_committee", re.compile(r"credit committee", re.I)),
    ("reviewing", re.compile(r"\b(review|materials)\b", re.I)),
]


def sentences(text):
    return re.split(r"(?<=[.!?])\s+", text)


def alias_pattern(e):
    names = [e["name"], *e.get("aliases", [])]
    return re.compile(r"\b(" + "|".join(re.escape(n) for n in sorted(names, key=len, reverse=True)) + r")\b")


def attribute_deal(doc, crm):
    """Entity resolution for deals: company named in the text, or a company domain among the participants.
    A company with several deals resolves to its most recently opened deal that is not closed."""
    blob = doc["title"] + " " + doc["text"]
    domains = {p.split("@")[1] for p in doc["people"]}
    for c in crm["companies"]:
        if alias_pattern(c).search(blob) or c["domain"] in domains:
            open_deals = [d for d in crm["deals"] if d["company"] == c["id"] and d["stage"] != "closed"]
            if open_deals:  # the advisor's own domain is on every email but it has no deals, so it never resolves
                return max(open_deals, key=lambda d: d["opened"])["id"]
    return None


def extract(crm, docs):
    contacts = {c["email"]: c for c in crm["contacts"]}
    named = crm["companies"] + crm["lenders"]
    edges = []
    for doc in docs:
        doc["deal"] = deal = attribute_deal(doc, crm)
        ref, ts = doc["source_ref"], doc["ts"]
        for e in named:
            if alias_pattern(e).search(doc["title"] + " " + doc["text"]):
                edges.append(dict(src=doc["doc_id"], rel="MENTIONS", dst=e["id"], value=None, ts=ts, source_ref=ref))
        if not deal:
            continue
        for email in doc["people"]:
            if email in contacts:
                edges.append(dict(src=contacts[email]["id"], rel="PARTICIPATED_IN", dst=deal, value=None, ts=ts, source_ref=ref))
        for s in sentences(doc["text"]):
            if m := REQUEST.search(s):
                edges.append(dict(src=deal, rel="FINANCING_REQUEST", dst=None, value=float(m.group(2)), ts=ts, source_ref=ref))
            for lender in crm["lenders"]:
                if alias_pattern(lender).search(s):
                    status = next((name for name, rx in LENDER_STATUS if rx.search(s)), None)
                    if status:
                        edges.append(dict(src=lender["id"], rel="REVIEWED", dst=deal, value=status, ts=ts, source_ref=ref))
    return edges


# ---------- hybrid retrieval ----------

def tokenize(text):
    return [t.strip(".,") for t in re.findall(r"[a-z0-9$+.,]+", text.lower()) if t.strip(".,")]


class BM25:
    def __init__(self, texts, k1=1.5, b=0.75):
        self.docs = [Counter(tokenize(t)) for t in texts]
        self.lens = [sum(d.values()) for d in self.docs]
        self.avg = sum(self.lens) / len(self.lens)
        df = Counter(term for d in self.docs for term in d)
        n = len(self.docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}
        self.k1, self.b = k1, b

    def scores(self, query):
        q = tokenize(query)
        return [sum(self.idf.get(t, 0) * d[t] * (self.k1 + 1) /
                    (d[t] + self.k1 * (1 - self.b + self.b * ln / self.avg)) for t in q)
                for d, ln in zip(self.docs, self.lens)]


class Vectors:
    def __init__(self, texts):
        from sentence_transformers import SentenceTransformer  # fail loudly if missing; no silent BM25-only mode
        self.model = SentenceTransformer(EMBED_MODEL, device="cpu")
        self.emb = self.model.encode(texts, normalize_embeddings=True)

    def scores(self, query):
        q = self.model.encode([BGE_QUERY_PREFIX + query], normalize_embeddings=True)[0]
        return (self.emb @ q).tolist()


def ranks(scores):
    order = sorted(range(len(scores)), key=lambda i: -scores[i])
    return {i: r for r, i in enumerate(order, 1)}


def rrf(*rank_maps, k=60):
    fused = Counter()
    for rm in rank_maps:
        for i, r in rm.items():
            fused[i] += 1 / (k + r)
    return [i for i, _ in fused.most_common()]


class Brain:
    def __init__(self, with_vectors=True):
        self.crm, self.docs = load()
        self.edges = extract(self.crm, self.docs)
        self.by_id = {**{e["id"]: e for k in ("companies", "lenders", "contacts", "deals") for e in self.crm[k]}}
        texts = [d["title"] + "\n" + d["text"] for d in self.docs]  # Known ceiling: one chunk per doc; chunk by paragraph past ~500 tokens
        self.bm25 = BM25(texts)
        self.vec = Vectors(texts) if with_vectors else None

    def search(self, query, k=3, mode="hybrid"):
        bm = ranks(self.bm25.scores(query))
        if mode == "bm25":
            order = sorted(bm, key=bm.get)
        else:
            vr = ranks(self.vec.scores(query))
            order = sorted(vr, key=vr.get) if mode == "vector" else rrf(bm, vr)
        return [self.docs[i] for i in order[:k]]

    def edges_for(self, rel, deal):
        return sorted((e for e in self.edges if e["rel"] == rel and deal in (e["src"], e["dst"])), key=lambda e: e["ts"])

    # ----- the five client questions; every list item carries source_ref(s) -----

    def latest(self, deal, n=3):
        events = [d for d in self.docs if d["deal"] == deal][-n:][::-1]
        return dict(recent=[dict(ts=d["ts"], title=d["title"], source_ref=d["source_ref"]) for d in events],
                    current_request=self.financing(deal)["current"],
                    lender_status=self.lenders_reviewed(deal))

    def who(self, deal):
        people = {}
        for e in self.edges_for("PARTICIPATED_IN", deal):
            p = people.setdefault(e["src"], dict(first_seen=e["ts"], sources=[]))
            p["last_seen"] = e["ts"]
            p["sources"].append(e["source_ref"])
        return [dict(name=self.by_id[pid]["name"], title=self.by_id[pid]["title"],
                     org=self.by_id[self.by_id[pid]["org"]]["name"], **p) for pid, p in people.items()]

    def financing(self, deal):
        history = []
        for e in self.edges_for("FINANCING_REQUEST", deal):
            if history and history[-1]["amount_musd"] == e["value"]:
                history[-1]["sources"].append(e["source_ref"])  # same amount restated: more evidence, not a change
            else:
                history.append(dict(amount_musd=e["value"], as_of=e["ts"], sources=[e["source_ref"]]))
        return dict(current=history[-1] if history else None, history=history)

    def lenders_reviewed(self, deal):
        out = {}
        for e in self.edges_for("REVIEWED", deal):
            out.setdefault(e["src"], []).append(dict(status=e["value"], ts=e["ts"], source_ref=e["source_ref"]))
        return [dict(lender=self.by_id[lid]["name"], current=h[-1]["status"], history=h) for lid, h in out.items()]

    def relevant_lenders(self, deal):
        """Lenders with funded deals in the same industry, excluding lenders already engaged on this deal.
        Same-industry declines are returned as a separate negative signal."""
        industry = self.by_id[self.by_id[deal]["company"]]["industry"]
        engaged = {e["src"] for e in self.edges_for("REVIEWED", deal)}
        peers = {d["id"] for d in self.crm["deals"] if d["id"] != deal and self.by_id[d["company"]]["industry"] == industry}
        hist = [h for h in self.crm["deal_lender_history"] if h["deal"] in peers]

        def row(h):
            return dict(lender=self.by_id[h["lender"]]["name"], deal=h["deal"], outcome=h["outcome"], date=h["date"],
                        source_ref=f"crm://deal_lender_history/{h['deal']}/{h['lender']}")
        return dict(industry=industry,
                    suggested=[row(h) for h in hist if h["outcome"] == "funded" and h["lender"] not in engaged],
                    negative_signals=[row(h) for h in hist if h["outcome"] == "declined"])


EVAL_SET = [  # (query, doc_ids that count as a hit, what it probes)
    ("SOFR+475", {"2026-09-18_Northwind_Quillfeather_call"}, "exact token"),
    ("leverage covenant", {"2026-09-05_Northwind_Term_Sheet_Request_v2"}, "keyword"),
    ("EBITDA", {"2026-08-12_Northwind_CIM_v1"}, "keyword"),
    ("Fabrikam refinancing", {"msg-006"}, "entity name"),
    ("which bank said no to Northwind", {"msg-005"}, "paraphrase"),
    ("why did the loan amount go up", {"msg-004", "2026-09-05_Northwind_Term_Sheet_Request_v2"}, "paraphrase"),
    ("lender expressed interest in the deal", {"msg-008"}, "paraphrase"),
    ("trucking exposure limit", {"msg-005"}, "mixed"),
    ("sold trucks so the ask went down", {"msg-009"}, "paraphrase"),
    ("who is taking it to credit committee", {"2026-09-18_Northwind_Quillfeather_call"}, "mixed"),
]


def evaluate(brain, k=3):
    report = {}
    for mode in ("bm25", "vector", "hybrid"):
        hits1 = hitsk = mrr = 0
        for q, gold, _ in EVAL_SET:
            ids = [d["doc_id"] for d in brain.search(q, k=len(brain.docs), mode=mode)]
            first = next(r for r, i in enumerate(ids, 1) if i in gold)
            hits1 += first == 1
            hitsk += first <= k
            mrr += 1 / first
        n = len(EVAL_SET)
        report[mode] = dict(hit_at_1=hits1 / n, hit_at_k=hitsk / n, mrr=round(mrr / n, 3))
    return report


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "answers"
    if cmd == "answers":
        b = Brain(with_vectors=False)
        for q, fn in [("What is the latest on the deal?", b.latest), ("Who has been involved?", b.who),
                      ("What is the current financing request, and how has it changed?", b.financing),
                      ("Which lenders have reviewed the opportunity?", b.lenders_reviewed),
                      ("Which lenders appear relevant based on prior company activity?", b.relevant_lenders)]:
            print(f"\n## {q}\n" + json.dumps(fn("D-101"), indent=2))
    elif cmd == "search":
        b = Brain()
        for d in b.search(" ".join(sys.argv[2:]), k=5):
            print(f"{d['ts'][:10]}  {d['source_ref']}\n    {d['title']}")
    elif cmd == "eval":
        print(json.dumps(evaluate(Brain()), indent=2))
