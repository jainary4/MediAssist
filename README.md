# MediAssist

MediAssist is a closed-corpus clinical question-answering prototype built over 50 fixed PDF monographs. It retrieves text, tables, figures, scanned appendices, and explicit links between documents; writes an answer only from retrieved evidence; returns page-level citations; and abstains when the evidence is missing or insufficient.

> This is an interview prototype over a supplied document collection. It is not medical advice and is not intended to process real patient data.

## Live demo

**[Open the MediAssist clinical document assistant](https://mediassist-kmyjtfkkufdqwsskjwmewg.streamlit.app/)**

Ask a question, inspect the cited document pages, and start a new conversation from the sidebar. Multi-turn context is stored by user and session in PostgreSQL, but previous chat messages are treated only as conversational context—not as medical evidence.

## What the system must handle

The corpus is deliberately difficult: native PDF text, scanned pages, multi-column layouts, rotated pages, dense and landscape tables, footnotes, figures containing numeric values, and questions whose evidence spans multiple PDFs. The system therefore has four requirements:

1. Preserve document structure during ingestion instead of flattening every page into plain text.
2. Retrieve by meaning, exact wording, structured fields, figures, and explicit document links.
3. Preserve provenance so every factual claim can be traced to a document, page, section, and Docling source reference.
4. Refuse to guess when the corpus does not provide sufficient evidence.

## End-to-end architecture

```mermaid
flowchart TD
    A[50 clinical PDFs] --> B[Modal Docling ingestion]
    B --> C{Quality checks}
    C -->|Pass| D[Verified JSON, Markdown, tables, figures, page images]
    C -->|Scan candidate plus weak output| E[Full-page OCR retry]
    E --> C

    D --> F[Evidence builder]
    F --> G[SQLite canonical evidence and table rows]
    F --> H[SQLite FTS5 keyword index]
    F --> I[FAISS semantic index]
    F --> J[Figure OCR records and explicit document links]

    K[Streamlit chat] --> L[Modal assistant API]
    L <--> M[PostgreSQL session memory]
    L --> N[Deterministic query router]
    N --> G
    N --> H
    N --> I
    N --> J
    G --> O[Ranked evidence pack]
    H --> O
    I --> O
    J --> O
    O --> P[One grounded answer-writing agent]
    P --> Q[Citation validation and heuristic diagnostics]
    Q -->|Grounded answer| R[Answer, citations, support score]
    Q -->|Answer absent or unsupported| S[Abstention with reason]
    R --> K
    S --> K
```

The design is hybrid RAG with explicit structured relations, not a general-purpose knowledge graph. A full knowledge graph would add entity extraction and relation-maintenance complexity that the fixed corpus does not require. Explicit cross-document instructions are represented as hard links in SQLite; broader similarity remains available through semantic and keyword retrieval.

## Why hybrid retrieval

No single retrieval method covers the corpus:

| Evidence need | Best retrieval path | Why |
| --- | --- | --- |
| Explanatory prose | Semantic search + keyword search | Meaning and exact clinical terms both matter. |
| Codes and registry identifiers | Metadata/SQLite + FTS5 | Exact strings should not depend on embedding similarity. |
| Dose or landscape-table values | Structured table rows | The row-to-column relationship should be preserved. |
| Figure values | Figure record + precomputed OCR | Captions, nearby text, OCR, and visual warnings are separate evidence. |
| Scanned appendices | OCR-derived text/table evidence | The answer may not exist in the PDF text layer. |
| Cross-document questions | Explicit reference traversal + target retrieval | The answer should cite the source instruction and target value. |
| Corpus-wide lists | Deterministic SQL aggregation | Top-k search cannot prove that a list is complete. |

Semantic and keyword results are fused with reciprocal-rank fusion. Structured evidence receives small priority bonuses because exact rows, metadata, figures, and explicit links are valuable for their matching intents.

## Retrieval depth and top-k policy

The initial controller requests up to 12 semantic and 12 keyword candidates. Intent-specific lookups are added when required. Semantic search examines a larger candidate pool—`max(10 × top_k, 50)`—before applying a document filter.

Retrieved records are merged by evidence ID using reciprocal-rank fusion with `k = 60`. The controller sends the top 12 fused evidence items to the answer-writing agent. The agent can use bounded follow-up hybrid search, table lookup, and reference traversal. Corpus-wide questions bypass top-k completeness problems through deterministic aggregation.

The small evidence pack limits prompt noise and latency. Its trade-off is that duplicated representations or weak routing can displace relevant one-channel evidence.

## Knowledge-base build

The evaluated build contains:

| Record | Count |
| --- | ---: |
| Documents | 50 |
| Searchable evidence chunks | 972 |
| Text chunks | 788 |
| Table chunks | 121 |
| Figure chunks | 13 |
| Explicit reference chunks | 50 |
| Canonical tables | 111 |
| Structured table rows | 362 |
| Resolved document references | 50 / 50 |
| Processing errors | 0 |

All 50 submitted PDFs selected the standard Docling pipeline and passed the prototype’s structural checks. A pass means the expected artifacts and basic structures exist; it does not prove that every footnote, OCR token, table relationship, or chart value is correct.

## Latest supplied evaluation

The supplied evaluation contains questions but no gold answers. The results below describe completion, answer coverage, abstention behavior, and citation support—not verified factual accuracy.

- Run ID: `evaluation-20260714T134805Z-2a2071f4`
- 40/40 calls completed.
- Backend errors: 0.
- 31 questions received answers.
- 9 questions were abstained.
- Observed answer coverage: 77.5%.

| Question group | Total | Answered | Abstained | Observed behavior |
| --- | ---: | ---: | ---: | --- |
| General text | 8 | 8 | 0 | Semantic and keyword retrieval worked well. |
| Classification codes | 5 | 5 | 0 | Exact metadata and table retrieval worked. |
| Review-body footnotes | 4 | 0 | 4 | The table marker was present, but the resolving footnote was absent from the knowledge base. |
| Figure numeric values | 4 | 3 | 1 | OCR recovered all four tested values; one result was an assistant false abstention over a missing unit. |
| Scanned appendices | 4 | 4 | 0 | OCR-derived appendix text was retrieved successfully. |
| Landscape tables | 3 | 3 | 0 | Table windows and structured rows preserved usable values. |
| Cross-document doses | 6 | 6 | 0 | All explicit source-to-target link chains completed. |
| Corpus-wide aggregation | 2 | 2 | 0 | Deterministic SQL avoided incomplete top-k lists. |
| General/possible abstention | 3 | 0 | 3 | Corpus review found mentions but not the requested answers. |
| Reverse registry lookup | 1 | 0 | 1 | `CDR-1500` was not present in the corpus. |

Four of the nine abstentions appear appropriate out-of-corpus outcomes: melanoma treatment, COVID-19 transmission, pancreatic cancer five-year survival, and `CDR-1500`. Four are explained by missing approval footnotes. The final abstention occurred even though figure OCR retrieved `959`; the assistant declined to attach the question’s `per 100,000` unit to OCR that did not repeat the unit.

The readable supplied-set answers and citations are in [`agent_answers.md`](agent_answers.md). Complete response objects, traces, confidence components, and build identifiers are in [`backend_responses.jsonl`](backend_responses.jsonl). The questions are in [`Medical_PDF/candidate_questions.md`](Medical_PDF/candidate_questions.md).

The persisted Modal artifacts are:

```json
{
  "run_id": "evaluation-20260714T134805Z-2a2071f4",
  "volume_name": "clinical-qa-evaluation-results",
  "backend_responses_path": "runs/evaluation-20260714T134805Z-2a2071f4/backend_responses.jsonl",
  "agent_answers_path": "runs/evaluation-20260714T134805Z-2a2071f4/agent_answers.md",
  "backend_responses_bytes": 225348,
  "agent_answers_bytes": 51220
}
```

## Contributed evaluation

The contributed set was designed to cover failure modes that a set of ordinary prose questions would miss.

- Run ID: `evaluation-20260714T134544Z-29a2cc4a`
- 10/10 calls completed.
- Backend errors: 0.
- 9 in-corpus questions were answered.
- The out-of-corpus Lyme disease question was correctly abstained.

| Contributed capability | Result |
| --- | --- |
| Table-only baseline/12-month lookup | Answered |
| Figure condition and 2024 value | Answered |
| Scanned-appendix percentage | Answered |
| Explicit hard-link dose and route | Answered |
| Soft comparison across two related documents | Answered |
| Normal-text anatomy mapping | Answered |
| Approval footnote that was successfully ingested | Answered |
| CancerGov classification plus Tier 3 filter | Answered |
| Two-figure comparison and subtraction | Answered |
| Out-of-corpus Lyme disease treatment | Abstained |

The contributed questions are documented in [`Contributed_Questions.md`](Contributed_Questions.md). The manually established expected answers and the PDF citations used to create them are in [`ground_truth_answers.md`](ground_truth_answers.md) for inclusion in GitHub. The contributed evaluation output retains the generated answers and returned citations for comparison.

The contributed set showed that the system performs well when table, text, figure, OCR, and explicit-link evidence are preserved. It also revealed that a correct result does not always mean the route is generally exhaustive: the CancerGov/Tier 3 answer came from bounded structured retrieval rather than deterministic corpus aggregation, and soft multi-document comparison worked because both source documents were named.

## Confidence and abstention

The answer-writing model does not assign its own confidence. Application code reports a transparent heuristic based on citation coverage, retrieval support, extraction quality, grounding structure, and evidence consistency.

The latest validation no longer automatically rejects an answer merely because it came from OCR, a figure, a scanned page, or an alphanumeric table value. Unknown citation IDs and uncited factual claims remain grounding failures. Visual and OCR warnings are retained as limitations rather than treated as proof that the answer is wrong.

The latest results show that the score is not calibrated:

- All 31 answered supplied questions were labelled `high`.
- No supplied answer was labelled `medium`.
- Mean supplied-answer confidence was approximately `0.942`.
- Fourteen supplied answers reached the `0.95` ceiling.
- All nine answered contributed questions were labelled `high`.
- Mean contributed-answer confidence was approximately `0.945`.
- Six contributed answers reached the `0.95` ceiling.

Correct out-of-corpus abstentions and the incorrect `959` figure abstention all received `0.25`. Review-body abstentions received approximately `0.65` when the model included citations to the unresolved table marker, while a similar abstention received `0.25` when it returned no explanatory citations. The score therefore measures response and citation structure more reliably than factual correctness.

The honest label is:

> Heuristic support score—not a calibrated probability that the answer is correct.

A production confidence model would require human-verified answers, retrieval labels, claim-level citation entailment labels, correct-abstention labels, a held-out calibration set, and reliability analysis.

## Design choices and observed trade-offs

| Design choice | Evaluation benefit | Trade-off revealed |
| --- | --- | --- |
| Standard Docling first, targeted OCR retry | Preserved native prose and tables while still handling scans | Small footnotes can be missed on otherwise readable pages |
| JSON, Markdown, structured rows, and assets | Multiple representations rescued awkward tables and enabled citations | Representations must be reconciled and deduplicated |
| Semantic + keyword + structured retrieval | Strong coverage across prose, exact codes, tables, figures, and links | More routing and fusion heuristics are required |
| Explicit SQLite links instead of a full graph | All tested hard-link chains completed with traceable citations | Implicit document relationships remain shallow |
| Deterministic corpus aggregation | Complete lists for supported Tier and formulary queries | Equivalent paraphrases may not trigger aggregation |
| One bounded answer-writing agent | Simple deployment and no errors in the latest 50 calls | Abstention instructions can still be applied inconsistently |
| Deterministic citation-ID validation | Prevents usable invented citations | Valid citation ID does not prove semantic entailment |
| External heuristic confidence | Score components are inspectable | Scores cluster near the ceiling and are not calibrated |

## What worked well

1. General prose retrieval when the relevant monograph was named.
2. Exact metadata, classification-code, and table-field lookup.
3. Scanned-appendix OCR across all tested questions.
4. Landscape and conventional table retrieval.
5. Figure numeric OCR in the tested charts.
6. Explicit cross-document registry traversal.
7. Supported deterministic corpus aggregation.
8. Page-level, evidence-ID-based citations.
9. Out-of-corpus abstention despite irrelevant nearest-neighbor results.
10. Explicitly named multi-document comparison.

## Known limitations

1. Approval footnotes are inconsistently represented in the current knowledge base.
2. A structural ingestion pass is not extraction ground truth.
3. Figure OCR is linear text and does not preserve chart geometry or units reliably.
4. Flattened wide tables can retain words while weakening header/value semantics.
5. FAISS always returns nearest neighbors; it does not yet have a calibrated irrelevance threshold.
6. Routing depends on recognizable wording, and not every corpus-wide question triggers exhaustive aggregation.
7. Duplicate figure and reference representations can consume top-k positions and inflate agreement.
8. Hard links cover explicit registry instructions; implicit conceptual relationships are not stored as graph edges.
10. Assistant abstention can be inconsistent when evidence is present but units or visual relationships are incomplete.
11. Confidence is heuristic, compressed, and not a calibrated probability.
12. The official set lacks gold answers, so true answer accuracy remains unmeasured.


## What we would do next

1. Add a footnote dependency audit and targeted footer OCR.
2. Create a manually verified extraction and QA gold set.
3. Store figure OCR coordinates and derive simple axis/year/value relationships.
4. Reconcile wide-table headers with row values and flag unresolved mappings.
5. Deduplicate alternate representations under one canonical evidence source.
6. Add a reranker and calibrate an out-of-corpus threshold.
7. Make query routing multi-label and broaden deterministic aggregation templates.
8. Add lightweight soft links between related condition and population documents.
9. Evaluate retrieval recall at `k`, claim correctness, citation entailment, and abstention separately.
10. Calibrate confidence on held-out human-reviewed examples.
11. Add repeated-paraphrase and multi-turn evaluations.

For the detailed connection between architecture, results, trade-offs, and future work, see [`DESIGN_TRADEOFFS_AND_EVALUATION.md`](DESIGN_TRADEOFFS_AND_EVALUATION.md).

## Technology choices

| Layer | Technology | Role |
| --- | --- | --- |
| Document parsing | Docling | Layout-aware PDF parsing, OCR, tables, JSON, Markdown, and image assets |
| Cloud execution/storage | Modal | Remote ingestion, evidence building, assistant serving, and persistent volumes |
| Text chunking | Docling HybridChunker | Token-aware chunks retaining headings and document structure |
| Semantic retrieval | all-MiniLM-L6-v2 + FAISS | Fast local 384-dimensional cosine search |
| Keyword retrieval | SQLite FTS5 | Exact-term and BM25-style lexical matching |
| Structured retrieval | SQLite | Canonical chunks, metadata, table rows, figures, links, and aggregations |
| Figure OCR | Tesseract | Offline text and numeric-token extraction from figure crops |
| Agent orchestration | Agno + OpenRouter | One bounded answer-writing agent with structured output |
| Conversation memory | PostgreSQL on Neon | Persistent memory isolated by user ID and session ID |
| Demo UI | Streamlit Community Cloud | Public chat interface for the deployed Modal endpoint |

## Repository guide

- [`ingestion/README.md`](ingestion/README.md): Docling configuration, standard/OCR retry flow, output contract, checks, and ingestion limits.
- [`retrieval/README.md`](retrieval/README.md): evidence records, chunking, tables, figures, hard links, SQLite/FTS5/FAISS, ranking, and citations.
- [`assistant/README.md`](assistant/README.md): deterministic routing, agent tools, validation, structured responses, PostgreSQL memory, and Modal serving.


