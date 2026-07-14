# MediAssist: Design Choices, Trade-offs, Evaluation, and Next Steps

## Executive summary

MediAssist is a closed-corpus clinical question-answering prototype over 50 supplied PDF monographs. Its architecture was shaped by the document collection: the answers may occur in normal text, dense tables, figures, scanned appendices, footnotes, or a second document referenced by the first. The resulting system combines layout-aware ingestion, hybrid retrieval, structured lookup, explicit document links, one grounded answer-writing agent, citations, and abstention.

The latest official evaluation completed all 40 requests with no backend errors. It answered 31 questions and abstained on 9. Four abstentions appear appropriate because the requested information was not present in the corpus. Four in-corpus failures were concentrated in approval footnotes that were missing from the knowledge base, and one was an assistant false abstention even though figure OCR had retrieved the value `959`.

The contributed evaluation completed all 10 requests with no errors. It answered all nine in-corpus questions and correctly abstained on the out-of-corpus Lyme disease question. These questions exercised a conventional table, a figure, a scanned page, an explicit hard link, a soft two-document comparison, normal text, a footnote, a corpus-wide metadata filter, figure arithmetic across two documents, and abstention.

The results support the central design choice: a hybrid RAG system with structured evidence is a better fit than either vector search alone or a full knowledge graph. The two main areas requiring more work are consistent footnote ingestion and statistically meaningful confidence calibration.

## Evaluation basis

### Supplied 40-question run

- Run ID: `evaluation-20260714T134805Z-2a2071f4`
- Questions: 40
- Completed: 40
- Backend errors: 0
- Answered: 31
- Abstained: 9
- Observed answer coverage: 77.5%
- Backend responses: `runs/evaluation-20260714T134805Z-2a2071f4/backend_responses.jsonl`
- Readable answers: `runs/evaluation-20260714T134805Z-2a2071f4/agent_answers.md`
- Modal Volume: `clinical-qa-evaluation-results`

The supplied questions do not include gold answers. These figures therefore measure completion, answer coverage, abstention behavior, retrieval traces, and citation support. They are not a verified factual-accuracy score.

### Contributed 10-question run

- Run ID: `evaluation-20260714T134544Z-29a2cc4a`
- Questions: 10
- Completed: 10
- Backend errors: 0
- Answered: 9
- Abstained: 1
- Observed answer coverage: 90%

The contributed questions are documented in [`Contributed_Questions.md`](Contributed_Questions.md). The manually established expected answers and the PDF citations used to create them are documented in [`ground_truth_answers.md`](ground_truth_answers.md) for inclusion in the GitHub repository. The generated answers and their returned citations are retained with the contributed evaluation output.

## Results by capability

| Capability | Supplied set | Contributed set | What the result shows |
| --- | ---: | ---: | --- |
| Normal explanatory text | 8/8 answered | 1/1 answered | Semantic and keyword retrieval are reliable when the relevant monograph is identifiable. |
| Classification and metadata | 5/5 answered | 1/1 answered | Exact strings and table fields are better handled by SQLite than embeddings alone. |
| Approval footnotes | 0/4 answered | 1/1 answered | The retrieval path works when the footnote was ingested, but footnote capture is inconsistent. |
| Figure values | 3/4 answered | 2/2 answered | OCR recovered the tested values; one failure came from assistant abstention rather than missing OCR. |
| Scanned appendices | 4/4 answered | 1/1 answered | OCR-derived text is usable even when the scanned page is not represented as a figure. |
| Landscape or dense tables | 3/3 answered | 1/1 answered | Table chunks and structured rows provide useful complementary representations. |
| Explicit cross-document links | 6/6 answered | 1/1 answered | Registry-code resolution and target-document lookup are among the strongest parts of the design. |
| Soft multi-document comparison | Not isolated | 1/1 answered | Comparison works when both documents are named, but implicit relationship discovery remains limited. |
| Corpus-wide aggregation | 2/2 answered | 1/1 answered | Deterministic SQL prevents incomplete lists caused by semantic top-k retrieval. |
| Out-of-corpus abstention | 4/4 abstained after corpus review | 1/1 abstained | The answer-writing layer rejected irrelevant nearest-neighbor results rather than guessing. |

The supplied-set column reports observed behavior. Because official gold answers were not provided, “answered” is not equivalent to independently verified correctness.

## How the ingestion design explains the results

### Design choice: Docling instead of treating every page as an image

The ingestion pipeline uses Docling to preserve native text, headings, layout, tables, page provenance, and picture records. OCR is enabled, but full-page OCR is reserved for pages that need it. The pipeline saves lossless JSON, readable Markdown, table assets, figure assets, and page images.

This choice avoids replacing a high-quality PDF text layer with noisier OCR. It also gives the evidence builder structural information that plain text extraction would lose.

### Strengths produced by this choice

- All 50 documents produced retrieval artifacts without a processing exception.
- Normal prose remained clean enough for all eight supplied explanatory questions.
- Document Control Record and landscape tables remained searchable.
- Scanned appendix content became searchable text and answered all five tested scanned-page questions across both runs.
- Figure crops and captions allowed offline Tesseract OCR to recover values such as `174`, `359`, `219`, `544`, `774`, `589`, and `959`.
- Docling source references and page numbers were retained for citations.

### Trade-offs and weaknesses

#### Footnotes on otherwise valid pages

The four supplied review-body questions retrieved the table field `Approving body: See footnote †`, but the actual footnote text was absent. Each had `footnote_result_count = 0`. The Urinary Tract Infections in Children contributed question succeeded because its approval sentence was present in an ordinary text chunk.

This suggests a boundary failure between parsing and evidence construction:

- If the sentence is absent from Docling JSON and Markdown, it is an ingestion/layout extraction failure.
- If it exists in Docling output but not in SQLite or `chunks.jsonl`, it is an evidence-builder filtering or chunking failure.

The selective nature of the failure is consistent with the decision not to full-page OCR every otherwise readable page: a small footer can be missed even when the body text and tables look excellent.

#### Figures become linear OCR text

OCR can recognize chart numbers without preserving the relationship between an axis, year, unit, and plotted point. For example, the stored evidence can resemble:

```text
500 400 359 300 200 100 2016 2018 2020 2022 2024
```

The likely value is present, but the evidence does not formally encode `2024 → 359 per 100,000`. Some figure labels also contain OCR corruption. Captions and document metadata often compensate, but the relationship is not structurally guaranteed.

#### Flattened wide tables

Some wide table rows produce awkward mappings in which alternating labels and values are represented as keys and values. The parallel Markdown table usually preserves enough visual order for the assistant to recover the meaning. This validates the decision to keep both a readable table chunk and structured rows, but it also shows that a structured row is not automatically a correct relational record.

### What we would do next at the ingestion layer

1. Add dependency checks: whenever a table says `See footnote †`, confirm that a matching footnote record exists in the same document.
2. Reconcile Docling JSON, Markdown, and page images for every unresolved footnote marker.
3. Run targeted OCR on footer regions rather than re-OCRing the entire page.
4. Store figure OCR coordinates and build simple axis/value associations instead of saving only linear text.
5. Add table-header reconciliation tests for wide and landscape tables.
6. Create a small page-level extraction gold set covering prose, footnotes, tables, scanned pages, and charts.

## How the retrieval design explains the results

### Design choice: hybrid RAG with structured relations

The knowledge base has three complementary retrieval paths:

1. FAISS semantic search over embedded evidence chunks.
2. SQLite FTS5 keyword search over the same canonical evidence.
3. SQLite structured lookup for metadata, table rows, figures, explicit links, and supported aggregations.

This is hybrid RAG with explicit relations, not a general knowledge graph. A full graph would require entity normalization and relation maintenance across the corpus. The fixed collection mainly needs exact table access and traversal of explicit registry-code instructions, which SQLite can represent more simply.

### Strengths produced by this choice

#### Redundant retrieval for prose and tables

The system could recover a relevant passage by meaning, wording, or structure. The contributed Diabetic Neuropathies table question was routed as general text but still found the complete table chunk through hybrid search. Classification and maintenance-dose questions benefited from exact table lookup.

#### Reliable hard-link traversal

The evidence builder converted explicit linking language and registry codes into resolved source-to-target records. All supplied hard-link questions succeeded. The answer could cite both the source instruction and the target value, which is more auditable than relying on semantic similarity between the two documents.

#### Deterministic aggregation

The Monitoring Tier 3 query returned an aggregation record with 13 source members. The Nefralon query returned four source members. Using SQL for supported corpus-wide questions avoided pretending that a top-12 semantic result list was complete.

#### Auditable citations

The SQLite record is the canonical evidence object. FAISS and FTS5 locate its ID; the assistant returns that ID; the backend derives document title, page, section, excerpt, source reference, and asset path from the stored record. Multi-document answers can therefore return multiple independently traceable citations.

### Trade-offs and weaknesses

#### Forced nearest neighbors are not relevance decisions

FAISS returns the nearest chunks even for out-of-corpus questions. Lyme disease, COVID-19, pancreatic cancer, and the nonexistent registry code still produced candidate evidence. The assistant correctly rejected it in these runs, but the retriever itself does not yet have a calibrated “none of the above” threshold.

#### Routing depends on question phrasing

Some contributed questions succeeded through fallback retrieval rather than their ideal route:

- The Gallstones hard-link question was classified as an induction-dose query rather than a cross-document query.
- The CancerGov plus Tier 3 query was classified as a classification-code query rather than corpus aggregation.
- The Diabetic Neuropathies table query was classified as general text.

Hybrid redundancy allowed correct answers, which is a strength, but a paraphrase could still miss an intent-specific structured lookup.

#### Correct output does not always prove completeness

The contributed CancerGov/Tier 3 question returned the expected three monographs. However, it used a bounded structured-table retrieval rather than an exhaustive aggregation. The answer was correct for the current corpus, but that route does not guarantee completeness as the corpus grows.

#### Duplicate representations

A single source can appear as both `figure:` and `figure-chunk-`, or as both a reference record and reference chunk. This is useful for retrieval, but it can consume multiple top-k positions and make one source look like independent agreement between multiple sources.

#### Soft links are still shallow

The adult-versus-child kidney-stone comparison succeeded because both documents were explicitly named. The system does not yet automatically discover every related population, condition, or document family.

### What we would do next at the retrieval layer

1. Add a reranker and calibrate an out-of-corpus relevance threshold using labeled positive and negative queries.
2. Group alternate representations under one canonical source before ranking and confidence calculation.
3. Use document titles as boosts rather than irreversible filters when a question may require related documents.
4. Make routing multi-label so figure, table, metadata, reference, and aggregation routes can run together.
5. Expand deterministic aggregation templates beyond the currently recognized wording.
6. Add lightweight soft document links using normalized condition, population, authority, and shared terminology metadata.
7. Evaluate recall at `k` separately for text, tables, figures, footnotes, and cross-document questions.

## How the assistant workflow explains the results

### Design choice: one answer-writing agent over deterministic tools

The system uses one Agno answer-writing agent rather than separate router, retriever, citation, and confidence agents. Deterministic Python chooses the initial retrieval routes and exposes bounded tools for additional hybrid search, structured-table lookup, and reference traversal. The model writes a structured draft, while application code validates citation IDs and creates the final citation objects.

This keeps latency and orchestration complexity manageable while preventing the model from issuing arbitrary SQL or inventing a valid citation record.

### Strengths produced by this choice

- All 50 requests across the two latest runs completed without a backend error.
- The agent wrote concise answers over prose, tables, scans, and figures.
- Cross-document answers cited both documents.
- The contributed figure comparison correctly subtracted `589` from `774` to produce `185`.
- The model correctly abstained when nearest-neighbor results did not contain Lyme disease, COVID-19 transmission, pancreatic cancer survival, or `CDR-1500`.
- The structured-output error seen in an earlier contributed run was not reproduced in the latest run.

### Trade-offs and weaknesses

#### Abstention remains an LLM judgment

The Kidney Stones in Children figure evidence contained the value `959`, but the agent abstained because the OCR text did not explicitly repeat the unit `per 100,000`. Three structurally similar supplied figure questions were answered, and the contributed two-figure comparison answered while noting the missing unit.

This is an assistant false negative, not a retrieval miss. It demonstrates that natural-language abstention instructions can be applied inconsistently.

#### Citation validity is not entailment

The deterministic validator can prove that an evidence ID exists and was available to the current request. It does not prove that the cited passage logically supports every word of the claim. A dedicated entailment check or human gold set would be needed for that stronger guarantee.

#### Bounded tools trade recall for predictable execution

A small tool-call limit prevents loops and controls cost, but a complex comparison may need several searches and structured lookups. The initial hybrid evidence pack handled the current tests well; broader questions may exhaust the bounded calls.

### What we would do next at the assistant layer

1. Separate result states into `answered`, `model_abstained`, `out_of_corpus`, `retrieval_failure`, `ingestion_failure`, and `technical_error`.
2. Permit a supported value with an explicit limitation when only the unit or visual relationship remains uncertain.
3. Add one robust structured-output retry and a deterministic fallback for malformed provider responses.
4. Add claim-to-evidence entailment evaluation over a human-reviewed sample.
5. Test repeated paraphrases to measure answer and abstention consistency.
6. Add multi-turn evaluation; the batch tests used independent sessions and did not measure memory quality.

## Confidence scoring: design, observed behavior, and trade-off

### Why confidence was separated from the language model

The model is not asked to invent its own confidence. Application code reports a heuristic support score derived from citation coverage, retrieval support, extraction quality, grounding structure, and evidence consistency. This is more auditable than asking the LLM whether it “feels confident.”

The latest checks no longer automatically reject an answer merely because it came from a figure, OCR, a scan, or an alphanumeric table value. Those properties are retained as limitations and diagnostics.

### What the results revealed

The score is not calibrated and is not currently discriminative:

- All 31 answered supplied questions were labelled `high`.
- No supplied answer received a `medium` label.
- Mean supplied-answer confidence was approximately `0.942`.
- Fourteen supplied answers reached the `0.95` ceiling.
- All nine answered contributed questions were labelled `high`.
- Mean contributed-answer confidence was approximately `0.945`.
- Six contributed answers reached the `0.95` ceiling.

Simple definitions, scanned OCR, figures, tables, hard links, and corpus aggregations therefore receive very similar scores despite having different failure risks.

The abstention scores are also difficult to interpret. Correct out-of-corpus abstentions and the incorrect `959` figure abstention all received `0.25`. Review-body abstentions received approximately `0.65` when the agent cited the unresolved table marker, while a similar abstention received `0.25` when the model returned no explanatory citation. The score is therefore partly affected by how the LLM formats an abstention.

### Honest interpretation

The current number should be described as:

> An uncalibrated heuristic support score based on retrieval and citation structure, not a probability that the answer is correct.

Confidence should be reported for transparency, not used as factual proof. A visual warning or a low manually assigned extraction value should not hide an otherwise grounded answer.

### What we would do next for confidence

1. Create verified gold answers with claim-level citations.
2. Label retrieval success, answer correctness, citation entailment, and correct abstention separately.
3. Fit a calibration model on held-out examples rather than choosing weights manually.
4. Plot reliability: among answers scored near 0.8, measure how often the answer is actually correct.
5. Report confidence intervals and retain an explicit `uncalibrated` label until the scores pass calibration tests.
6. Avoid double-counting duplicate representations of the same source.
7. Report separate signals—retrieval strength, extraction risk, citation validity, and out-of-corpus likelihood—rather than hiding them behind one number.

## Design trade-offs in one view

| Design choice | Why it was chosen | Benefit demonstrated by evaluation | Cost or limitation |
| --- | --- | --- | --- |
| Standard Docling first; targeted OCR retry | Preserve native text and avoid unnecessary OCR noise | Strong prose, tables, and scanned-page coverage | Small footnotes can be missed on otherwise readable pages |
| JSON plus Markdown plus assets | Preserve structure and provide a readable fallback | Tables remained usable when structured rows were awkward | More representations require reconciliation |
| Hybrid semantic, keyword, and SQL retrieval | Different evidence types need different lookup behavior | Strong results across prose, codes, tables, links, and aggregation | Ranking and routing become more complex |
| Explicit links rather than a full knowledge graph | Fixed corpus contains deterministic registry references | All tested hard-link chains completed | Implicit relationships are not represented as graph edges |
| Small top-k evidence pack | Limit noise, latency, and prompt size | Most questions answered without tool calls | Relevant one-channel evidence can be displaced by duplicates |
| Deterministic aggregation | Top-k cannot establish completeness | Complete Tier 3 and Nefralon lists for supported patterns | Only implemented query patterns are exhaustive |
| One bounded answer agent | Simpler, cheaper, easier to trace | No backend errors in latest 50 calls | LLM abstention remains somewhat inconsistent |
| Deterministic citation validation | Prevent invented evidence IDs | Traceable page-level citations | Existing citation does not prove semantic entailment |
| Heuristic confidence outside the LLM | Make scoring inspectable | Transparent component breakdown | Scores cluster and are not calibrated probabilities |

## Short reflection on the contributed questions

Our ten questions showed that the system is strongest when different evidence representations reinforce one another. It correctly answered a table-only question, figure OCR questions, a scanned-appendix percentage, an explicit hard-link dose lookup, a two-document clinical comparison, a normal-text anatomy question, a successfully ingested approval footnote, a corpus metadata filter, and arithmetic across two figures; it also correctly abstained on Lyme disease, which is outside the collection. The questions also exposed the remaining risks: figure OCR can recover numbers without units or geometry, corpus-wide correctness can depend on whether routing chooses deterministic aggregation, soft links work best when both documents are named, and footnote success depends on whether ingestion preserved the footer. With more time, the first fixes would be footnote reconciliation, structured figure relationships, canonical deduplication, broader aggregation routing, and calibration against human-verified answers.

## Final conclusion

The evaluation supports the architecture rather than proving perfection. Layout-aware ingestion and hybrid structured retrieval allowed one system to handle substantially different evidence types without building a complex general knowledge graph. The remaining failures are concentrated and explainable: missing footnotes at the ingestion/evidence boundary, missing geometry in figure OCR, phrase-dependent routing, and an assistant confidence/abstention layer that has not been calibrated against ground truth. These are useful findings because they show exactly where additional engineering and evaluation effort should be spent.
