# Methodology

## Research Question
Does structure-aware chunking — treating tables, figures, and captions as atomic retrieval units — improve RAG performance over standard fixed-size chunking, for QA over scientific research papers?

## Corpus
~30 RAG research papers, parsed from PDF into a block-based JSON schema ({`paper_id`, `blocks`, `flagged_items`}) using PyMuPDF, Camelot (lattice/stream), and pdfplumber. PDFs are not versioned in this repo; see README for storage location.

## Benchmark Construction
171 QA pairs were generated via the Gemini API and manually verified across four categories: `plain_factual` (40), `table_lookup` (60), `figure_caption` (40), and `multi_hop_comparative` (31). See [QA schema](../benchmark/qa_schema.md) for the full field-level schema.
Cross-paper multi-hop questions were attempted but discontinued: at this corpus scale, coincidental term/label overlap between papers produced a near-total false-connection failure rate during verification. All final multi-hop pairs are within-paper.

## Chunking Strategies
Two chunking strategies were implemented and indexed as separate ChromaDB collections (cosine similarity space):
1. **Structure-aware** (`structure_aware`, ~14K items): preserves document structure — tables, figures, and captions are kept as atomic retrieval units rather than being split or merged with surrounding text.
2. **Fixed-size** (`fixed_size_1000`, ~2.9K items): standard fixed-size chunking (1000-token windows), the common baseline approach.

## RAG Pipeline
A minimal, direct RAG implementation was used rather than LangChain/LangGraph, to avoid dependency overhead and because those frameworks' abstractions didn't align well with evaluating against a custom gold-label benchmark. Generation uses `gemini-3.5-flash`; a separate model, `gemini-3.1-pro-preview`, is used as the LLM-as-judge to reduce self-evaluation bias.

## Evaluation
Dual-axis evaluation was used:
- **Retrieval metrics:** Recall@k and MRR, computed against gold block IDs.
- **Answer quality:** LLM-as-judge scoring of end-to-end generated answers.

## Key Results
- Structure-aware chunking outperformed fixed-size chunking on every retrieval metric **(Recall@10: 0.883 vs. 0.608; MRR: 0.683 vs. 0.346)**, with both differences statistically significant. The largest gap appeared in the `table_lookup` category.
- Answer quality followed the same pattern overall, with two notable exceptions: `faithfulness` and `multi_hop_comparative` did not show statistically significant differences between strategies.
- The `multi_hop_comparative` non-significance was traced to a **joint top-5 coverage gap**: most multi-hop question pairs didn't have all their gold blocks present in the generation context, regardless of chunking strategy. This is a chunking-strategy-independent limitation of retrieval depth/context window, not a sign that structure-aware chunking fails on multi-hop questions specifically.

## Known Limitations
See [Corpus notes](corpus_notes.md) for corpus-level gaps: RAPTOR appendix tables excluded from QA generation, six papers with hand-transcribed tables, and Camelot's inability to reliably isolate borderless/side-by-side tables in dense two-column IEEE layouts.

## Reproducibility / Idempotency
Given multi-day work across fresh Colab runtimes, the pipeline was built to be idempotent:
+ ChromaDB indexing checks existing collection IDs before re-embedding.
+ Question embeddings are cached to `question_embeddings_cache.json`, keyed by `qa_id`.
+ Judge scores are loaded from `judged_answers.json` rather than relying on in-memory state.
+ Long-running QA generation/judging steps checkpoint every 10 items to survive Gemini API daily quota exhaustion mid-run.
