## Data
This folder documents the source corpus used for this project. Raw PDFs and parsed JSON output are **not versioned in this repo** (size/copyright). See the main README for details.

## Corpus Overview
~30 RAG research papers, selected with a deliberate weighting toward table-heavy benchmark/survey/comparison papers (to stress-test structure-aware chunking on tables/figures), alongside a smaller set of text-heavy architecture papers as controls.

## Paper List
| #  | Title  | Author  | DOI  |
| -  | -----  | ------  | ---  |
| 1  | **Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection**  | Asai et al.  | https://doi.org/10.48550/arXiv.2310.11511  |
| 2  | **Benchmarking Large Language Models in Retrieval-Augmented Generation**  | Chen et al.  | https://doi.org/10.48550/arXiv.2309.01431  |
| 3  | **CORAL: Benchmarking Multi-turn Conversational Retrieval-Augmentation Generation**  | Cheng et al.  | https://doi.org/10.48550/arXiv.2410.23090  |
| 4  | **Ragas: Automated Evaluation of Retrieval Augmented Generation**  | Es et al.  | https://doi.org/10.48550/arXiv.2309.15217  |
| 5  | **A Survey on RAG Meeting LLMs: Towards Retrieval-Augmented Large Language Models**  | Fan et al.  | https://doi.org/10.48550/arXiv.2405.06211  |
| 6  | **RAGBench: Explainable Benchmark for Retrieval-Augmented Generation Systems**  | Friel et al.  | https://doi.org/10.48550/arXiv.2407.11005  |
| 7  | **Retrieval-Augmented Generation for Large Language Models: A Survey**  | Gao et al.  | https://doi.org/10.48550/arXiv.2312.10997  |
| 8  | **Atlas: Few-shot Learning with Retrieval Augmented Language Models**  | Izacard et al.  | https://doi.org/10.48550/arXiv.2208.03299  |
| 9  | **Leveraging Passage Retrieval with Generative Models for Open Domain Question Answering**  | Izacard and Grave  | https://doi.org/10.48550/arXiv.2007.01282  |
| 10  | **Adaptive-RAG: Learning to Adapt Retrieval-Augmented Large Language Models through Question Complexity**  | Jeong et al.  | https://doi.org/10.48550/arXiv.2403.14403  |
| 11  | **Active Retrieval Augmented Generation**  | Jiang et al.  | https://doi.org/10.48550/arXiv.2305.06983  |
| 12  | **Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks**  | Lewis et al.  | https://doi.org/10.48550/arXiv.2005.11401  |
| 13  | **RA-DIT: Retrieval-Augmented Dual Instruction Tuning**  | Lin et al.  | https://doi.org/10.48550/arXiv.2310.01352  |
| 14  | **RECALL: A Benchmark for LLMs Robustness against External Counterfactual Knowledge**  | Liu et al.  | https://doi.org/10.48550/arXiv.2311.08147  |
| 15  | **CRUD-RAG: A Comprehensive Chinese Benchmark for Retrieval-Augmented Generation of Large Language Models**  | Lyu et al.  | https://doi.org/10.48550/arXiv.2401.17043  |
| 16  | **LLM Readiness Harness: Evaluation, Observability, and CI Gates for LLM/RAG Applications**  | Maiorano  | https://doi.org/10.48550/arXiv.2603.27355  |
| 17  | **RAG-Fusion: a New Take on Retrieval-Augmented Generation**  | Rackauckas  | https://doi.org/10.5121/ijnlc.2024.13103  |
| 18  | **ARES: An Automated Evaluation Framework for Retrieval-Augmented Generation Systems**  | Saad-Falcon et al.  | https://doi.org/10.48550/arXiv.2311.09476  |
| 19  | **RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval**  | Sarthi et al.  | https://doi.org/10.48550/arXiv.2401.18059  |
| 20  | **Retrieval-Augmented Generation: A Comprehensive Survey of Architectures, Enhancements, and Robustness Frontiers**  | Sharma  | https://doi.org/10.48550/arXiv.2506.00054  |
| 21  | **REPLUG: Retrieval-Augmented Black-Box Language Models**  | Shi et al.  | https://doi.org/10.48550/arXiv.2301.12652  |
| 22  | **A Survey of Query Optimization in Large Language Models**  | Song and Zheng  | https://doi.org/10.48550/arXiv.2412.17558  |
| 23  | **MultiHop-RAG: Benchmarking Retrieval-Augmented Generation for Multi-Hop Queries**  | Tang and Yang  | https://doi.org/10.48550/arXiv.2401.15391  |
| 24  | **RECOMP: Improving Retrieval-Augmented LMs with Compression and Selective Augmentation**  | Xu et al.  | https://doi.org/10.48550/arXiv.2310.04408  |
| 25  | **Corrective Retrieval Augmented Generation**  | Yan et al.  | https://doi.org/10.48550/arXiv.2401.15884  |
| 26  | **CRAG -- Comprehensive RAG Benchmark**  | Yang et al.  | https://doi.org/10.48550/arXiv.2406.04744  |
| 27  | **Chain-of-Note: Enhancing Robustness in Retrieval-Augmented Language Models**  | Yu et al.  | https://doi.org/10.48550/arXiv.2311.09210  |
| 28  | **RAFT: Adapting Language Model to Domain Specific RAG**  | Zhang et al.  | https://doi.org/10.48550/arXiv.2403.10131  |
| 29  | **Benchmarking Poisoning Attacks against Retrieval-Augmented Generation**  | Zhang et al.  | https://doi.org/10.48550/arXiv.2505.18543  |
| 30  | **Take a Step Back: Evoking Reasoning via Abstraction in Large Language Models**  | Zheng et al.  | https://doi.org/10.48550/arXiv.2310.06117  |

## Flagged Papers
Six papers have one or more hand-transcribed tables (`manual_transcription_flag` in the parsed JSON and QA schema), due to extraction limitations documented in [docs/corpus_notes.md](#docs/corpus_notes.md):
- RGB (2309.01431)
- Active RAG (2305.06983)
- Chain-of-Note (2311.09210)
- RAFT (2403.10131)
- RAG poisoning (2505.18543)
- Gao et al. (2312.10997)

One paper (RAPTOR) has appendix tables permanently excluded from QA generation — see `docs/corpus_notes.md`.

## Extending the Corpus
The parsing pipeline (`scripts/parsing/parse_papers.py`) is designed to generalize beyond this specific set of 30 papers — it is not hardcoded to them. In principle, other RAG papers, or papers from adjacent domains, can be dropped in and parsed the same way.

That said, a few assumptions baked into this project are worth flagging if you extend the corpus:
+ **Layout assumption:** Camelot-based table extraction was tuned against dense two-column IEEE-style layouts. Papers with substantially different layouts (single-column, non-IEEE templates, HTML-native papers) may need re-tuning or produce more extraction gaps.
+ **Section labeling:** The `section` field in parsed output is unreliable corpus-wide (most blocks land under "Front Matter"); content-keyword matching is used instead. this heuristic was tuned against this corpus' papers and may need adjustment for very different writing styles/structures.
+ **Known ceiling:** borderless/side-by-side tables in dense two-column layouts are a hard extraction ceiling for Camelot, not something fixable by re-tuning — expect this to recur in any similarly-formatted new papers.
+ **QA generation categories:** `plain_factual`, `table_lookup`, `figure_caption`, and `multi_hop_comparative` were designed with this corpus' content mix in mind. A domain with very few tables/figures (e.g., pure theory papers) may need a rebalanced category split.

If you do extend the corpus, it's worth re-running the retrieval/answer-quality evaluation separately on the new papers before merging results, since layout and content differences could otherwise confound the structure-aware vs. fixed-size comparison.
