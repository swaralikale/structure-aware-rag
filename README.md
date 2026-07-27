# Structure-Aware RAG for Scientific Research Papers

This repository investigates whether **structure-aware chunking** — treating tables, figures, and captions as atomic retrieval units — improves Retrieval-Augmented Generation (RAG) performance on scientific research papers, compared to standard fixed-size chunking.


##  Table of Contents
1. [Research Question](#research-question)
2. [Corpus](#corpus)
3. [Project Structure](#project-structure)
4. [Pipeline](#pipeline)
5. [Data Schema](#data-schema)
6. [Setup](#setup)
7. [Results](#results)
8. [Known Limitations](#known-limitations)
9. [License](#license)



## Research Question
Does preserving document structure (tables, figures, captions as atomic units) during chunking improve retrieval and answer quality on a hand-built QA benchmark, relative to naive fixed-size chunking?

**Out of scope (deferred to future work):** citation-graph retrieval, equation-awareness, temporal versioning.


## Corpus
~30 RAG research papers, weighted toward table-heavy benchmark/survey/comparison papers (e.g., RGB, CRUD-RAG, RAGBench, MultiHop-RAG), with a few architecture papers included as text-heavy controls.
Papers and parsed output are stored in Google Drive (`/content/drive/MyDrive/RAG/`) rather than this repo, due to size and copyright. See [Corpus notes](../docs/corpus_notes.md) for known corpus gaps and hard detection ceilings.


## Project Structure
```
structure-aware-rag/
├── README.md
├── requirements.txt
├── .gitignore
├── LICENSE   
│
├── data/
│   ├── README.md
│   ├── raw_pdfs/                             # (gitignored — too large; note in README)
│   └── parsed_output/                        # 30 paper JSONs + quality summary (gitignored, linked via Drive)
│      
├── benchmark/                                # QA pairs + schema documentation
│   ├── qa_pairs.json                         # final verified ~171 QA pairs
│   └── qa_schema.md                          # documents the QA schema fields
│
├── scripts/
│   ├── parsing/
│      └── parse_papers.py
│   ├── correction/
│      └── correction_helper.py
│   ├── qa_generation/
│      ├── generate_qa_pairs.py
│      └── verify_qa_pairs.py
│   ├── utils/
│      └── fix_manual_transcription_flags.py
│
├── notebook/                                 # Colab notebook
│   └── structure_aware_RAG.ipynb
│
├── results/                                  # experiment outputs
├── answer_quality/
│   └── judged_scores_summary.csv 
├── retrieval_metrics/
│   └── retrieval_scores_summary.csv       
│
└── docs/
    ├── corpus_notes.md                       # known gaps: RAPTOR appendix, manual transcription flags, Camelot ceilings
    └── methodology.md

```


## Pipeline
1. **Parsing** — Extract text, tables, figures, and captions from 30 PDFs into a unified block-based JSON schema (`parse_papers.py`).
2. **Correction** — Manual review pass over flagged extraction issues (`correction_helper.py`).
3. **Benchmark construction** — Generate QA pairs via Gemini API across four question categories, then verify manually (`generate_qa_pairs.py`, `verify_qa_pairs.py`).
4. **Retrieval / RAG pipeline** — Build both chunking strategies (structure-aware vs. fixed-size) using a minimal, direct RAG implementation via Gemini API.
5. **Evaluation** — Compare strategies on retrieval metrics (Recall@k, MRR against gold chunks) and end-to-end answer quality (LLM-as-judge).


## Data Schema
Each parsed paper is a JSON object: {`paper_id`, `blocks`, `flagged_items`}, where each block has `block_id`, `block_type`, `content`, `refers_to`, `caption_of`. Table content is stored as row/cell lists; text and captions are stored as plain strings.
Each QA pair includes: `qa_id`, `category`, `question`, `answer`, `gold_block_ids`, `paper_ids`, `cross_paper`, `manual_transcription_flag`, `verified`, `notes`.


## Setup
```
Bash

git clone <repo-url>
cd structure-aware-rag
pip install -r requirements.txt
```

This project is developed and run on **Google Colab (free tier)**. Data lives in Google Drive; mount your Drive and point scripts at /content/drive/MyDrive/RAG/.
You'll need a Gemini API key (billing enabled) set as an environment variable for QA generation scripts.


## Results
The [Results](../results) folder contains the evaluation outputs comparing structure-aware vs. fixed-size chunking, at the per-QA-pair level. See [Methodology](../docs/methodology.md) for how these numbers were produced, and [QA schema](../benchmark/qa_schema.md) for the underlying QA pair schema.

### Answer Quality 
`judged_scores_summary.csv`

Per-QA-pair LLM-as-judge scores (via `gemini-3.1-pro-preview`) for the end-to-end generated answer, under each chunking strategy.
| Column  | Description  |
| ------  | -----------  |
| `qa_id`  | QA pair identifier, joins to benchmark/`qa_pairs_verified.json`.  |
| `category`  | `plain_factual` / `table_lookup` / `figure_caption` / `multi_hop_comparative`.  |
| `structure_correctness`  | Judge score for answer correctness, structure-aware chunking.  |
| `structure_completeness`  | Judge score for answer completeness, structure-aware chunking.  |
| `structure_faithfulness`  | Judge score for answer faithfulness (groundedness in retrieved context), structure-aware chunking.  |
| `fixed_correctness`  | Judge score for answer correctness, fixed-size chunking.  |
| `fixed_completeness`  | Judge score for answer completeness, fixed-size chunking.  |
| `fixed_faithfulness`  | Judge score for answer faithfulness, fixed-size chunking.  |


### Retrieval Metrics 
`retrieved_scores_summary.csv`

Per-QA-pair retrieval metrics against gold block IDs, under each chunking strategy.

| Column  | Description  |
| ------  | -----------  |
| `qa_id`  | QA pair identifier, joins to benchmark/`qa_pairs_verified.json`.  |
| `category`  | `plain_factual` / `table_lookup` / `figure_caption` / `multi_hop_comparative`.  |
| `cross_paper`   | Whether the question spans multiple papers (always `false` in the final set — see `benchmark/qa_schema.md`).  |
| `manual_transcription`  | Whether any gold block comes from a hand-transcribed table (see `docs/corpus_notes.md`).  |
| `structure_recall@5`  | Recall@5, structure-aware chunking.  |
| `structure_recall@10`  | Recall@10, structure-aware chunking.  |
| `structure_mrr`  | Mean Reciprocal Rank, structure-aware chunking.  |
| `fixed_recall@5`  | Recall@5, fixed-size chunking.  |
| `fixed_recall@10`  | Recall@10, fixed-size chunking.  |
| `fixed_mrr`  | Mean Reciprocal Rank, fixed-size chunking.  |

### Headline Results
- **Structure-aware chunking outperforms fixed-size on every retrieval metric (Recall@10: 0.883 vs. 0.608; MRR: 0.683 vs. 0.346)**, both statistically significant, with the largest gap in `table_lookup`.
- Answer quality follows the same pattern, except `faithfulness` and `multi_hop_comparative` do not show a statistically significant difference between strategies — traced to a **joint top-5 coverage gap** rather than a chunking-strategy weakness (see [Methodology](../docs/methodology.md)).

Both CSVs are joinable on `qa_id` for combined retrieval + answer-quality analysis (e.g., checking whether `manual_transcription-flagged` pairs behave differently).


## Known Limitations
See [Corpus notes](../docs/corpus_notes.md) for details on:
- RAPTOR appendix tables excluded from QA generation (no detected captions)
- Six papers with hand-transcribed tables (flagged separately)
- Camelot's inability to reliably isolate borderless/side-by-side tables in dense two-column IEEE layouts


## License
TBD
