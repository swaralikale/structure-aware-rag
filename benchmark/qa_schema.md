# QA Schema
Documents the schema used for each QA pair in `qa_pairs_verified.json` (171 verified pairs total).

## Fields
| Field  | Type  | Description  |
| -----  | ----  | -----------  |
| `qa_id`  | string  | Unique identifier for the QA pair.  |
| `category`  | string  | One of `plain_factual`, `table_lookup`, `figure_caption`, `multi_hop_comparative`. Underscore-delimited — not space-separated.  |
| `question`  | string  | The question text.  |
| `answer`  | string  | The gold answer text.  |
| `gold_block_ids`  | list[string]  | Block ID(s) in the parsed paper JSON that contain the information needed to answer the question. Used as the retrieval ground truth for Recall@k / MRR.  |
| `paper_ids`  | list[string]  | Paper(s) the question draws from. Length 1 for within-paper questions; only within-paper pairs exist in the final set (see note below).  |
| `cross_paper`  | boolean  | Whether the question spans multiple papers. Always `false` in the final verified set — cross-paper multi-hop QA was discontinued (see [Methodology](docs/methodology.md)).  |
| `manual_transcription_flag`  | boolean  | `true` if any gold block comes from a hand-transcribed table (see [Corpus Notes](docs/corpus_notes.md) for the six affected papers).  |
| `verified`  | boolean  | `true` once a pair has passed manual verification. Only verified pairs are included in the final benchmark.  |
| `notes`  | string  | Free-text notes from verification (e.g., ambiguity flags, edge cases).  |

## Category breakdown (final verified set: 171 pairs)
| Category  | Count  |
| --------  | -----  |
| `plain_factual`  | 40  |
| `table_lookup`  | 60  |
| `figure_caption`  | 40  |
| `multi_hop_comparative`  | 31  |


## Notes
- All 31 `multi_hop_comparative pairs` are **within-paper only**. Cross-paper multi-hop questions were generated in early exploration but discontinued after verification revealed a near-total false-connection failure rate caused by coincidental term/label overlap across papers at this corpus scale (~30 papers). This superseded the originally planned within/cross-paper split.
- Rejected QA pairs (failed verification) are logged separately in `qa_rejected_log.json` and are not part of the schema above — that log exists to prevent regenerating and re-rejecting the same bad pairs across pipeline runs.
