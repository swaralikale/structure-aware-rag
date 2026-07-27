# Corpus Notes
Known gaps and hard detection ceilings encountered during parsing and QA generation. These are documented rather than silently worked around, since they affect what the benchmark can and cannot test.

## Excluded content
- **RAPTOR (2401.18059) — Appendix Tables 9, 14-21:** Parser failed to detect caption blocks for the listed tables. Excluded from QA generation entirely rather than generating questions against uncaptioned/ambiguous table content.

## Manually transcribed tables
The following six papers have one or more tables that were hand-transcribed (rather than extracted via Camelot/pdfplumber) and are flagged via `manual_transcription_flag` for separate analysis:
- RGB (2309.01431)
- Active RAG (2305.06983)
- Chain-of-Note (2311.09210)
- RAFT (2403.10131)
- RAG poisoning (2505.18543)
- Gao et al. (2312.10997)

 ## Confirmed hard detection ceilings
 **Borderless/side-by-side tables in dense two-column IEEE layouts:** Camelot cannot reliably isolate these (e.g., RBG Tables 3, 5, 7). This is a tooling limitation, not a bug fix within project scope.

 ## Why this matters for evaluation?
 Any retrieval or answer-quality results involving the papers/tables above should be interpreted with these caveats in mind, e.g., consider reporting metrics with and without the manually-transcribed subset to check whether structure-aware's advantage (if any) holds up outside of clean, auto-extracted tables.
