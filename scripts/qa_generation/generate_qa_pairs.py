import os
import json
import glob
import time
import random
import uuid
import argparse
from pathlib import Path

# ---------------------------------------------------------------------------
# CONFIG 
# ---------------------------------------------------------------------------

PARSED_OUTPUT_DIR = "parsed_output"   # folder of {arxiv_id}.json files
OUTPUT_PATH = "qa_pairs_raw.json"

GEMINI_MODEL_FLASH = "gemini-3.5-flash"       # cheap/fast: plain_factual, figure_caption
GEMINI_MODEL_PRO = "gemini-3.1-pro-preview"   # stronger reasoning: table_lookup, multi_hop

TARGET_COUNTS = {
    "plain_factual": 40,
    "table_lookup": 60,
    "figure_caption": 40,
    "multi_hop_comparative": 45,   # further split below
}
MULTI_HOP_WITHIN_PAPER_FRAC = 1.0   

RAPTOR_PAPER_ID = "2401.18059"
RAPTOR_EXCLUDED_TABLE_LABELS = {
    "Table 9", "Table 14", "Table 15", "Table 16", "Table 17",
    "Table 18", "Table 19", "Table 20", "Table 21",
}

MANUAL_TRANSCRIPTION_TABLES = {
    "2309.01431": {"Table 3", "Table 5", "Table 7"},   # RGB
    "2305.06983": {"Table 5", "Table 6"},               # Active RAG
    "2311.09210": {"Table 1"},                          # Chain-of-Note
    "2403.10131": {"Table 1"},                          # RAFT
    "2505.18543": {"Table_poisoning"},                  # CRAG poisoning table
    "2312.10997": {"TABLE I", "TABLE II"},              # Gao
}

random.seed(42)

# ---------------------------------------------------------------------------
# Loading & filtering parsed papers
# ---------------------------------------------------------------------------

def save_qa(all_qa, path=OUTPUT_PATH):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(all_qa, f, indent=2)


def load_existing_qa(path=OUTPUT_PATH):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


def load_all_papers(parsed_dir=PARSED_OUTPUT_DIR):
    papers = {}
    skipped = []
    for path in glob.glob(os.path.join(parsed_dir, "*.json")):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict) or "blocks" not in data:
            skipped.append(os.path.basename(path))
            continue
        paper_id = Path(path).stem
        papers[paper_id] = data["blocks"]

    if skipped:
        print(f"  Skipped {len(skipped)} non-paper file(s) in {parsed_dir}: {skipped}")
    return papers


def block_text(block):
    content = block.get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        lines = []
        for row in content:
            if isinstance(row, list):
                lines.append(" | ".join(str(cell) for cell in row))
            else:
                lines.append(str(row))
        return "\n".join(lines)
    return str(content)


def _find_caption_text(paper, block_id):
    for b in paper:
        if b.get("caption_of") == block_id:
            return block_text(b)
    return None


BOILERPLATE_CONTENT_MARKERS = (
    "we thank", "we would like to thank", "acknowledg", "funded by",
    "supported by", "grant no", "grant number", "author contribution",
)


def is_boilerplate_content(block):
    text = block_text(block).lower()
    return any(marker in text for marker in BOILERPLATE_CONTENT_MARKERS)


def _table_label_of(paper, block):
    caption_text = _find_caption_text(paper, block.get("block_id"))
    text = caption_text if caption_text is not None else block_text(block)
    first_line = text.strip().split("\n")[0] if text.strip() else ""
    # Captions are typically "Table N: description" or "Table N. description" or
    # "Table N, description" -- split on whichever separator appears first.
    for sep in (":", ".", ","):
        if sep in first_line:
            first_line = first_line.split(sep)[0]
            break
    return first_line.strip()


def is_excluded_gap(paper_id, paper, block):
    if paper_id != RAPTOR_PAPER_ID or block.get("block_type") not in ("table", "caption"):
        return False
    return _table_label_of(paper, block) in RAPTOR_EXCLUDED_TABLE_LABELS


def is_manual_transcription(paper_id, paper, block):
    block_id = block.get("block_id", "")
    if "manual" in block_id.lower():
        return True
    labels = MANUAL_TRANSCRIPTION_TABLES.get(paper_id, set())
    if not labels or block.get("block_type") not in ("table", "caption"):
        return False
    return _table_label_of(paper, block) in labels


# ---------------------------------------------------------------------------
# Candidate block selection per category
# ---------------------------------------------------------------------------

def get_text_blocks(paper):
    return [b for b in paper if b.get("block_type") == "text"]


def get_table_blocks(paper_id, paper):
    out = []
    for b in paper:
        if b.get("block_type") != "table":
            continue
        if is_excluded_gap(paper_id, paper, b):
            continue
        out.append(b)
    return out


def get_prose_refs_for_table(paper, table_block_id):
    return [b for b in paper if table_block_id in (b.get("refers_to") or [])]


def get_figure_blocks(paper_id, paper):
    return [
        b for b in paper
        if b.get("block_type") == "figure" and not is_excluded_gap(paper_id, paper, b)
    ]


def get_caption_for(paper, block_id):
    for b in paper:
        if b.get("caption_of") == block_id:
            return b
    return None


# ---------------------------------------------------------------------------
# Gemini API call wrapper
# ---------------------------------------------------------------------------

QA_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "question": {"type": "string"},
        "answer": {"type": "string"},
        "reasoning_uses_gold_blocks_only": {"type": "boolean"},
    },
    "required": ["question", "answer", "reasoning_uses_gold_blocks_only"],
}

MULTI_HOP_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "question": {"type": "string"},
        "answer": {"type": "string"},
        "reasoning_uses_gold_blocks_only": {"type": "boolean"},
        "both_blocks_strictly_necessary": {"type": "boolean"},
        "comparison_is_substantively_related": {"type": "boolean"},
        "answer_is_single_synthesized_conclusion": {"type": "boolean"},
    },
    "required": [
        "question", "answer", "reasoning_uses_gold_blocks_only",
        "both_blocks_strictly_necessary", "comparison_is_substantively_related",
        "answer_is_single_synthesized_conclusion",
    ],
}

_genai_client = None


def _get_client():
    global _genai_client
    if _genai_client is None:
        from google import genai
        _genai_client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    return _genai_client


def call_gemini(prompt, model=GEMINI_MODEL_FLASH, max_retries=5, response_schema=None):
    from google.genai import types
    from google.genai import errors

    client = _get_client()
    schema = response_schema if response_schema is not None else QA_RESPONSE_SCHEMA

    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=schema,
                    temperature=0.7,
                ),
            )
            return json.loads(response.text)
        except errors.APIError as e:
            is_rate_limit = getattr(e, "code", None) == 429
            wait = 60 if is_rate_limit else 2 ** attempt
            print(f"  [retry {attempt+1}/{max_retries}] {e.code if hasattr(e, 'code') else e} "
                  f"-- sleeping {wait}s")
            time.sleep(wait)
        except Exception as e:
            wait = 2 ** attempt
            print(f"  [retry {attempt+1}/{max_retries}] {e} -- sleeping {wait}s")
            time.sleep(wait)
    return None


# ---------------------------------------------------------------------------
# Prompt templates (one per category)
# ---------------------------------------------------------------------------

PLAIN_FACTUAL_PROMPT = """You are creating a QA benchmark item to test retrieval-augmented \
generation systems. Below is a passage of prose from a research paper. Write ONE factual \
question that is answerable entirely from this passage, with no need for any table, figure, \
or outside context. This is a CONTROL question -- keep it simple and directly stated in the text.

Passage:
\"\"\"
{content}
\"\"\"

Return the question, its correct answer (concise, 1-2 sentences), and confirm the answer uses \
only this passage.
"""

TABLE_LOOKUP_PROMPT = """You are creating a QA benchmark item to test retrieval-augmented \
generation systems on TABLE comprehension. Below is a table (with its caption) from a research \
paper{prose_context_note}.

Write ONE question whose answer requires reading a specific value from this table{prose_context_clause}. \
The question should not be answerable from general knowledge -- it must require this exact table's data. \
Be specific (name the exact metric, model, or row/column referenced).

Table caption + content:
\"\"\"
{table_content}
\"\"\"
{prose_block}
Return the question, the correct answer (state the specific value(s)), and confirm the answer \
uses only the table (and prose reference, if given) above.
"""

FIGURE_CAPTION_PROMPT = """You are creating a QA benchmark item to test retrieval-augmented \
generation systems on FIGURE comprehension. Below is a figure's caption/description from a \
research paper. Write ONE question whose answer depends on understanding what this figure shows \
or claims, as described in its caption. Do not invent visual details not stated in the caption text.

Figure caption:
\"\"\"
{caption_content}
\"\"\"

Return the question, the correct answer, and confirm the answer uses only this caption's content.
"""

MULTI_HOP_WITHIN_PROMPT = """You are creating a QA benchmark item requiring MULTI-HOP reasoning \
across two sections of the SAME research paper. Below are two blocks of content from that paper. \
Write ONE question that requires genuinely SYNTHESIZING both blocks into a single connected answer.

AVOID THESE THREE COMMON MISTAKES:

1. ARBITRARY COMPARISON: Do not compare two numbers/facts just because they both happen to be \
numeric or nameable. Example of what NOT to do: "Is the F1 score in Block A greater than the number \
of datasets listed in Block B?" -- these two quantities have no substantive relationship to each \
other; it's a coincidental comparison, not genuine reasoning. Only compare things that are \
conceptually related (e.g. the same metric reported under two different conditions, or a method's \
stated purpose vs. its actual measured outcome).

2. NON-LOAD-BEARING BLOCK: Do not write a question whose stem already restates one block's content \
as background framing, leaving the actual answer derivable from the OTHER block alone. Example of \
what NOT to do: "Method X is proposed to solve problem Y [this restates Block B]. According to the \
text, what does Table 3 show about its efficiency? [only Block A is actually needed]" -- if you \
removed one block entirely, the question should become genuinely unanswerable, not just slightly \
less informative.

3. COMPOUND QUESTION: Do not staple two independently-answerable sub-questions together with "and". \
Example of what NOT to do: "What issue does approach A address, and how successful was approach B?" \
-- if each half can be answered by looking at only one block, with no connection between the two \
answers, this is not multi-hop reasoning, it's two single-hop questions in a trenchcoat.

A genuinely good multi-hop question requires BOTH blocks to arrive at ONE synthesized conclusion \
that neither block supports on its own -- e.g. one block establishes what something IS or WHAT IT'S \
FOR, and the other reports a fact ABOUT it that only makes sense once you know the first block's \
context.

Block A:
\"\"\"
{content_a}
\"\"\"

Block B:
\"\"\"
{content_b}
\"\"\"

Return the question, the correct answer, and honestly self-assess (do not just default to true):
- reasoning_uses_gold_blocks_only: the answer requires no outside knowledge
- both_blocks_strictly_necessary: if you deleted either block, the question would become unanswerable
- comparison_is_substantively_related: if this involves comparing two things, they have a real \
conceptual relationship, not a coincidental numeric one
- answer_is_single_synthesized_conclusion: the answer is ONE connected conclusion, not two \
independently-derived facts joined by "and"
"""

MULTI_HOP_CROSS_PROMPT = """You are creating a QA benchmark item requiring MULTI-HOP reasoning \
ACROSS TWO DIFFERENT research papers. Below is one block from each paper. Write ONE question that \
requires genuinely SYNTHESIZING both blocks into a single connected answer.

AVOID THESE THREE COMMON MISTAKES:

1. ARBITRARY COMPARISON: Do not compare two numbers/facts just because they both happen to be \
numeric or nameable. Only compare things that are conceptually related (e.g. the same metric/\
benchmark reported by both papers, or one paper's stated method vs. the other's measured result on \
a comparable task). Do not compare, say, one paper's parameter count against the other's dataset \
count just because both are numbers.

2. NON-LOAD-BEARING BLOCK: Do not restate one paper's block as background framing in the question \
stem such that the answer is really only coming from the other paper's block. Both blocks must be \
indispensable -- if you deleted either one, the question should become genuinely unanswerable.

3. COMPOUND QUESTION: Do not staple two independently-answerable sub-questions together with "and". \
Each half being answerable from only one paper, with no real connection between the two answers, is \
not multi-hop reasoning.

A genuinely good cross-paper question requires BOTH papers' blocks to arrive at ONE synthesized \
conclusion -- e.g. comparing how two papers each measure or report on a genuinely comparable concept.

Paper A ({paper_a_id}) block:
\"\"\"
{content_a}
\"\"\"

Paper B ({paper_b_id}) block:
\"\"\"
{content_b}
\"\"\"

Return the question, the correct answer, and honestly self-assess (do not just default to true):
- reasoning_uses_gold_blocks_only: the answer requires no outside knowledge
- both_blocks_strictly_necessary: if you deleted either block, the question would become unanswerable
- comparison_is_substantively_related: if this involves comparing two things, they have a real \
conceptual relationship, not a coincidental numeric one
- answer_is_single_synthesized_conclusion: the answer is ONE connected conclusion, not two \
independently-derived facts joined by "and"
"""

# ---------------------------------------------------------------------------
# Per-category generation functions
# ---------------------------------------------------------------------------

def make_qa_record(category, question, answer, gold_block_ids, paper_ids,
                    cross_paper=False, manual_transcription_flag=False):
    return {
        "qa_id": f"qa_{uuid.uuid4().hex[:8]}",
        "category": category,
        "question": question,
        "answer": answer,
        "gold_block_ids": gold_block_ids,
        "paper_ids": paper_ids,
        "cross_paper": cross_paper,
        "manual_transcription_flag": manual_transcription_flag,
        "verified": False,
        "notes": "",
    }


def generate_plain_factual(papers, n_target, master_qa, used_block_ids):
    have = sum(1 for qa in master_qa if qa["category"] == "plain_factual")
    pool = []
    for paper_id, paper in papers.items():
        for b in get_text_blocks(paper):
            if len(block_text(b)) > 200 and b["block_id"] not in used_block_ids:
                pool.append((paper_id, b))
    random.shuffle(pool)

    for paper_id, block in pool:
        if have >= n_target:
            break
        prompt = PLAIN_FACTUAL_PROMPT.format(content=block_text(block))
        resp = call_gemini(prompt, model=GEMINI_MODEL_FLASH)
        if not resp or not resp.get("reasoning_uses_gold_blocks_only"):
            continue
        master_qa.append(make_qa_record(
            "plain_factual", resp["question"], resp["answer"],
            [block["block_id"]], [paper_id],
        ))
        have += 1
        save_qa(master_qa)
        print(f"  plain_factual: {have}/{n_target} (checkpointed)")


def generate_table_lookup(papers, n_target, master_qa, used_block_ids):
    have = sum(1 for qa in master_qa if qa["category"] == "table_lookup")
    pool = []
    for paper_id, paper in papers.items():
        for tbl in get_table_blocks(paper_id, paper):
            if tbl["block_id"] in used_block_ids:
                continue
            prose_refs = get_prose_refs_for_table(paper, tbl["block_id"])
            pool.append((paper_id, paper, tbl, prose_refs))
    random.shuffle(pool)

    for paper_id, paper, tbl, prose_refs in pool:
        if have >= n_target:
            break
        use_prose = bool(prose_refs) and random.random() < 0.4  # ~40% prose-reference variant
        gold_ids = [tbl["block_id"]]
        prose_block, note, clause = "", "", " (from the table cells directly)"
        if use_prose:
            prose = random.choice(prose_refs)
            prose_block = f'\nProse referencing this table:\n"""\n{block_text(prose)}\n"""\n'
            note = ", along with a passage of prose that references it"
            clause = " (the value may appear as prose referencing the table, not only as a table cell)"
            gold_ids.append(prose["block_id"])

        prompt = TABLE_LOOKUP_PROMPT.format(
            table_content=block_text(tbl),
            prose_context_note=note,
            prose_context_clause=clause,
            prose_block=prose_block,
        )
        resp = call_gemini(prompt, model=GEMINI_MODEL_PRO)
        if not resp or not resp.get("reasoning_uses_gold_blocks_only"):
            continue
        master_qa.append(make_qa_record(
            "table_lookup", resp["question"], resp["answer"],
            gold_ids, [paper_id],
            manual_transcription_flag=is_manual_transcription(paper_id, paper, tbl),
        ))
        have += 1
        save_qa(master_qa)
        print(f"  table_lookup: {have}/{n_target} (checkpointed)")


def generate_figure_caption(papers, n_target, master_qa, used_block_ids):
    have = sum(1 for qa in master_qa if qa["category"] == "figure_caption")
    pool = []
    for paper_id, paper in papers.items():
        for fig in get_figure_blocks(paper_id, paper):
            if fig["block_id"] in used_block_ids:
                continue
            caption = get_caption_for(paper, fig["block_id"])
            if caption:
                pool.append((paper_id, fig, caption))
    random.shuffle(pool)

    for paper_id, fig, caption in pool:
        if have >= n_target:
            break
        prompt = FIGURE_CAPTION_PROMPT.format(caption_content=block_text(caption))
        resp = call_gemini(prompt, model=GEMINI_MODEL_FLASH)
        if not resp or not resp.get("reasoning_uses_gold_blocks_only"):
            continue
        master_qa.append(make_qa_record(
            "figure_caption", resp["question"], resp["answer"],
            [fig["block_id"], caption["block_id"]], [paper_id],
        ))
        have += 1
        save_qa(master_qa)
        print(f"  figure_caption: {have}/{n_target} (checkpointed)")


def _passes_multi_hop_checks(resp):
    if not resp:
        return False
    return bool(
        resp.get("reasoning_uses_gold_blocks_only")
        and resp.get("both_blocks_strictly_necessary")
        and resp.get("comparison_is_substantively_related")
        and resp.get("answer_is_single_synthesized_conclusion")
    )


def generate_multi_hop(papers, n_target, master_qa, used_pairs,
                        within_frac=MULTI_HOP_WITHIN_PAPER_FRAC):
    n_within = round(n_target * within_frac)
    n_cross = n_target - n_within
    have_within = sum(1 for qa in master_qa
                       if qa["category"] == "multi_hop_comparative" and not qa["cross_paper"])
    have_cross = sum(1 for qa in master_qa
                      if qa["category"] == "multi_hop_comparative" and qa["cross_paper"])

    # --- within-paper ---
    within_pool = []
    for paper_id, paper in papers.items():
        substantive = [
            b for b in paper
            if b.get("block_type") in ("table", "text")
            and len(block_text(b)) > 100
            and not is_boilerplate_content(b)
        ]
        if len(substantive) >= 2:
            within_pool.append((paper_id, substantive))
    random.shuffle(within_pool)

    MAX_PAIR_ATTEMPTS_PER_PAPER = 5

    for paper_id, blocks in within_pool:
        if have_within >= n_within:
            break
        tried_this_paper = set()
        for _ in range(MAX_PAIR_ATTEMPTS_PER_PAPER):
            if have_within >= n_within:
                break
            if len(tried_this_paper) >= len(blocks) * (len(blocks) - 1) // 2:
                break  # exhausted all possible pairs from this paper
            a, b = random.sample(blocks, 2)
            pair_key = tuple(sorted([a["block_id"], b["block_id"]]))
            if pair_key in used_pairs or pair_key in tried_this_paper:
                continue
            tried_this_paper.add(pair_key)
            prompt = MULTI_HOP_WITHIN_PROMPT.format(content_a=block_text(a), content_b=block_text(b))
            resp = call_gemini(prompt, model=GEMINI_MODEL_PRO, response_schema=MULTI_HOP_RESPONSE_SCHEMA)
            if not _passes_multi_hop_checks(resp):
                continue
            master_qa.append(make_qa_record(
                "multi_hop_comparative", resp["question"], resp["answer"],
                [a["block_id"], b["block_id"]], [paper_id], cross_paper=False,
            ))
            used_pairs.add(pair_key)
            have_within += 1
            save_qa(master_qa)
            print(f"  multi_hop (within): {have_within}/{n_within} (checkpointed)")

    # --- cross-paper ---
    paper_ids = list(papers.keys())
    attempts = 0
    while have_cross < n_cross and attempts < (n_cross * 40):
        attempts += 1
        pid_a, pid_b = random.sample(paper_ids, 2)
        blocks_a = [b for b in papers[pid_a] if b.get("block_type") in ("table", "text")
                    and len(block_text(b)) > 100 and not is_boilerplate_content(b)]
        blocks_b = [b for b in papers[pid_b] if b.get("block_type") in ("table", "text")
                    and len(block_text(b)) > 100 and not is_boilerplate_content(b)]
        if not blocks_a or not blocks_b:
            continue
        a, b = random.choice(blocks_a), random.choice(blocks_b)
        pair_key = tuple(sorted([a["block_id"], b["block_id"]]))
        if pair_key in used_pairs:
            continue
        prompt = MULTI_HOP_CROSS_PROMPT.format(
            paper_a_id=pid_a, content_a=block_text(a),
            paper_b_id=pid_b, content_b=block_text(b),
        )
        resp = call_gemini(prompt, model=GEMINI_MODEL_PRO, response_schema=MULTI_HOP_RESPONSE_SCHEMA)
        if not _passes_multi_hop_checks(resp):
            continue
        master_qa.append(make_qa_record(
            "multi_hop_comparative", resp["question"], resp["answer"],
            [a["block_id"], b["block_id"]], [pid_a, pid_b], cross_paper=True,
        ))
        used_pairs.add(pair_key)
        have_cross += 1
        save_qa(master_qa)
        print(f"  multi_hop (cross): {have_cross}/{n_cross} (checkpointed)")


# ---------------------------------------------------------------------------
# Batch runner
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume from an existing qa_pairs_raw.json instead of starting over. "
             "Already-generated items (and the blocks they used) are kept and skipped; "
             "generation continues only for the remaining shortfall in each category.",
    )
    return parser.parse_args()


REJECTED_LOG_PATH = "qa_rejected_log.json"  # written by verify_qa_pairs.py's reject()/save()


def load_rejected_material(path=REJECTED_LOG_PATH):
    if not os.path.exists(path):
        return {}, set()
    with open(path, "r", encoding="utf-8") as f:
        rejected = json.load(f)
    singles_by_cat = {}
    pairs = set()
    for qa in rejected:
        cat = qa.get("category")
        ids = qa.get("gold_block_ids", [])
        if len(ids) == 1:
            singles_by_cat.setdefault(cat, set()).add(ids[0])
        elif len(ids) >= 2:
            pairs.add(tuple(sorted(ids[:2])))
    return singles_by_cat, pairs


def main():
    if "GEMINI_API_KEY" not in os.environ:
        raise RuntimeError("Set GEMINI_API_KEY environment variable before running.")

    args = parse_args()

    print("Loading parsed papers...")
    papers = load_all_papers()
    print(f"  Loaded {len(papers)} papers.")

    if args.resume:
        all_qa = load_existing_qa()
        print(f"  --resume: loaded {len(all_qa)} existing QA pairs from {OUTPUT_PATH}")
    else:
        if os.path.exists(OUTPUT_PATH):
            print(f"  WARNING: {OUTPUT_PATH} already exists and will be OVERWRITTEN "
                  f"(run with --resume to continue it instead).")
        all_qa = []

    rejected_singles, rejected_pairs = load_rejected_material()
    n_rejected_singles = sum(len(v) for v in rejected_singles.values())
    if n_rejected_singles or rejected_pairs:
        print(f"  Loaded {REJECTED_LOG_PATH}: permanently excluding "
              f"{n_rejected_singles} previously-rejected block(s) and "
              f"{len(rejected_pairs)} previously-rejected pair(s) from resampling.")

    def used_ids_for(category):
        used = set()
        for qa in all_qa:
            if qa["category"] == category:
                used.update(qa["gold_block_ids"])
        used.update(rejected_singles.get(category, set()))
        return used

    used_pairs = set(rejected_pairs)
    for qa in all_qa:
        if qa["category"] == "multi_hop_comparative" and len(qa["gold_block_ids"]) == 2:
            used_pairs.add(tuple(sorted(qa["gold_block_ids"])))

    print("Generating plain_factual...")
    generate_plain_factual(papers, TARGET_COUNTS["plain_factual"], all_qa,
                            used_ids_for("plain_factual"))

    print("Generating table_lookup...")
    generate_table_lookup(papers, TARGET_COUNTS["table_lookup"], all_qa,
                           used_ids_for("table_lookup"))

    print("Generating figure_caption...")
    generate_figure_caption(papers, TARGET_COUNTS["figure_caption"], all_qa,
                             used_ids_for("figure_caption"))

    print("Generating multi_hop_comparative...")
    generate_multi_hop(papers, TARGET_COUNTS["multi_hop_comparative"], all_qa, used_pairs)

    print(f"\nDone. {len(all_qa)} total QA pairs saved to {OUTPUT_PATH}")
    print("Category breakdown:")
    from collections import Counter
    print(Counter(qa["category"] for qa in all_qa))


if __name__ == "__main__":
    main()
