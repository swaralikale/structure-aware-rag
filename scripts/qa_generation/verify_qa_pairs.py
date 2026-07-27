import os
import json
import glob
from pathlib import Path
from collections import Counter

PARSED_OUTPUT_DIR = "parsed_output"
RAW_QA_PATH = "qa_pairs_raw.json"          
VERIFIED_QA_PATH = "qa_pairs_verified.json"  
REJECTED_LOG_PATH = "qa_rejected_log.json"   # audit trail of removed items + reasons

_papers = {}     
_qa = []         
_rejected = []   # audit log of rejected items


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def _block_text(block):
    content = block.get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        lines = []
        for row in content:
            if isinstance(row, list):
                lines.append(" | ".join(str(c) for c in row))
            else:
                lines.append(str(row))
        return "\n".join(lines)
    return str(content)


def _load_papers(parsed_dir=PARSED_OUTPUT_DIR):
    papers = {}
    for path in glob.glob(os.path.join(parsed_dir, "*.json")):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict) or "blocks" not in data:
            continue
        paper_id = Path(path).stem
        papers[paper_id] = {b["block_id"]: b for b in data["blocks"]}
    return papers


def load():
    global _papers, _qa, _rejected
    _papers = _load_papers()
    print(f"Loaded {len(_papers)} papers.")

    if os.path.exists(VERIFIED_QA_PATH):
        with open(VERIFIED_QA_PATH, "r", encoding="utf-8") as f:
            _qa = json.load(f)
        print(f"Resumed {len(_qa)} QA pairs from {VERIFIED_QA_PATH} "
              f"(prior verification progress found).")
    else:
        with open(RAW_QA_PATH, "r", encoding="utf-8") as f:
            _qa = json.load(f)
        print(f"Loaded {len(_qa)} QA pairs from {RAW_QA_PATH} (fresh start).")

    if os.path.exists(REJECTED_LOG_PATH):
        with open(REJECTED_LOG_PATH, "r", encoding="utf-8") as f:
            _rejected = json.load(f)
    else:
        _rejected = []

    n_verified = sum(1 for qa in _qa if qa["verified"])
    print(f"{n_verified}/{len(_qa)} already verified, {len(_rejected)} previously rejected.")


def save():
    with open(VERIFIED_QA_PATH, "w", encoding="utf-8") as f:
        json.dump(_qa, f, indent=2)
    with open(REJECTED_LOG_PATH, "w", encoding="utf-8") as f:
        json.dump(_rejected, f, indent=2)
    n_verified = sum(1 for qa in _qa if qa["verified"])
    print(f"Saved. {n_verified}/{len(_qa)} verified, {len(_rejected)} rejected (logged).")


# ---------------------------------------------------------------------------
# Lookup helpers
# ---------------------------------------------------------------------------

def _find(qa_id):
    for qa in _qa:
        if qa["qa_id"] == qa_id:
            return qa
    raise KeyError(f"qa_id {qa_id!r} not found (already reviewed/rejected, or a typo?)")


def _resolve_gold_blocks(qa):
    resolved = []
    paper_ids = qa["paper_ids"]
    for block_id in qa["gold_block_ids"]:
        block = None
        for pid in paper_ids:
            block = _papers.get(pid, {}).get(block_id)
            if block:
                resolved.append((pid, block_id, block.get("block_type", "?"), _block_text(block)))
                break
        if block is None:
            resolved.append(("?", block_id, "MISSING", "[block not found in parsed_output -- "
                                                          "check paper_ids/gold_block_ids]"))
    return resolved


# ---------------------------------------------------------------------------
# Review display
# ---------------------------------------------------------------------------

def reject_all_unverified(category, reason):
    global _qa
    to_reject = [qa["qa_id"] for qa in _qa if not qa["verified"] and qa["category"] == category]
    if not to_reject:
        print(f"Nothing unverified in '{category}' to reject.")
        return []
    for qa_id in to_reject:
        reject(qa_id, reason)
    print(f"\nBulk-rejected {len(to_reject)} unverified '{category}' item(s). "
          f"Rerun generation with --resume to backfill all of them.")
    return to_reject

def show_qa(qa_id):
    qa = _find(qa_id)
    print("=" * 78)
    print(f"qa_id: {qa['qa_id']}   category: {qa['category']}   "
          f"verified: {qa['verified']}   cross_paper: {qa['cross_paper']}")
    if qa["manual_transcription_flag"]:
        print("** MANUAL TRANSCRIPTION SOURCE -- extra scrutiny recommended **")
    print("-" * 78)
    print(f"Q: {qa['question']}")
    print(f"A: {qa['answer']}")
    if qa.get("notes"):
        print(f"notes: {qa['notes']}")
    print("-" * 78)
    print("Gold blocks:")
    for pid, bid, btype, text in _resolve_gold_blocks(qa):
        preview = text if len(text) <= 600 else text[:600] + " ...[truncated]"
        print(f"  [{pid} / {bid} / {btype}]")
        print(f"  {preview}")
    print("=" * 78)


def show_unverified(category=None, n=10):
    items = [qa for qa in _qa if not qa["verified"]]
    if category:
        items = [qa for qa in items if qa["category"] == category]
    print(f"{len(items)} unverified" + (f" in '{category}'" if category else "") + f" (showing up to {n}):")
    for qa in items[:n]:
        flag = " [MANUAL-TRANSCRIPTION]" if qa["manual_transcription_flag"] else ""
        print(f"  {qa['qa_id']}  [{qa['category']}]{flag}  {qa['question'][:90]}")


def show_all_unverified():
    for cat in sorted(set(qa["category"] for qa in _qa)):
        show_unverified(category=cat, n=10_000)
        print()

def show_next_batch(n=5, category=None):
    items = [qa for qa in _qa if not qa["verified"]]
    if category:
        items = [qa for qa in items if qa["category"] == category]
    batch = items[:n]
    if not batch:
        print("Nothing left to review" + (f" in '{category}'" if category else "") + ".")
        return []
    for qa in batch:
        show_qa(qa["qa_id"])
    return [qa["qa_id"] for qa in batch]

def show_next_unverified(category=None):
    items = [qa for qa in _qa if not qa["verified"]]
    if category:
        items = [qa for qa in items if qa["category"] == category]
    if not items:
        print("Nothing left to review" + (f" in '{category}'" if category else "") + ".")
        return None
    show_qa(items[0]["qa_id"])
    return items[0]["qa_id"]


# ---------------------------------------------------------------------------
# Actions: approve / edit / reject
# ---------------------------------------------------------------------------

def approve(qa_id, notes=""):
    qa = _find(qa_id)
    qa["verified"] = True
    if notes:
        qa["notes"] = notes
    print(f"Approved {qa_id}.")


def apply_fix(qa_id, question=None, answer=None, notes=None, gold_block_ids=None, paper_ids=None):
    qa = _find(qa_id)
    if question is not None:
        qa["question"] = question
    if answer is not None:
        qa["answer"] = answer
    if notes is not None:
        qa["notes"] = notes
    if gold_block_ids is not None:
        qa["gold_block_ids"] = gold_block_ids
    if paper_ids is not None:
        qa["paper_ids"] = paper_ids
    qa["verified"] = True
    print(f"Fixed and approved {qa_id}.")
    show_qa(qa_id)


def reject(qa_id, reason):
    global _qa
    qa = _find(qa_id)
    qa["_rejection_reason"] = reason
    _rejected.append(qa)
    _qa = [x for x in _qa if x["qa_id"] != qa_id]
    print(f"Rejected {qa_id} (reason: {reason}). "
          f"{qa['category']} now short by 1 -- rerun generation with --resume to backfill.")


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------

def verification_report():
    print("=== Verification progress ===")
    for cat in sorted(set(qa["category"] for qa in _qa)):
        items = [qa for qa in _qa if qa["category"] == cat]
        verified = sum(1 for qa in items if qa["verified"])
        print(f"  {cat}: {verified}/{len(items)} verified")

    print(f"\nTotal: {sum(1 for qa in _qa if qa['verified'])}/{len(_qa)} verified. "
          f"{len(_rejected)} rejected so far.")

    print("\n=== Duplicate gold-block usage (within category) ===")
    any_dupes = False
    for cat in sorted(set(qa["category"] for qa in _qa)):
        items = [qa for qa in _qa if qa["category"] == cat]
        counts = Counter(tuple(qa["gold_block_ids"]) for qa in items)
        dupes = {k: v for k, v in counts.items() if v > 1}
        if dupes:
            any_dupes = True
            print(f"  {cat}: {len(dupes)} block-set(s) used more than once -> {dupes}")
    if not any_dupes:
        print("  None found.")

    manual_transcription_report()


def manual_transcription_report():
    flagged = [qa for qa in _qa if qa["manual_transcription_flag"]]
    if not flagged:
        print("\nNo manual-transcription-flagged items in the current set.")
        return
    verified = sum(1 for qa in flagged if qa["verified"])
    print(f"\n=== Manual-transcription-flagged items: {verified}/{len(flagged)} verified ===")
    for qa in flagged:
        mark = "[OK]" if qa["verified"] else "[ ]"
        print(f"  {mark} {qa['qa_id']}  [{qa['category']}]  {qa['question'][:80]}")


if __name__ == "__main__":
    print("This module is meant to be used interactively, e.g. in Colab:")
    print("  from verify_qa_pairs import *")
    print("  load()")
    print("  verification_report()")
