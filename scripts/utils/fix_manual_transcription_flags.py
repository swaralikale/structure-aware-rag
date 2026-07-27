import json
import shutil
import importlib.util

RAW_QA_PATH = "qa_pairs_raw.json"
BACKUP_PATH = "qa_pairs_raw_backup.json"
GENERATE_SCRIPT_PATH = "scripts/generate_qa_pairs.py"


def load_generate_module(path=GENERATE_SCRIPT_PATH):
    spec = importlib.util.spec_from_file_location("gqp", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def resolve_block(papers, paper_ids, block_id):
    for pid in paper_ids:
        block = papers.get(pid, {}).get(block_id)
        if block:
            return pid, block
    return None, None


def main():
    gqp = load_generate_module()

    print("Loading parsed papers...")
    papers_raw = gqp.load_all_papers()  
    papers_indexed = {
        pid: {b["block_id"]: b for b in blocks} for pid, blocks in papers_raw.items()
    }
    print(f"  Loaded {len(papers_raw)} papers.")

    print(f"Backing up {RAW_QA_PATH} -> {BACKUP_PATH}")
    shutil.copy(RAW_QA_PATH, BACKUP_PATH)

    with open(RAW_QA_PATH, "r", encoding="utf-8") as f:
        qa_pairs = json.load(f)

    changed = 0
    for qa in qa_pairs:
        was_flagged = qa["manual_transcription_flag"]
        now_flagged = False

        for block_id in qa["gold_block_ids"]:
            pid, block = resolve_block(papers_indexed, qa["paper_ids"], block_id)
            if block is None:
                continue  # block not found -- leave flag as-is for this block, check others
            if block.get("block_type") != "table":
                continue  
            if gqp.is_manual_transcription(pid, papers_raw[pid], block):
                now_flagged = True
                break

        if now_flagged != was_flagged:
            changed += 1
        qa["manual_transcription_flag"] = now_flagged

    with open(RAW_QA_PATH, "w", encoding="utf-8") as f:
        json.dump(qa_pairs, f, indent=2)

    n_flagged = sum(1 for qa in qa_pairs if qa["manual_transcription_flag"])
    print(f"\nDone. {changed} item(s) had their flag corrected.")
    print(f"Total now flagged as manual-transcription-sourced: {n_flagged}/{len(qa_pairs)}")
    print("Breakdown by category:")
    for cat in sorted(set(qa["category"] for qa in qa_pairs)):
        items = [qa for qa in qa_pairs if qa["category"] == cat]
        flagged = sum(1 for qa in items if qa["manual_transcription_flag"])
        print(f"  {cat}: {flagged}/{len(items)} flagged")

    print(f"\nOriginal (pre-fix) file preserved at {BACKUP_PATH} in case you need to compare.")


if __name__ == "__main__":
    main()
