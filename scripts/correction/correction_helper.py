"""
Helper for reviewing and correcting the 'needs_review' flagged captions
produced by parse_papers.py / batch_parse_papers().

"""

import json
import os

def load_paper(paper_id, output_dir="parsed_output"):
    with open(os.path.join(output_dir, f"{paper_id}.json")) as f:
        return json.load(f)

def save_paper(paper_id, data, output_dir="parsed_output"):
    with open(os.path.join(output_dir, f"{paper_id}.json"), "w") as f:
        json.dump(data, f, indent=2, default=str)
    print(f"Saved {paper_id}.json")

def _table_preview(content, max_rows=2):
    if not isinstance(content, list):
        return str(content)[:100]
    return " | ".join(str(row) for row in content[:max_rows])

def show_flagged(paper_id, output_dir="parsed_output"):
    data = load_paper(paper_id, output_dir)
    blocks = data["blocks"]
    all_captions = [b for b in blocks if b["block_type"] == "caption"]
    flagged = [b for b in all_captions if b.get("needs_review")]

    if not flagged:
        print(f"{paper_id}: nothing flagged.")
        return

    for cap in flagged:
        obj_type = "table" if cap["content"].strip().lower().startswith(("table", "TABLE")) else "figure"
        print("=" * 70)
        print(f"CAPTION [{cap['block_id']}] page {cap['page']}: {cap['content'][:90]}")
        print(f"  currently linked to: {cap['caption_of']}   reason: {cap['review_reason']}")

        nearby = [b for b in blocks if b["block_type"] == obj_type and abs(b["page"] - cap["page"]) <= 1]
        if not nearby:
            print(f"  No {obj_type} objects at all near page {cap['page']} - likely never detected.")
            print(f"  -> use add_missing_table(...) if you want to hand-enter it, "
                  f"or apply_fix('{paper_id}', '{cap['block_id']}', None) to confirm and move on.")
            continue

        print(f"  Nearby {obj_type} candidates:")
        for obj in nearby:
            claimed_by = [c["block_id"] for c in all_captions if c["caption_of"] == obj["block_id"]]
            print(f"    [{obj['block_id']}] page {obj['page']}  (claimed by: {claimed_by or 'nobody'})")
            if obj_type == "table":
                print(f"        preview: {_table_preview(obj['content'])}")


def show_all_flagged(paper_ids, output_dir="parsed_output"):
    for paper_id in paper_ids:
        show_flagged(paper_id, output_dir)


def apply_fix(paper_id, caption_block_id, correct_object_block_id, output_dir="parsed_output"):
    data = load_paper(paper_id, output_dir)
    blocks = data["blocks"]
    cap = next((b for b in blocks if b["block_id"] == caption_block_id), None)
    if cap is None:
        print(f"No block with id {caption_block_id} found in {paper_id}.")
        return

    if correct_object_block_id:
        target = next((b for b in blocks if b["block_id"] == correct_object_block_id), None)
        if target is None:
            print(f"No block with id {correct_object_block_id} found in {paper_id} - not applied.")
            return
        cap["caption_of"] = correct_object_block_id
        cap["needs_review"] = False
        cap["review_reason"] = None
        print(f"[{caption_block_id}] -> [{correct_object_block_id}], flag cleared.")
    else:
        cap["caption_of"] = None
        cap["needs_review"] = True
        cap["review_reason"] = "confirmed manually: table/figure was never detected by the parser"
        print(f"[{caption_block_id}] confirmed as a genuine gap (left flagged with a manual note).")

    save_paper(paper_id, data, output_dir)


def add_missing_table(paper_id, page, rows, caption_block_id, bbox=None, output_dir="parsed_output"):
    data = load_paper(paper_id, output_dir)
    blocks = data["blocks"]
    cap = next((b for b in blocks if b["block_id"] == caption_block_id), None)
    if cap is None:
        print(f"No caption block with id {caption_block_id} found in {paper_id}.")
        return

    existing_ids = [b["block_id"] for b in blocks]
    new_id = f"{paper_id}_p{page:03d}_manual{sum(1 for i in existing_ids if 'manual' in i) + 1:03d}"

    new_block = {
        "block_id": new_id,
        "page": page,
        "section": cap.get("section"),
        "block_type": "table",
        "content": rows,
        "bbox": bbox,
        "refers_to": None,
        "caption_of": None,
    }
    blocks.append(new_block)
    cap["caption_of"] = new_id
    cap["needs_review"] = False
    cap["review_reason"] = None

    save_paper(paper_id, data, output_dir)
    print(f"Added new table block [{new_id}] and linked caption [{caption_block_id}] to it.")

def add_missing_figure(paper_id, page, caption_block_id, content=None, bbox=None, output_dir="parsed_output"):
  data = load_paper(paper_id, output_dir)
  blocks = data["blocks"]
  cap = next((b for b in blocks if b["block_id"] == caption_block_id), None)
  if cap is None:
    print(f"No caption block with id {caption_block_id} found in {paper_id}.")
    return

  existing_ids = [b["block_id"] for b in blocks]
  new_id = f"{paper_id}_p{page:03d}_manual{sum(1 for i in existing_ids if 'manual' in i) + 1:03d}"

  new_block = {
      "block_id": new_id,
      "page": page,
      "section": cap.get("section"),
      "block_type": "figure",
      "content": content,
      "bbox": bbox,
      "xref": None,
      "source": "manual",
      "refers_to": None,
      "caption_of": None,
  }
  blocks.append(new_block)
  cap["caption_of"] = new_id
  cap["needs_review"] = False
  cap["review_reason"] = None

  save_paper(paper_id, data, output_dir)
  print(f"Added new figure block [{new_id}] and linked caption [{caption_block_id}] to it.")