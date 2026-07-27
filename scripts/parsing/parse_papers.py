"""
Structure-aware parser for RAG research papers.

Colab setup:
    !pip install pymupdf pdfplumber camelot-py opencv-python-headless requests
    !apt-get install -y ghostscript
"""

import re
import json
import os
import argparse
import traceback
from pathlib import Path

import fitz
import pdfplumber
import camelot


# Config

SECTION_HEADING_RE = re.compile(
    r"^\s*(\d+(\.\d+)*)\s+[A-Z][A-Za-z0-9\-:,'’ ]{2,80}$"
)
BIBLIOGRAPHY_HEADING_RE = re.compile(
    r"^\s*(references|bibliography)\s*$", re.IGNORECASE
)
TABLE_CAPTION_RE = re.compile(r"^\s*Table\s+([IVXLCDM]+|\d+)\b", re.IGNORECASE)
FIGURE_CAPTION_RE = re.compile(r"^\s*Fig(?:ure)?\.?\s+([IVXLCDM]+|\d+)\b", re.IGNORECASE)
TABLE_REF_RE = re.compile(r"\bTable\s+([IVXLCDM]+|\d+)\b", re.IGNORECASE)
FIGURE_REF_RE = re.compile(r"\bFig(?:ure)?\.?\s+([IVXLCDM]+|\d+)\b", re.IGNORECASE)
EQUATION_LINE_RE = re.compile(r"(\\[a-zA-Z]+|[=∑∫∏√±≤≥≈∞∂∇αβγδθλμσω]{1,}|\b[A-Za-z]\s*=\s*[^,.;]{1,40}$)")

CAPTION_PROSE_VERBS = {
    "reports", "report", "shows", "show", "shown", "summarizes", "summarize",
    "presents", "present", "illustrates", "illustrate", "demonstrates", "demonstrate",
    "depicts", "depict", "displays", "display", "lists", "list", "gives", "give",
    "provides", "provide", "contains", "contain", "describes", "describe",
    "compares", "compare", "indicates", "indicate", "includes", "include",
    "is", "are", "was", "were", "reported", "highlights", "highlight",
    "confirms", "confirm", "suggests", "suggest", "reveals", "reveal",
    "details", "detail", "should",
}

def _is_real_caption(first_line, caption_re):
    m = caption_re.match(first_line)
    if not m:
        return False
    rest = first_line[m.end():].strip()
    if not rest:
        return True
    first_char = rest[0]
    if first_char == ",":
      return False
    if not first_char.isalpha():
        return True
    if first_char.islower():
        return False
    first_word_match = re.match(r"[A-Za-z]+", rest)
    if first_word_match and first_word_match.group(0).lower() in CAPTION_PROSE_VERBS:
        return False
    return True


# Core extraction

def extract_text_blocks(doc):
    """Use PyMuPDF for text blocks with font-size metadata (for heading detection)."""
    pages_blocks = []
    for page_num, page in enumerate(doc, start=1):
        blocks = page.get_text("dict")["blocks"]
        page_blocks = []
        for b in blocks:
            if b.get("type") != 0:  # 0 = text
                continue
            lines_text = []
            max_font_size = 0
            for line in b.get("lines", []):
                line_str = "".join(span["text"] for span in line["spans"])
                lines_text.append(line_str)
                for span in line["spans"]:
                    max_font_size = max(max_font_size, span["size"])
            text = "\n".join(t for t in lines_text if t.strip())
            if not text.strip():
                continue
            page_blocks.append({
                "page": page_num,
                "bbox": list(b["bbox"]),
                "text": text,
                "max_font_size": round(max_font_size, 1),
            })
        pages_blocks.append(page_blocks)
    return pages_blocks

def _looks_like_real_table(df, accuracy, min_accuracy=80):
    if accuracy is not None and accuracy < min_accuracy:
        return False
    if df.shape[0] < 2 or df.shape[1] < 2:
        return False
    cell_lengths = [len(str(c)) for row in df.values.tolist() for c in row if str(c).strip()]
    if not cell_lengths:
        return False
    avg_len = sum(cell_lengths) / len(cell_lengths)
    return avg_len < 40  # <-- tuning required based on paper

def _bbox_overlaps(b1, b2, threshold=0.5):
    if not b1 or not b2:
        return False
    x0 = max(b1[0], b2[0]); y0 = max(b1[1], b2[1])
    x1 = min(b1[2], b2[2]); y1 = min(b1[3], b2[3])
    if x1 <= x0 or y1 <= y0:
        return False
    inter = (x1 - x0) * (y1 - y0)
    area1 = (b1[2]-b1[0]) * (b1[3]-b1[1])
    area2 = (b2[2]-b2[0]) * (b2[3]-b2[1])
    smaller = min(area1, area2)
    return smaller > 0 and (inter / smaller) > threshold

def _is_single_cell_row(row):
    non_empty = [c for c in row if str(c).strip()]
    if len(non_empty) == 1:
        return True, str(non_empty[0]).strip()
    return False, None

def _split_on_embedded_captions(t):
    rows = t.df.values.tolist()
    full_bbox = list(t._bbox) if hasattr(t, "_bbox") else None
    if not rows:
        return [(rows, full_bbox)]

    def bbox_for_range(start, end):
        if hasattr(t, "cells") and full_bbox:
            try:
                cells = t.cells[start:end]
                xs0 = [c.x1 for row in cells for c in row]
                ys0 = [c.y1 for row in cells for c in row]
                xs1 = [c.x2 for row in cells for c in row]
                ys1 = [c.y2 for row in cells for c in row]
                if xs0 and ys0:
                    return [min(xs0), min(ys0), max(xs1), max(ys1)]
            except Exception:
                pass
        return full_bbox

    boundaries = []
    seg_start = 0
    i = 0
    n = len(rows)
    while i < n:
        is_single, text = _is_single_cell_row(rows[i])
        is_caption = is_single and text and (_is_real_caption(text, TABLE_CAPTION_RE) or _is_real_caption(text, FIGURE_CAPTION_RE))
        if is_caption:
            if i > seg_start:
                boundaries.append((seg_start, i))
            i += 1
            while i < n:
                is_single2, _ = _is_single_cell_row(rows[i])
                if is_single2:
                    i += 1
                else:
                    break
            seg_start = i
        else:
            i += 1
    if seg_start < n:
        boundaries.append((seg_start, n))
    if len(boundaries) <= 1:
        return [(rows, full_bbox)]
    return [(rows[s:e], bbox_for_range(s, e)) for s, e in boundaries]

def extract_tables(pdf_path):
    tables = []
    try:
        lattice_found = camelot.read_pdf(pdf_path, pages="all", flavor="lattice")
    except Exception:
        lattice_found = []

    for t in lattice_found:
        acc = t.parsing_report.get("accuracy") if hasattr(t, "parsing_report") else None
        if _looks_like_real_table(t.df, acc):
            for rows, bbox in _split_on_embedded_captions(t):
                tables.append({
                    "page": t.page,
                    "flavor": "lattice",
                    "bbox": bbox,
                    "rows": rows,
                    "accuracy": acc,
                })

    try:
        stream_found = camelot.read_pdf(pdf_path, pages="all", flavor="stream")
    except Exception:
        stream_found = []
    for t in stream_found:
        acc = t.parsing_report.get("accuracy") if hasattr(t, "parsing_report") else None
        if not _looks_like_real_table(t.df, acc):
            continue
        for rows, bbox in _split_on_embedded_captions(t):
            duplicate = any(
                existing["page"] == t.page and _bbox_overlaps(existing["bbox"], bbox)
                for existing in tables
            )
            if not duplicate:
                tables.append({
                    "page": t.page,
                    "flavor": "stream",
                    "bbox": bbox,
                    "rows": rows,
                    "accuracy": acc,
                })
    return tables

def _cluster_drawing_rects(rects, gap=15):
    if not rects:
        return []
    rects = sorted(rects, key=lambda r: (r[1], r[0]))
    clusters = []
    for r in rects:
        placed = False
        for c in clusters:
            if (r[0] <= c[2] + gap and r[2] >= c[0] - gap and
                    r[1] <= c[3] + gap and r[3] >= c[1] - gap):
                c[0] = min(c[0], r[0]); c[1] = min(c[1], r[1])
                c[2] = max(c[2], r[2]); c[3] = max(c[3], r[3])
                placed = True
                break
        if not placed:
            clusters.append(list(r))
    return clusters

def extract_figures(doc, min_drawings=5, min_area=2000):
    figures = []
    for page_num, page in enumerate(doc, start=1):
        for img in page.get_images(full=True):
            xref = img[0]
            for r in page.get_image_rects(xref):
                figures.append({
                    "page": page_num,
                    "bbox": [r.x0, r.y0, r.x1, r.y1],
                    "xref": xref,
                    "source": "raster",
                })
        drawings = page.get_drawings()
        if len(drawings) >= min_drawings:
            rects = [tuple(d["rect"]) for d in drawings if d.get("rect")]
            for cx0, cy0, cx1, cy1 in _cluster_drawing_rects(rects):
                area = (cx1 - cx0) * (cy1 - cy0)
                if area >= min_area:
                    figures.append({
                        "page": page_num,
                        "bbox": [cx0, cy0, cx1, cy1],
                        "xref": None,
                        "source": "vector_cluster",
                    })
    return figures
    

# Assembly into schema

def is_equation_block(text):
    lines = [l for l in text.splitlines() if l.strip()]
    if not lines:
        return False
    hits = sum(1 for l in lines if EQUATION_LINE_RE.search(l))
    return hits / len(lines) > 0.5 and len(lines) <= 5

def detect_section(text, current_section):
    first_line = text.strip().splitlines()[0] if text.strip() else ""
    if SECTION_HEADING_RE.match(first_line.strip()):
        return first_line.strip()
    return current_section

def build_blocks(paper_id, pdf_path):
    doc = fitz.open(pdf_path)
    text_pages = extract_text_blocks(doc)
    tables = extract_tables(pdf_path)
    figures = extract_figures(doc)

    all_blocks = []
    block_counter = 0
    current_section = "Front matter"
    in_bibliography = False

    def next_id(page):
        nonlocal block_counter
        block_counter += 1
        return f"{paper_id}_p{page:03d}_b{block_counter:04d}"

    # --- text blocks ---
    for page_blocks in text_pages:
        for tb in page_blocks:
            text = tb["text"]
            first_line = text.strip().splitlines()[0].strip() if text.strip() else ""
            flattened_start = " ".join(text.strip().split())[:200]

            if BIBLIOGRAPHY_HEADING_RE.match(first_line):
                in_bibliography = True

            current_section = detect_section(text, current_section)

            if in_bibliography:
                block_type = "reference"
            elif is_equation_block(text):
                block_type = "equation"
            elif _is_real_caption(flattened_start, TABLE_CAPTION_RE):
                block_type = "caption"
            elif _is_real_caption(flattened_start, FIGURE_CAPTION_RE):
                block_type = "caption"
            else:
                block_type = "text"

            all_blocks.append({
                "block_id": next_id(tb["page"]),
                "page": tb["page"],
                "section": current_section,
                "block_type": block_type,
                "content": text,
                "bbox": tb["bbox"],
                "refers_to": None,
                "caption_of": None,
            })

    # --- table blocks ---
    for t in tables:
        all_blocks.append({
            "block_id": next_id(t["page"]),
            "page": t["page"],
            "section": current_section,
            "block_type": "table",
            "content": t["rows"],
            "bbox": t["bbox"],
            "refers_to": None,
            "caption_of": None,
        })

    # --- figure blocks ---
    for f in figures:
        all_blocks.append({
            "block_id": next_id(f["page"]),
            "page": f["page"],
            "section": current_section,
            "block_type": "figure",
            "content": None,
            "bbox": f["bbox"],
            "xref": f["xref"],
            "source": f.get("source"),
            "refers_to": None,
            "caption_of": None,
        })

    link_captions_to_objects(all_blocks)
    link_prose_references(all_blocks)

    return all_blocks

def link_captions_to_objects(blocks):
    tables = [b for b in blocks if b["block_type"] == "table"]
    figures = [b for b in blocks if b["block_type"] == "figure"]
    table_captions = []
    figure_captions = []
    for b in blocks:
        if b["block_type"] != "caption":
            continue
        first_line = b["content"].strip().splitlines()[0]
        if TABLE_CAPTION_RE.match(first_line):
            table_captions.append(b)
        elif FIGURE_CAPTION_RE.match(first_line):
            figure_captions.append(b)

    def x_overlaps(bbox1, bbox2, min_frac=0.3):
        if not bbox1 or not bbox2:
            return False
        x0 = max(bbox1[0], bbox2[0]); x1 = min(bbox1[2], bbox2[2])
        if x1 <= x0:
            return False
        width = min(bbox1[2] - bbox1[0], bbox2[2] - bbox2[0])
        return width > 0 and (x1 - x0) / width >= min_frac

    def directional_ok(c, cap_bbox, direction):
        if not c["bbox"] or not cap_bbox:
            return True
        if direction == "object_below":
            return c["bbox"][1] >= cap_bbox[1] - 5
        else:
            return c["bbox"][3] <= cap_bbox[3] + 5

    def greedy_one_to_one(captions, objects, direction):
        candidate_pairs = []
        for cap in captions:
            cap_bbox = cap["bbox"]
            for obj in objects:
                if abs(obj["page"] - cap["page"]) > 1:
                    continue
                if not directional_ok(obj, cap_bbox, direction):
                    continue
                same_col = x_overlaps(obj["bbox"], cap_bbox)
                page_dist = abs(obj["page"] - cap["page"])
                y_dist = abs((obj["bbox"][1] if obj["bbox"] else 0) - (cap_bbox[1] if cap_bbox else 0))
                score = (page_dist, 0 if same_col else 1, y_dist)
                candidate_pairs.append((score, cap["block_id"], obj["block_id"]))

        candidate_pairs.sort(key=lambda p: p[0])
        assignment = {cap["block_id"]: None for cap in captions}
        claimed_objects = set()
        for score, cap_id, obj_id in candidate_pairs:
            if assignment[cap_id] is not None or obj_id in claimed_objects:
                continue
            assignment[cap_id] = obj_id
            claimed_objects.add(obj_id)
        return assignment

    def unmatched_count(assignment):
        return sum(1 for v in assignment.values() if v is None)

    if table_captions:
        attempt_below = greedy_one_to_one(table_captions, tables, "object_below")
        attempt_above = greedy_one_to_one(table_captions, tables, "object_above")
        best_assignment = (
            attempt_below if unmatched_count(attempt_below) <= unmatched_count(attempt_above)
            else attempt_above
        )
        for cap in table_captions:
            cap["caption_of"] = best_assignment[cap["block_id"]]

    if figure_captions:
        figure_assignment = greedy_one_to_one(figure_captions, figures, "object_above")
        for cap in figure_captions:
            cap["caption_of"] = figure_assignment[cap["block_id"]]

def link_prose_references(blocks):
    caption_lookup = {}
    for b in blocks:
        if b["block_type"] == "caption" and b["caption_of"]:
            first_line = b["content"].strip().splitlines()[0]
            m_table = TABLE_CAPTION_RE.match(first_line)
            m_fig = FIGURE_CAPTION_RE.match(first_line)
            if m_table:
                caption_lookup[("table", m_table.group(1))] = b["caption_of"]
            elif m_fig:
                caption_lookup[("figure", m_fig.group(1))] = b["caption_of"]

    for b in blocks:
        if b["block_type"] != "text":
            continue
        refs = []
        for m in TABLE_REF_RE.finditer(b["content"]):
            target = caption_lookup.get(("table", m.group(1)))
            if target:
                refs.append(target)
        for m in FIGURE_REF_RE.finditer(b["content"]):
            target = caption_lookup.get(("figure", m.group(1)))
            if target:
                refs.append(target)
        if refs:
            b["refers_to"] = list(set(refs))


# Entry point

def flag_unreliable_links(blocks):
    captions = [b for b in blocks if b["block_type"] == "caption"]
    target_counts = {}
    for c in captions:
        if c["caption_of"]:
            target_counts[c["caption_of"]] = target_counts.get(c["caption_of"], 0) + 1

    for c in captions:
        if c["caption_of"] is None:
            c["needs_review"] = True
            c["review_reason"] = "no matching table/figure object found nearby"
        elif target_counts.get(c["caption_of"], 0) > 1:
            c["needs_review"] = True
            c["review_reason"] = "shares its linked object with another caption - likely a missed detection"
        else:
            c["needs_review"] = False
            c["review_reason"] = None

    flagged_targets = {c["caption_of"] for c in captions if c.get("needs_review") and c["caption_of"]}
    for b in blocks:
        if b["block_type"] == "text" and b.get("refers_to"):
            if any(t in flagged_targets for t in b["refers_to"]):
                b["needs_review"] = True
                b["review_reason"] = "refers to a table/figure with an unresolved or ambiguous caption link"


def summarize_parse_quality(blocks, paper_id, verbose=True):
    captions = [b for b in blocks if b["block_type"] == "caption"]
    flagged = [b for b in blocks if b.get("needs_review")]
    flagged_items = []
    for b in flagged:
        if b["block_type"] == "caption":
            snippet = b["content"].strip().splitlines()[0][:60]
            flagged_items.append({
                "page": b["page"],
                "type": "caption",
                "snippet": snippet,
                "reason": b["review_reason"],
            })
    if verbose:
        print(f"--- Parse quality report: {paper_id} ---")
        print(f"Captions found: {len(captions)}")
        print(f"Blocks flagged for review: {len(flagged)}")
        for item in flagged_items:
            print(f"  [p{item['page']}] CAPTION '{item['snippet']}' -> {item['reason']}")
        if not flagged_items:
            print("  none -- all captions linked uniquely, nothing to check by hand")
        print("-" * 40)
    return flagged_items

def parse_paper(pdf_path, paper_id, out_path=None, verbose=True):
    blocks = build_blocks(paper_id, pdf_path)
    flag_unreliable_links(blocks)
    flagged_items = summarize_parse_quality(blocks, paper_id, verbose=verbose)
    result = {"paper_id": paper_id, "blocks": blocks, "flagged_items": flagged_items}
    if out_path:
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2, default=str)
    return result

def batch_parse_papers(papers, output_dir="parsed_output"):
    os.makedirs(output_dir, exist_ok=True)
    corpus_summary = []
    failed_papers = []
    for entry in papers:
        pdf_path, paper_id = entry
        print(f"Parsing {paper_id} ({pdf_path})...")
        out_path = os.path.join(output_dir, f"{paper_id}.json")
        try:
            result = parse_paper(pdf_path, paper_id, out_path=out_path, verbose=False)
        except Exception as e:
            tb = traceback.format_exc()
            print(f"  FAILED: {e}")
            print(tb)
            failed_papers.append({"paper_id": paper_id, "pdf_path": pdf_path, "error": str(e), "traceback": tb})
            continue

        blocks = result["blocks"]
        flagged_items = result["flagged_items"]
        corpus_summary.append({
            "paper_id": paper_id,
            "num_blocks": len(blocks),
            "num_captions": sum(1 for b in blocks if b["block_type"] == "caption"),
            "num_flagged": len(flagged_items),
            "flagged_items": flagged_items,
        })
        flag_note = f"{len(flagged_items)} flagged" if flagged_items else "clean"
        print(f"  done: {len(blocks)} blocks, {flag_note}")

    # --- consolidated report ---
    print("\n" + "=" * 60)
    print(f"BATCH SUMMARY: {len(papers)} papers attempted, {len(failed_papers)} failed")
    print("=" * 60)
    for entry in corpus_summary:
        flag_note = f"{entry['num_flagged']} flagged" if entry["num_flagged"] else "clean"
        print(f"  {entry['paper_id']}: {entry['num_blocks']} blocks, "
              f"{entry['num_captions']} captions, {flag_note}")

    if failed_papers:
        print("\nFAILED PAPERS (check these PDFs manually):")
        for f in failed_papers:
            print(f"  {f['paper_id']} ({f['pdf_path']}): {f['error']}")

    total_flagged = sum(e["num_flagged"] for e in corpus_summary)
    if total_flagged:
        print("\n" + "=" * 60)
        print(f"ALL ITEMS NEEDING MANUAL REVIEW ({total_flagged} total):")
        print("=" * 60)
        for entry in corpus_summary:
            if not entry["flagged_items"]:
                continue
            print(f"\n{entry['paper_id']}:")
            for item in entry["flagged_items"]:
                print(f"  [p{item['page']}] {item['snippet']} -> {item['reason']}")
    else:
        print("\nNothing flagged across the whole corpus.")
    combined_path = os.path.join(output_dir, "_corpus_quality_summary.json")
    with open(combined_path, "w") as f:
        json.dump({"papers": corpus_summary, "failed": failed_papers}, f, indent=2, default=str)
    print(f"\nCombined summary written to {combined_path}")
    return {"summary": corpus_summary, "failed": failed_papers}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", required=True)
    ap.add_argument("--paper-id", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    parse_paper(args.pdf, args.paper_id, args.out)
    print(f"Wrote {args.out}")