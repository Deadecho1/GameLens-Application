import json
import os

from openai import OpenAI

from .config import OCR_OUTPUT_DIR
from .util import NumpyEncoder

client = OpenAI()

SYSTEM_PROMPT = """
You are analyzing OCR-extracted text lines from roguelike UI screens.

Task: Return ONLY the indices of input lines that are TITLE TEXT.

A "title" is the proper name shown for an option/item/ability/upgrade/choice.

Rules:
- If a single on-screen title is split across multiple OCR lines,
  you MUST include ALL of the lines that together form the title (include fragments), not just one token.
- Do NOT include generic UI instructions (e.g. "CHOOSE ONE", "SELECT", "CONFIRM", "Choose A Card:").
- Do NOT include attribute/field labels used in stat blocks or key/value UI (often end with ":" like "Cast Damage:").
- Do NOT include description/explanation text or flavor text.

Return only the indices.
"""


def extract_title_indices(lines):
    schema = {
        "name": "title_line_indices",
        "schema": {
            "type": "object",
            "properties": {
                "title_indices": {
                    "type": "array",
                    "items": {"type": "integer", "minimum": 0},
                    "description": "Indices in the input array that are titles.",
                }
            },
            "required": ["title_indices"],
            "additionalProperties": False,
        },
    }

    payload = [{"i": i, "text": t} for i, t in enumerate(lines)]

    resp = client.chat.completions.create(
        model="gpt-4.1-mini",
        temperature=0,
        response_format={"type": "json_schema", "json_schema": schema},
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps({"lines": payload})},
        ],
    )
    obj = json.loads(resp.choices[0].message.content)
    return obj["title_indices"]


def extract_titles_exact(lines):
    idxs = extract_title_indices(lines)
    return [lines[i] for i in idxs if 0 <= i < len(lines)]


def get_filtered_lines(lines_dict):
    text_lines_by_img = {}

    for img_name, ocr_lines in lines_dict.items():
        if not ocr_lines:
            text_lines_by_img[img_name] = []
            continue
        text_lines = [t.strip() for (_, t, _) in ocr_lines if t and t.strip()]
        text_lines_by_img[img_name] = text_lines

    # --- API call ---
    title_indices_by_img = {}

    for img_name, text_lines in text_lines_by_img.items():
        if not text_lines:
            title_indices_by_img[img_name] = set()
            continue

        idxs = extract_title_indices(text_lines)
        title_indices_by_img[img_name] = set(
            i for i in idxs if 0 <= i < len(text_lines)
        )

    # --- Filtering + printing removed lines ---
    filtered_lines_dict = {}

    for img_name, ocr_lines in lines_dict.items():
        if not ocr_lines:
            continue
        title_idxs = title_indices_by_img.get(img_name, set())

        kept = []
        removed = []

        text_idx = 0  # index into text_lines

        for poly, text, score in ocr_lines:
            if not text or not text.strip():
                continue

            if text_idx in title_idxs:
                kept.append((poly, text, score))
            else:
                removed.append((text, score))

            text_idx += 1

        if removed:
            print(f"\n{img_name} — removing {len(removed)} lines:")
            for text, score in removed:
                print(f"  - ({score:.2f}) {text}")

        if kept:
            filtered_lines_dict[img_name] = kept

    lines_dict = filtered_lines_dict

    with open(os.path.join(OCR_OUTPUT_DIR, "filtered_lines.json"), "w") as f:
        print(lines_dict)
        json.dump(lines_dict, f, cls=NumpyEncoder, indent=2)

    return lines_dict
