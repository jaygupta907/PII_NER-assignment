import json
import argparse
import torch
import re
from transformers import AutoTokenizer, AutoModelForTokenClassification
from labels import ID2LABEL, label_is_pii
import os


def is_valid_email(text):
    """
    Validate EMAIL using regex pattern.
    This helps filter out false positives for EMAIL class.
    """
    # Basic email pattern: local@domain
    # More lenient pattern to catch various email formats
    email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    
    # Clean the text (remove extra whitespace)
    text_clean = text.strip()
    
    # Check if it matches email pattern
    if re.match(email_pattern, text_clean):
        return True
    
    # Also check for emails with subdomains or longer TLDs
    extended_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}(\.[a-zA-Z]{2,})?$'
    if re.match(extended_pattern, text_clean):
        return True
    
    return False


def bio_to_spans(text, offsets, label_ids):
    spans = []
    current_label = None
    current_start = None
    current_end = None

    for (start, end), lid in zip(offsets, label_ids):
        if start == 0 and end == 0:
            continue
        label = ID2LABEL.get(int(lid), "O")
        if label == "O":
            if current_label is not None:
                spans.append((current_start, current_end, current_label))
                current_label = None
            continue

        prefix, ent_type = label.split("-", 1)
        if prefix == "B":
            if current_label is not None:
                spans.append((current_start, current_end, current_label))
            current_label = ent_type
            current_start = start
            current_end = end
        elif prefix == "I":
            if current_label == ent_type:
                current_end = end
            else:
                if current_label is not None:
                    spans.append((current_start, current_end, current_label))
                current_label = ent_type
                current_start = start
                current_end = end

    if current_label is not None:
        spans.append((current_start, current_end, current_label))

    return spans


def find_email_patterns(text):
    """
    Find EMAIL patterns in text using regex as a second-pass detection.
    This helps improve recall for EMAIL class.
    Returns list of (start, end) tuples.
    """
    email_pattern = r'\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b'
    matches = []
    for match in re.finditer(email_pattern, text):
        matches.append((match.start(), match.end()))
    return matches


def validate_email_spans(text, spans):
    """
    Post-process spans to validate EMAIL predictions using regex.
    Removes invalid EMAIL predictions to improve precision.
    Also adds missed EMAIL patterns to improve recall.
    """
    validated_spans = []
    email_spans_found = set()
    
    # First, validate existing EMAIL predictions
    for start, end, label in spans:
        if label == "EMAIL":
            # Extract the text span
            span_text = text[start:end]
            # Validate using regex
            if is_valid_email(span_text):
                validated_spans.append((start, end, label))
                email_spans_found.add((start, end))
            # If invalid, skip this EMAIL prediction (improves precision)
        else:
            # Keep non-EMAIL spans as-is
            validated_spans.append((start, end, label))
    
    # Second pass: Find EMAIL patterns that might have been missed
    # This helps improve recall
    email_patterns = find_email_patterns(text)
    for start, end in email_patterns:
        # Only add if not already found and doesn't overlap with existing spans
        if (start, end) not in email_spans_found:
            # Check for overlap with existing spans
            overlaps = False
            for s, e, _ in validated_spans:
                if not (end <= s or start >= e):  # Overlap detected
                    overlaps = True
                    break
            if not overlaps:
                validated_spans.append((start, end, "EMAIL"))
    
    # Sort spans by start position
    validated_spans.sort(key=lambda x: x[0])
    
    return validated_spans


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", default="out")
    ap.add_argument("--model_name", default=None)
    ap.add_argument("--input", default="data/dev.jsonl")
    ap.add_argument("--output", default="out/dev_pred.json")
    ap.add_argument("--max_length", type=int, default=256)
    ap.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_dir if args.model_name is None else args.model_name)
    model = AutoModelForTokenClassification.from_pretrained(args.model_dir)
    model.to(args.device)
    model.eval()

    results = {}

    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            text = obj["text"]
            uid = obj["id"]

            enc = tokenizer(
                text,
                return_offsets_mapping=True,
                truncation=True,
                max_length=args.max_length,
                return_tensors="pt",
            )
            offsets = enc["offset_mapping"][0].tolist()
            input_ids = enc["input_ids"].to(args.device)
            attention_mask = enc["attention_mask"].to(args.device)

            with torch.no_grad():
                out = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = out.logits[0]
                pred_ids = logits.argmax(dim=-1).cpu().tolist()

            spans = bio_to_spans(text, offsets, pred_ids)
            # Validate EMAIL predictions using regex
            spans = validate_email_spans(text, spans)
            
            ents = []
            for s, e, lab in spans:
                ents.append(
                    {
                        "start": int(s),
                        "end": int(e),
                        "label": lab,
                        "pii": bool(label_is_pii(lab)),
                    }
                )
            results[uid] = ents

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"Wrote predictions for {len(results)} utterances to {args.output}")


if __name__ == "__main__":
    main()
