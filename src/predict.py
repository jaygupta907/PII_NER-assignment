import json
import argparse
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
from labels import ID2LABEL, LABEL2ID, label_is_pii
import os


def bio_to_spans(text, offsets, label_ids):
    """Convert BIO labels to spans with improved handling."""
    spans = []
    current_label = None
    current_start = None
    current_end = None

    for (start, end), lid in zip(offsets, label_ids):
        if start == 0 and end == 0:
            # Special tokens - end current span
            if current_label is not None:
                spans.append((current_start, current_end, current_label))
                current_label = None
            continue
        
        label = ID2LABEL.get(int(lid), "O")
        if label == "O":
            if current_label is not None:
                spans.append((current_start, current_end, current_label))
                current_label = None
            continue

        try:
            prefix, ent_type = label.split("-", 1)
        except ValueError:
            # Invalid label format
            if current_label is not None:
                spans.append((current_start, current_end, current_label))
                current_label = None
            continue

        if prefix == "B":
            if current_label is not None:
                spans.append((current_start, current_end, current_label))
            current_label = ent_type
            current_start = start
            current_end = end
        elif prefix == "I":
            if current_label == ent_type:
                # Valid continuation
                current_end = end
            else:
                # Invalid I-tag - treat as B-tag to recover
                if current_label is not None:
                    spans.append((current_start, current_end, current_label))
                current_label = ent_type
                current_start = start
                current_end = end

    if current_label is not None:
        spans.append((current_start, current_end, current_label))

    return spans


def filter_low_confidence_spans(spans, logits, offsets, confidence_threshold=0.5):
    """Filter out spans with low confidence predictions."""
    if len(spans) == 0:
        return spans
    
    filtered = []
    probs = torch.softmax(logits, dim=-1)
    
    for start, end, label in spans:
        # Find tokens that overlap with this span
        max_conf = 0.0
        for i, (off_start, off_end) in enumerate(offsets):
            if off_start == 0 and off_end == 0:
                continue
            # Check if this token overlaps with the span
            if not (off_end <= start or off_start >= end):
                # Try B-tag first, then I-tag
                b_label_id = LABEL2ID.get(f"B-{label}", None)
                i_label_id = LABEL2ID.get(f"I-{label}", None)
                label_id = b_label_id if b_label_id is not None else (i_label_id if i_label_id is not None else 0)
                if i < len(probs) and label_id < len(probs[i]):
                    conf = probs[i][label_id].item()
                    max_conf = max(max_conf, conf)
        
        # Keep span if confidence is above threshold
        if max_conf >= confidence_threshold:
            filtered.append((start, end, label))
    
    return filtered


def post_process_spans(spans, text):
    """Apply simple post-processing rules to improve span quality."""
    if len(spans) == 0:
        return spans
    
    processed = []
    for start, end, label in spans:
        # Validate span boundaries
        if start < 0 or end <= start or end > len(text):
            continue
        
        # Remove very short spans (likely noise)
        span_text = text[start:end].strip()
        if len(span_text) < 2:
            continue
        
        # Remove spans that are just whitespace/punctuation
        if span_text.strip() == "" or span_text.strip() in [".", ",", "!", "?"]:
            continue
        
        # Remove spans that are too long (likely errors)
        if end - start > 200:
            continue
        
        processed.append((start, end, label))
    
    # Remove overlapping spans (keep the longer one, or first if same length)
    if len(processed) > 1:
        processed.sort(key=lambda x: (x[0], x[1]))
        non_overlapping = []
        for start, end, label in processed:
            overlap = False
            for prev_start, prev_end, prev_label in non_overlapping:
                # Check if spans overlap
                if not (end <= prev_start or start >= prev_end):
                    overlap = True
                    # Keep the longer span, or keep the first if same length
                    if (end - start) > (prev_end - prev_start):
                        non_overlapping.remove((prev_start, prev_end, prev_label))
                        non_overlapping.append((start, end, label))
                    break
            if not overlap:
                non_overlapping.append((start, end, label))
        processed = non_overlapping
    
    # Merge adjacent spans of the same type
    if len(processed) > 1:
        merged = []
        for i, (start, end, label) in enumerate(processed):
            if i == 0:
                merged.append((start, end, label))
            else:
                prev_start, prev_end, prev_label = merged[-1]
                # Merge if same label and close together (within 2 chars)
                if prev_label == label and start - prev_end <= 2:
                    merged[-1] = (prev_start, end, label)
                else:
                    merged.append((start, end, label))
        processed = merged
    
    return processed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", default="out")
    ap.add_argument("--model_name", default=None)
    ap.add_argument("--input", default="data/dev.jsonl")
    ap.add_argument("--output", default="out/dev_pred.json")
    ap.add_argument("--max_length", type=int, default=384)
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
                
                # Apply confidence thresholding for low-confidence predictions
                probs = torch.softmax(logits, dim=-1)
                max_probs = probs.max(dim=-1)[0]
                pred_ids = logits.argmax(dim=-1).cpu().tolist()
                
                # Filter low-confidence predictions (set to O)
                for i, (start, end) in enumerate(offsets):
                    if start == 0 and end == 0:
                        pred_ids[i] = 0  # O for special tokens
                    elif i < len(max_probs) and max_probs[i].item() < 0.3:
                        # Very low confidence - set to O
                        pred_ids[i] = 0

            spans = bio_to_spans(text, offsets, pred_ids)
            
            # Apply post-processing
            spans = post_process_spans(spans, text)
            
            # Optional: filter by confidence (can be disabled for speed)
            # spans = filter_low_confidence_spans(spans, logits, offsets, confidence_threshold=0.4)
            
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
