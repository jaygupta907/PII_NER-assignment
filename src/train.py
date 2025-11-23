import os
import argparse
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
from collections import defaultdict

from dataset import PIIDataset, collate_batch
from labels import LABELS, ID2LABEL
from model import create_model


class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance by focusing on hard examples."""
    def __init__(self, alpha=1.0, gamma=2.0, ignore_index=-100):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
    
    def forward(self, logits, targets):
        logits_flat = logits.view(-1, logits.size(-1))
        targets_flat = targets.view(-1)
        
        ce_loss = F.cross_entropy(logits_flat, targets_flat, 
                                  ignore_index=self.ignore_index, 
                                  reduction='none')
        
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        mask = (targets_flat != self.ignore_index).float()
        focal_loss = focal_loss * mask
        
        return focal_loss.sum() / (mask.sum() + 1e-8)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", default="bert-base-uncased")
    ap.add_argument("--train", default="data/train.jsonl")
    ap.add_argument("--dev", default="data/dev.jsonl")
    ap.add_argument("--out_dir", default="out")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--max_length", type=int, default=384)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--gradient_clip", type=float, default=1.0)
    ap.add_argument("--warmup_ratio", type=float, default=0.1)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--scheduler", type=str, default="cosine", choices=["linear", "cosine"])
    ap.add_argument("--gradient_accumulation_steps", type=int, default=4)
    ap.add_argument("--use_focal_loss", action="store_true")
    ap.add_argument("--no_focal_loss", dest="use_focal_loss", action="store_false")
    ap.set_defaults(use_focal_loss=True)
    ap.add_argument("--focal_gamma", type=float, default=1.5)
    ap.add_argument("--focal_alpha", type=float, default=0.75)
    ap.add_argument("--label_smoothing", type=float, default=0.0)
    ap.add_argument("--eval_steps", type=int, default=None, help="Evaluate every N steps (None = end of epoch)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return ap.parse_args()


def bio_to_spans(text, offsets, label_ids):
    """Convert BIO labels to spans."""
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


def evaluate_model(model, tokenizer, dev_path, device, max_length):
    """Evaluate model on dev set and return Macro-F1."""
    model.eval()
    dev_ds = PIIDataset(dev_path, tokenizer, LABELS, max_length=max_length, is_train=False)
    
    # Load gold labels
    gold = {}
    with open(dev_path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            uid = obj["id"]
            spans = []
            for e in obj.get("entities", []):
                spans.append((e["start"], e["end"], e["label"]))
            gold[uid] = spans
    
    # Get predictions
    pred = {}
    with torch.no_grad():
        for item in dev_ds:
            uid = item["id"]
            text = item["text"]
            input_ids = torch.tensor([item["input_ids"]], device=device)
            attention_mask = torch.tensor([item["attention_mask"]], device=device)
            offsets = item["offset_mapping"]
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits[0]
            pred_ids = logits.argmax(dim=-1).cpu().tolist()
            
            spans = bio_to_spans(text, offsets, pred_ids)
            pred[uid] = spans
    
    # Compute metrics
    labels = set()
    for spans in gold.values():
        for _, _, lab in spans:
            labels.add(lab)
    
    tp = defaultdict(int)
    fp = defaultdict(int)
    fn = defaultdict(int)
    
    for uid in gold.keys():
        g_spans = set(gold.get(uid, []))
        p_spans = set(pred.get(uid, []))
        
        for span in p_spans:
            if span in g_spans:
                tp[span[2]] += 1
            else:
                fp[span[2]] += 1
        for span in g_spans:
            if span not in p_spans:
                fn[span[2]] += 1
    
    def compute_f1(tp_val, fp_val, fn_val):
        prec = tp_val / (tp_val + fp_val) if tp_val + fp_val > 0 else 0.0
        rec = tp_val / (tp_val + fn_val) if tp_val + fn_val > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec > 0 else 0.0
        return prec, rec, f1
    
    macro_f1_sum = 0.0
    macro_count = 0
    for lab in sorted(labels):
        p, r, f1 = compute_f1(tp[lab], fp[lab], fn[lab])
        macro_f1_sum += f1
        macro_count += 1
    
    macro_f1 = macro_f1_sum / max(1, macro_count)
    model.train()
    return macro_f1


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    train_ds = PIIDataset(args.train, tokenizer, LABELS, max_length=args.max_length, is_train=True)

    train_dl = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_batch(b, pad_token_id=tokenizer.pad_token_id),
    )

    model = create_model(args.model_name, dropout=args.dropout, label_smoothing=args.label_smoothing)
    model.to(args.device)
    model.train()

    # Use focal loss if specified
    if args.use_focal_loss:
        loss_fn = FocalLoss(alpha=args.focal_alpha, gamma=args.focal_gamma, ignore_index=-100)
        print(f"Using Focal Loss (alpha={args.focal_alpha}, gamma={args.focal_gamma})")
    else:
        loss_fn = None
        print("Using standard CrossEntropyLoss")

    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=args.lr, 
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999),
        eps=1e-8
    )
    # Account for gradient accumulation in total steps
    total_steps = (len(train_dl) // args.gradient_accumulation_steps) * args.epochs
    warmup_steps = int(args.warmup_ratio * total_steps)
    
    if args.scheduler == "cosine":
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, 
            num_warmup_steps=warmup_steps, 
            num_training_steps=total_steps,
            num_cycles=0.5
        )
    else:
        scheduler = get_linear_schedule_with_warmup(
            optimizer, 
            num_warmup_steps=warmup_steps, 
            num_training_steps=total_steps
        )

    best_macro_f1 = 0.0
    patience = 10  # Even more patience for exploration
    patience_counter = 0
    no_improve_epochs = 0

    for epoch in range(args.epochs):
        running_loss = 0.0
        model.train()
        optimizer.zero_grad()
        
        for step, batch in enumerate(tqdm(train_dl, desc=f"Epoch {epoch+1}/{args.epochs}")):
            input_ids = torch.tensor(batch["input_ids"], device=args.device)
            attention_mask = torch.tensor(batch["attention_mask"], device=args.device)
            labels = torch.tensor(batch["labels"], device=args.device)

            if loss_fn is not None:
                # Use focal loss
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                loss = loss_fn(logits, labels) / args.gradient_accumulation_steps
            else:
                # Use standard loss with label smoothing if supported
                if hasattr(model.config, 'label_smoothing_factor') and model.config.label_smoothing_factor > 0:
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                    logits = outputs.logits
                    # Manual label smoothing
                    loss_fct = nn.CrossEntropyLoss(ignore_index=-100, label_smoothing=model.config.label_smoothing_factor)
                    loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1)) / args.gradient_accumulation_steps
                else:
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                    loss = outputs.loss / args.gradient_accumulation_steps
            
            loss.backward()
            running_loss += loss.item() * args.gradient_accumulation_steps  # Scale back for logging

            # Update weights every gradient_accumulation_steps or at the end of epoch
            if (step + 1) % args.gradient_accumulation_steps == 0 or (step + 1) == len(train_dl):
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

        avg_loss = running_loss / max(1, len(train_dl))
        print(f"Epoch {epoch+1} average loss: {avg_loss:.4f}")
        
        # Evaluate on dev set
        if args.dev:
            macro_f1 = evaluate_model(model, tokenizer, args.dev, args.device, args.max_length)
            print(f"Epoch {epoch+1} dev Macro-F1: {macro_f1:.4f}")
            
            if macro_f1 > best_macro_f1:
                best_macro_f1 = macro_f1
                patience_counter = 0
                no_improve_epochs = 0
                model.save_pretrained(args.out_dir)
                tokenizer.save_pretrained(args.out_dir)
                print(f"✓ New best model saved! Macro-F1: {macro_f1:.4f}")
            else:
                patience_counter += 1
                no_improve_epochs += 1
                # Only stop if no improvement for patience epochs AND we're past epoch 10
                if patience_counter >= patience and epoch >= 10:
                    print(f"Early stopping triggered after {epoch+1} epochs (no improvement for {patience} epochs)")
                    break
                # Also try learning rate reduction if stuck
                if no_improve_epochs >= 5 and no_improve_epochs % 3 == 0:
                    for param_group in optimizer.param_groups:
                        old_lr = param_group['lr']
                        param_group['lr'] = old_lr * 0.5
                        print(f"Reducing learning rate from {old_lr:.2e} to {param_group['lr']:.2e}")
    
    # Save final model if no dev set evaluation or if best wasn't saved
    if not args.dev or best_macro_f1 == 0.0:
        model.save_pretrained(args.out_dir)
        tokenizer.save_pretrained(args.out_dir)
    
    print(f"Training completed. Best Macro-F1: {best_macro_f1:.4f}")
    print(f"Saved model + tokenizer to {args.out_dir}")


if __name__ == "__main__":
    main()
