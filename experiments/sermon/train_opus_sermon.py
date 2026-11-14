#!/usr/bin/env python3
"""Fine-tune OPUS-MT tc-big (Marian) on the Danish→English sermon pairs.

This is the small-model alternative to the EuroLLM student: it keeps the
~230M-parameter Marian architecture and the CTranslate2 int8 deployment path.

Example:
    python experiments/sermon/train_opus_sermon.py \
        --data data/sermons/sermon_qwen_agy_full_mix_filtered.jsonl \
        --output models/opus-mt-tc-big-gmq-en-sermon-hf \
        --epochs 1 --batch-size 32 --lr 2e-5
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import (
    MarianMTModel,
    MarianTokenizer,
    get_linear_schedule_with_warmup,
)


class TranslationDataset(Dataset):
    def __init__(
        self,
        rows: list[dict[str, str]],
        tokenizer: MarianTokenizer,
        max_length: int,
    ) -> None:
        self.examples: list[dict[str, torch.Tensor]] = []
        skipped = 0
        for row in rows:
            source = " ".join(str(row.get("source", "")).split())
            target = " ".join(str(row.get("target", "")).split())
            if not source or not target:
                skipped += 1
                continue
            source_ids = tokenizer(
                source,
                max_length=max_length,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )
            target_ids = tokenizer(
                text_target=target,
                max_length=max_length,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )
            labels = target_ids["input_ids"].squeeze(0)
            labels[labels == tokenizer.pad_token_id] = -100
            self.examples.append(
                {
                    "input_ids": source_ids["input_ids"].squeeze(0),
                    "attention_mask": source_ids["attention_mask"].squeeze(0),
                    "labels": labels,
                }
            )
        if skipped:
            print(f"skipped {skipped} empty rows", file=sys.stderr)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return self.examples[index]


def read_jsonl(path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def evaluate(model: MarianMTModel, loader: DataLoader, device: str) -> float:
    model.eval()
    total_loss = 0.0
    batches = 0
    with torch.no_grad():
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            loss = model(**batch).loss
            total_loss += float(loss)
            batches += 1
    model.train()
    return total_loss / max(1, batches)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Helsinki-NLP/opus-mt-tc-big-gmq-en")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--val-samples", type=int, default=2000)
    parser.add_argument("--eval-steps", type=int, default=500)
    parser.add_argument("--logging-steps", type=int, default=100)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="float32")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(min(32, torch.get_num_threads()))

    rows = read_jsonl(args.data)
    if args.max_samples:
        rows = rows[: args.max_samples]
    random.Random(args.seed).shuffle(rows)
    if len(rows) < 100:
        raise SystemExit(f"not enough rows: {len(rows)}")

    val_rows = rows[: args.val_samples] if args.val_samples else []
    train_rows = rows[args.val_samples :] if args.val_samples else rows
    print(f"{len(train_rows)} train rows, {len(val_rows)} validation rows", file=sys.stderr)

    tokenizer = MarianTokenizer.from_pretrained(args.model)
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32
    model = MarianMTModel.from_pretrained(
        args.model,
        torch_dtype=dtype,
        attn_implementation="eager",
    ).to(args.device)
    model.train()
    print(f"loaded {args.model} on {args.device} as {dtype}", file=sys.stderr)

    train_dataset = TranslationDataset(train_rows, tokenizer, args.max_length)
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0
    )
    val_loader = (
        DataLoader(
            TranslationDataset(val_rows, tokenizer, args.max_length),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
        )
        if val_rows
        else None
    )

    updates_per_epoch = math.ceil(len(train_loader) / args.grad_accum)
    total_steps = max(1, args.epochs * updates_per_epoch)
    warmup_steps = int(total_steps * args.warmup_ratio)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )
    scaler = torch.amp.GradScaler(args.device, enabled=False)

    print(
        f"{len(train_dataset)} examples, {len(train_loader)} batches/epoch, "
        f"{total_steps} optimizer steps, effective batch "
        f"{args.batch_size * args.grad_accum}",
        file=sys.stderr,
    )

    global_step = 0
    loss_since_log = 0.0
    steps_since_log = 0
    started = time.perf_counter()
    optimizer.zero_grad(set_to_none=True)

    for epoch in range(args.epochs):
        for batch_index, batch in enumerate(train_loader):
            batch = {key: value.to(args.device) for key, value in batch.items()}
            with torch.autocast(
                device_type=args.device,
                dtype=dtype,
                enabled=args.dtype == "bfloat16",
            ):
                loss = model(**batch).loss / args.grad_accum
            scaler.scale(loss).backward()
            loss_since_log += float(loss.detach()) * args.grad_accum
            steps_since_log += 1

            if (batch_index + 1) % args.grad_accum == 0 or (batch_index + 1) == len(
                train_loader
            ):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                global_step += 1

                if global_step % args.logging_steps == 0:
                    elapsed = time.perf_counter() - started
                    print(
                        f"epoch {epoch + 1}/{args.epochs} step {global_step}/{total_steps} "
                        f"loss {loss_since_log / max(1, steps_since_log):.4f} "
                        f"lr {scheduler.get_last_lr()[0]:.2e} "
                        f"steps/s {steps_since_log / elapsed:.2f}",
                        file=sys.stderr,
                    )
                    loss_since_log = 0.0
                    steps_since_log = 0
                    started = time.perf_counter()

                if (
                    val_loader is not None
                    and args.eval_steps
                    and global_step % args.eval_steps == 0
                ):
                    val_loss = evaluate(model, val_loader, args.device)
                    print(f"  val loss {val_loss:.4f} at step {global_step}", file=sys.stderr)

    if val_loader is not None:
        print(f"final val loss {evaluate(model, val_loader, args.device):.4f}", file=sys.stderr)

    args.output.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)
    print(f"saved fine-tuned model to {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
