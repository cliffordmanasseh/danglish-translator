#!/usr/bin/env python3
"""LoRA / full SFT for the Danish -> English student.

This is a small, dependency-light training loop: plain JSONL input, a custom
torch Dataset, and a manual optimizer loop.  It does not need TRL or Accelerate.
It works on CPU, CUDA, and ROCm (PyTorch exposes ROCm as ``cuda``).

The prompt format must match ``generate_synthetic.py`` and
``mt_benchmark.py``: ChatML, with the target after ``<|im_start|>assistant``.

Example
-------
    python experiments/train_student.py \
        --model utter-project/EuroLLM-1.7B-Instruct \
        --data /tmp/pilot/pilot_train.jsonl \
        --output /tmp/pilot/eurollm-1.7b-da-en-lora \
        --epochs 1 \
        --batch-size 4 \
        --grad-accum 8 \
        --lr 1e-4 \
        --bf16 \
        --merge
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

PROMPT = (
    "<|im_start|>system\n<|im_end|>\n"
    "<|im_start|>user\n"
    "Translate the following Danish source text to English:\n"
    "Danish: {source}\n"
    "English: <|im_end|>\n"
    "<|im_start|>assistant\n"
)

TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if obj.get("source") and obj.get("target"):
                rows.append(obj)
    return rows


class TranslationDataset(Dataset):
    def __init__(
        self,
        rows: list[dict[str, Any]],
        tokenizer: Any,
        max_length: int,
    ) -> None:
        self.examples: list[tuple[list[int], list[int]]] = []
        skipped = 0
        eos_id = tokenizer.eos_token_id
        for row in rows:
            prompt_ids = tokenizer(
                PROMPT.format(source=row["source"]),
                add_special_tokens=False,
            )["input_ids"]
            target_ids = tokenizer(
                row["target"],
                add_special_tokens=False,
            )["input_ids"]
            target_ids = [*target_ids, eos_id]
            input_ids = [*prompt_ids, *target_ids]
            if len(input_ids) > max_length:
                skipped += 1
                continue
            labels = [-100] * len(prompt_ids) + target_ids
            self.examples.append((input_ids, labels))
        if skipped:
            print(f"skipped {skipped} examples longer than {max_length} tokens", file=sys.stderr)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> tuple[list[int], list[int]]:
        return self.examples[index]


def make_collate(pad_token_id: int):
    def collate(batch: list[tuple[list[int], list[int]]]) -> dict[str, torch.Tensor]:
        max_len = max(len(input_ids) for input_ids, _ in batch)
        input_ids = torch.full((len(batch), max_len), pad_token_id, dtype=torch.long)
        labels = torch.full((len(batch), max_len), -100, dtype=torch.long)
        attention_mask = torch.zeros((len(batch), max_len), dtype=torch.long)
        for index, (ids, target_labels) in enumerate(batch):
            length = len(ids)
            input_ids[index, :length] = torch.tensor(ids, dtype=torch.long)
            labels[index, :length] = torch.tensor(target_labels, dtype=torch.long)
            attention_mask[index, :length] = 1
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

    return collate


def choose_dtype(name: str, device: str) -> torch.dtype:
    if name == "float32":
        return torch.float32
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    # auto
    if device == "cuda" and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float32


def build_scheduler(optimizer: torch.optim.Optimizer, total_steps: int, warmup_steps: int):
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="utter-project/EuroLLM-1.7B-Instruct")
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--max-length", type=int, default=768)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--dtype", default="auto", choices=["auto", "bfloat16", "float16", "float32"]
    )
    parser.add_argument(
        "--attn-implementation",
        default="sdpa",
        choices=["eager", "sdpa", "flash_attention_2"],
        help="transformers attention backend; use eager on gfx1151 (SDPA fails)",
    )
    parser.add_argument("--bf16", action="store_true", help="use bf16 autocast on CUDA/ROCm")
    parser.add_argument("--fp16", action="store_true", help="use fp16 autocast + GradScaler")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--full", action="store_true", help="full fine-tune instead of LoRA")
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--merge", action="store_true", help="merge LoRA and save a standalone model"
    )
    parser.add_argument("--max-samples", type=int, default=0)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    rows = read_jsonl(args.data)
    if args.max_samples:
        rows = rows[: args.max_samples]
    if not rows:
        raise SystemExit(f"no usable rows in {args.data}")
    random.Random(args.seed).shuffle(rows)

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA/ROCm is not available; falling back to CPU", file=sys.stderr)
        device = "cpu"

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    dtype = choose_dtype(args.dtype, device)
    print(f"loading {args.model} on {device} as {dtype}", file=sys.stderr)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        attn_implementation=args.attn_implementation,
    )
    model.to(device)
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    if not args.full:
        from peft import LoraConfig, TaskType, get_peft_model

        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=TARGET_MODULES,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        model = get_peft_model(model, lora_config)
        if args.gradient_checkpointing:
            # LoRA freezes the base model; checkpointing needs input grads enabled.
            model.base_model.model.enable_input_require_grads()
        model.print_trainable_parameters()
    else:
        trainable_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_count = sum(p.numel() for p in model.parameters())
        print(
            f"trainable params: {trainable_count:,} / {total_count:,} "
            f"({100 * trainable_count / total_count:.2f}%)",
            file=sys.stderr,
        )

    dataset = TranslationDataset(rows, tokenizer, args.max_length)
    if not dataset:
        raise SystemExit("all examples were longer than --max-length")
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=make_collate(tokenizer.pad_token_id),
        num_workers=0,
        drop_last=False,
    )

    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable,
        lr=args.lr,
        betas=(0.9, 0.95),
        weight_decay=args.weight_decay,
    )
    steps_per_epoch = math.ceil(len(loader) / args.grad_accum)
    total_steps = max(1, args.epochs * steps_per_epoch)
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = build_scheduler(optimizer, total_steps, warmup_steps)

    use_amp = (args.bf16 or args.fp16) and device == "cuda"
    amp_dtype = torch.bfloat16 if args.bf16 else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=args.fp16 and device == "cuda")

    args.output.mkdir(parents=True, exist_ok=True)
    print(
        f"{len(dataset)} examples, {len(loader)} batches/epoch, "
        f"{total_steps} optimizer steps, effective batch "
        f"{args.batch_size * args.grad_accum}",
        file=sys.stderr,
    )

    model.train()
    global_step = 0
    tokens_since_log = 0
    loss_since_log = 0.0
    started = time.perf_counter()

    for epoch in range(args.epochs):
        for batch_index, batch in enumerate(loader):
            batch = {key: value.to(device) for key, value in batch.items()}
            autocast = (
                torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp)
                if use_amp
                else nullcontext()
            )
            with autocast:
                outputs = model(**batch)
                loss = outputs.loss / args.grad_accum

            scaler.scale(loss).backward()
            tokens_since_log += int(batch["attention_mask"].sum().item())
            loss_since_log += float(loss.detach().item()) * args.grad_accum

            if (batch_index + 1) % args.grad_accum == 0 or (batch_index + 1) == len(loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable, args.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                global_step += 1

                if global_step % args.logging_steps == 0:
                    elapsed = time.perf_counter() - started
                    lr = scheduler.get_last_lr()[0]
                    print(
                        f"epoch {epoch + 1}/{args.epochs} step {global_step}/{total_steps} "
                        f"loss {loss_since_log / args.logging_steps:.4f} "
                        f"lr {lr:.2e} tokens/s {tokens_since_log / elapsed:.0f}",
                        file=sys.stderr,
                    )
                    tokens_since_log = 0
                    loss_since_log = 0.0
                    started = time.perf_counter()

                if args.save_steps and global_step % args.save_steps == 0:
                    checkpoint = args.output / f"checkpoint-{global_step}"
                    model.save_pretrained(checkpoint)
                    tokenizer.save_pretrained(checkpoint)
                    print(f"saved {checkpoint}", file=sys.stderr)

    final_dir = args.output / "final"
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"saved adapter/final model to {final_dir}", file=sys.stderr)

    if args.merge and not args.full:
        merged_dir = args.output / "merged"
        merged = model.merge_and_unload()
        merged.save_pretrained(merged_dir)
        tokenizer.save_pretrained(merged_dir)
        print(f"saved merged model to {merged_dir}", file=sys.stderr)


if __name__ == "__main__":
    main()
