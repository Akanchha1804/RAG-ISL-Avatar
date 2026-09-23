"""
Flan-T5-small Fine-Tuning for ISL Gloss Generation.

Works in Google Colab (GPU) or locally (CPU).

Colab:
  1. Upload this file + backend/augmented_glosses.csv (auto-detected if repo mounted)
  2. Runtime -> Change runtime type -> GPU
  3. Run:  python finetune_t5.py --epochs 20 --batch-size 16
  4. Download isl_gloss_t5_merged.zip -> extract into backend/models/isl_gloss_t5/

Local:
  python notebooks/finetune_t5.py --epochs 8 --batch-size 4
  Output is written directly to backend/models/isl_gloss_t5/
"""

from __future__ import annotations

import argparse
import shutil
import sys
import zipfile
from pathlib import Path

PREFIX = "translate English to ISL: "
MODEL_NAME = "google/flan-t5-small"


def find_csv() -> Path:
    candidates = [
        Path("augmented_glosses.csv"),
        Path("backend/augmented_glosses.csv"),
        Path(__file__).resolve().parent.parent / "backend" / "augmented_glosses.csv",
        Path("/content/augmented_glosses.csv"),
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        "augmented_glosses.csv not found. Upload it next to this script "
        "or place it at backend/augmented_glosses.csv."
    )


def default_out_dir() -> Path:
    local = Path(__file__).resolve().parent.parent / "backend" / "models" / "isl_gloss_t5"
    if local.parent.exists() or local.parent.parent.exists():
        return local
    return Path("./isl_gloss_t5_merged")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fine-tune Flan-T5-small for ISL glosses")
    p.add_argument("--csv", type=Path, default=None, help="Path to augmented_glosses.csv")
    p.add_argument("--output", type=Path, default=None, help="Save dir (backend/models/isl_gloss_t5)")
    p.add_argument("--epochs", type=float, default=20.0)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-input", type=int, default=128)
    p.add_argument("--max-target", type=int, default=64)
    p.add_argument("--lora-r", type=int, default=8)
    p.add_argument("--no-lora", action="store_true", help="Full fine-tune instead of LoRA")
    p.add_argument("--zip-only", action="store_true", help="Only write zip next to output")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    import pandas as pd
    import torch
    from datasets import Dataset
    from transformers import (
        AutoModelForSeq2SeqLM,
        AutoTokenizer,
        DataCollatorForSeq2Seq,
        Seq2SeqTrainer,
        Seq2SeqTrainingArguments,
    )

    print(f"torch={torch.__version__} cuda={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"gpu={torch.cuda.get_device_name(0)}")

    csv_path = args.csv or find_csv()
    out_dir = (args.output or default_out_dir()).resolve()
    print(f"csv={csv_path}")
    print(f"output={out_dir}")

    df = pd.read_csv(csv_path)
    if not {"input", "output"}.issubset(df.columns):
        raise ValueError(f"CSV must have input,output columns; got {list(df.columns)}")
    df = df.dropna(subset=["input", "output"]).astype(str)
    print(f"rows={len(df)}")

    def format_example(row):
        return {
            "input_text": f"{PREFIX}{row['input'].strip()}",
            "target_text": " ".join(row["output"].split()).strip(),
        }

    dataset = Dataset.from_pandas(df)
    dataset = dataset.map(format_example, remove_columns=dataset.column_names)
    split = dataset.train_test_split(test_size=0.1, seed=args.seed)
    train_ds, val_ds = split["train"], split["test"]
    print(f"train={len(train_ds)} val={len(val_ds)}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    print(f"params={model.num_parameters():,}")

    if not args.no_lora:
        from peft import LoraConfig, TaskType, get_peft_model

        model = get_peft_model(
            model,
            LoraConfig(
                task_type=TaskType.SEQ_2_SEQ_LM,
                inference_mode=False,
                r=args.lora_r,
                lora_alpha=32,
                lora_dropout=0.1,
                target_modules=["q", "v"],
                bias="none",
            ),
        )
        model.print_trainable_parameters()

    def tokenize(batch):
        inputs = tokenizer(
            batch["input_text"],
            max_length=args.max_input,
            truncation=True,
            padding="max_length",
        )
        targets = tokenizer(
            batch["target_text"],
            max_length=args.max_target,
            truncation=True,
            padding="max_length",
        )
        labels = targets["input_ids"]
        labels = [[(t if t != tokenizer.pad_token_id else -100) for t in seq] for seq in labels]
        inputs["labels"] = labels
        return inputs

    train_tok = train_ds.map(tokenize, batched=True, remove_columns=train_ds.column_names)
    val_tok = val_ds.map(tokenize, batched=True, remove_columns=val_ds.column_names)

    use_cuda = torch.cuda.is_available()
    training_args = Seq2SeqTrainingArguments(
        output_dir=str(out_dir.parent / "isl_gloss_t5_checkpoints"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=0.01,
        warmup_steps=20,
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        predict_with_generate=False,
        fp16=use_cuda,
        bf16=False,
        report_to="none",
        save_total_limit=2,
        seed=args.seed,
        dataloader_num_workers=0,
    )

    collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        label_pad_token_id=-100,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_tok,
        eval_dataset=val_tok,
        data_collator=collator,
        processing_class=tokenizer,
    )

    result = trainer.train()
    print(f"train_loss={result.training_loss:.4f}")
    eval_result = trainer.evaluate()
    print(f"eval_loss={eval_result.get('eval_loss'):.4f}")

    def gen(sentence: str) -> str:
        inputs = tokenizer(
            f"{PREFIX}{sentence}",
            return_tensors="pt",
            max_length=args.max_input,
            truncation=True,
        )
        if use_cuda:
            inputs = {k: v.cuda() for k, v in inputs.items()}
        with torch.no_grad():
            out = model.generate(**inputs, max_length=args.max_target, num_beams=4, early_stopping=True)
        return tokenizer.decode(out[0], skip_special_tokens=True).strip()

    print("--- sample generations ---")
    for s in [
        "hello how are you",
        "please help me",
        "what is your name",
        "I am fine thank you",
        "can you repeat that",
    ]:
        print(f"EN:  {s}")
        print(f"ISL: {gen(s)}")

    if hasattr(model, "merge_and_unload"):
        model = model.merge_and_unload()

    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))
    (out_dir / "TRAINING_INFO.txt").write_text(
        f"base={MODEL_NAME}\nepochs={args.epochs}\nbatch={args.batch_size}\n"
        f"lr={args.lr}\nlora_r={args.lora_r}\nrows={len(df)}\n"
        f"train_loss={result.training_loss}\neval_loss={eval_result.get('eval_loss')}\n",
        encoding="utf-8",
    )
    print(f"saved -> {out_dir}")

    zip_path = out_dir.parent / f"{out_dir.name}.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in out_dir.rglob("*"):
            if path.is_file():
                zf.write(path, path.relative_to(out_dir.parent))
    print(f"zip -> {zip_path}")

    ckpt_root = out_dir.parent / "isl_gloss_t5_checkpoints"
    if ckpt_root.exists():
        shutil.rmtree(ckpt_root, ignore_errors=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
