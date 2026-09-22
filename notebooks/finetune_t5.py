"""
Flan-T5-small Fine-Tuning for ISL Gloss Generation
Run this in Google Colab with GPU runtime.

Steps:
1. Upload augmented_glosses.csv to Colab
2. Run all cells
3. Download the fine-tuned model
4. Copy to backend/models/isl_gloss_t5/
"""

# Cell 1: Install dependencies
# !pip install -q transformers datasets peft accelerate torch sentencepiece

# Cell 2: Import libraries
import torch
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
)
from datasets import Dataset
import pandas as pd
import numpy as np

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# Cell 3: Load dataset
# Upload augmented_glosses.csv first
# from google.colab import files
# uploaded = files.upload()

df = pd.read_csv("augmented_glosses.csv")
print(f"Dataset size: {len(df)}")
print(f"\nSample entries:")
print(df.head(10))

# Cell 4: Format data for T5
def format_example(row):
    input_text = f"translate English to ISL: {row['input']}"
    target_text = row['output']
    return {"input_text": input_text, "target_text": target_text}

dataset = Dataset.from_pandas(df)
dataset = dataset.map(format_example, remove_columns=dataset.column_names)

# Split train/val
split = dataset.train_test_split(test_size=0.1, seed=42)
train_dataset = split["train"]
val_dataset = split["test"]

print(f"Train size: {len(train_dataset)}")
print(f"Val size: {len(val_dataset)}")

# Cell 5: Load model and tokenizer
MODEL_NAME = "google/flan-t5-small"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

print(f"Model parameters: {model.num_parameters():,}")

# Cell 6: Tokenize datasets
def tokenize_function(examples):
    inputs = tokenizer(
        examples["input_text"],
        max_length=128,
        truncation=True,
        padding="max_length",
    )
    targets = tokenizer(
        examples["target_text"],
        max_length=64,
        truncation=True,
        padding="max_length",
    )
    labels = targets["input_ids"]
    labels[labels == tokenizer.pad_token_id] = -100
    inputs["labels"] = labels
    return inputs

train_tokenized = train_dataset.map(tokenize_function, batched=True, remove_columns=train_dataset.column_names)
val_tokenized = val_dataset.map(tokenize_function, batched=True, remove_columns=val_dataset.column_names)

# Cell 7: LoRA configuration
from peft import LoraConfig, TaskType, get_peft_model

lora_config = LoraConfig(
    task_type=TaskType.SEQ_2_SEQ_LM,
    inference_mode=False,
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["q", "v"],
    bias="none",
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# Cell 8: Training arguments
training_args = Seq2SeqTrainingArguments(
    output_dir="./isl_gloss_t5_lora",
    num_train_epochs=20,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    learning_rate=3e-4,
    weight_decay=0.01,
    warmup_steps=50,
    logging_steps=10,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    predict_with_generate=True,
    fp16=torch.cuda.is_available(),
    report_to="none",
    save_total_limit=2,
)

# Cell 9: Data collator
data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=model,
    label_pad_token_id=-100,
)

# Cell 10: Trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_tokenized,
    eval_dataset=val_tokenized,
    data_collator=data_collator,
    tokenizer=tokenizer,
)

# Cell 11: Train
train_result = trainer.train()
print(f"Training loss: {train_result.training_loss:.4f}")

# Cell 12: Evaluate
eval_results = trainer.evaluate()
print(f"Eval loss: {eval_results['eval_loss']:.4f}")

# Cell 13: Test inference
def generate_gloss(sentence):
    input_text = f"translate English to ISL: {sentence}"
    inputs = tokenizer(input_text, return_tensors="pt", max_length=128, truncation=True)
    if torch.cuda.is_available():
        inputs = {k: v.cuda() for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=64, num_beams=4, early_stopping=True)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

test_sentences = [
    "hello how are you",
    "please help me",
    "what is your name",
    "I am fine thank you",
    "can you repeat that",
]

for sent in test_sentences:
    gloss = generate_gloss(sent)
    print(f"EN: {sent}")
    print(f"ISL: {gloss}")
    print()

# Cell 14: Merge LoRA weights and save
merged_model = model.merge_and_unload()
merged_model.save_pretrained("./isl_gloss_t5_merged")
tokenizer.save_pretrained("./isl_gloss_t5_merged")

print("Model saved to ./isl_gloss_t5_merged")
print("Download this folder and copy to backend/models/isl_gloss_t5/")

# Cell 15: Optional - download from Colab
# from google.colab import files
# !zip -r isl_gloss_t5_merged.zip isl_gloss_t5_merged/
# files.download("isl_gloss_t5_merged.zip")
