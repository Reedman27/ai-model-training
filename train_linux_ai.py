import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    TrainingArguments, 
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model

# We use TinyLlama (1.1B params) - it's the limit for 8GB RAM + CPU
MODEL_NAME = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"

def train():
    print("Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token

    # Load the model in FP32 (CPU doesn't like 16-bit as much as GPUs do)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

    # LORA CONFIG: This is what makes it possible on your A4 chip
    config = LoraConfig(
        r=8, 
        lora_alpha=32, 
        target_modules=["q_proj", "v_proj"], 
        lora_dropout=0.05, 
        bias="none", 
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, config)
    model.print_trainable_parameters()

    # DATA PREP
    dataset = load_dataset("text", data_files="data/linux_docs/*.txt")
    
    def tokenize(element):
        return tokenizer(element["text"], truncation=True, max_length=256) # Small context for speed

    tokenized_dataset = dataset.map(tokenize, batched=True, remove_columns=["text"])

    # TRAINING ARGS: Optimized for Low-End Hardware
    training_args = TrainingArguments(
        output_dir="./linux_ai_results",
        per_device_train_batch_size=1,      # Crucial: Keeps RAM usage low
        gradient_accumulation_steps=8,     # Simulates a larger batch size
        num_train_epochs=1,                # Start with one pass
        learning_rate=2e-4,
        logging_steps=10,
        save_total_limit=1,
        use_cpu=True,                      # Force CPU mode
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )

    print("Starting training (this will be slow, grab a coffee)...")
    trainer.train()
    model.save_pretrained("./linux_expert_model")

if __name__ == "__main__":
    train()
