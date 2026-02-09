import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# The base model we started with
BASE_MODEL = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"
# The folder created by your training script
ADAPTER_PATH = "./linux_expert_model"

print("--- Loading Linux Expert AI (AMD A4 Optimized) ---")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
model = AutoModelForCausalLM.from_pretrained(BASE_MODEL)

# This layers your trained "brain" over the base model
model = PeftModel.from_pretrained(model, ADAPTER_PATH)
model.to("cpu") # Ensure we stay on CPU to avoid RAM spikes

def ask_ai(question):
    # Format the prompt to guide the AI
    prompt = f"System: You are a Linux expert specializing in Arch, Alpine, and LFS.\nUser: {question}\nAI:"
    inputs = tokenizer(prompt, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=100, 
            temperature=0.7, 
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Print only the AI's answer
    print("\n" + response.split("AI:")[-1].strip() + "\n")

if __name__ == "__main__":
    print("AI Loaded! Type 'exit' to quit.")
    while True:
        query = input("Ask a Linux question: ")
        if query.lower() in ['exit', 'quit']:
            break
        ask_ai(query)
