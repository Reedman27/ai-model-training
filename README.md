# Linux Wiki AI Trainer (Low-Spec Optimized)

This repository contains tools to harvest documentation from minimalist Linux distributions and fine-tune a language model (TinyLlama-1.1B) on entry-level hardware.

## Target Distributions
The AI is trained on the official handbooks and wikis of:
- **Arch Linux**
- **Alpine Linux**
- **Linux From Scratch (LFS)**
- **CRUX**
- **Slackware**

## Hardware Optimization
This project is specifically designed to run on low-end systems, such as the **AMD A4-9125** with **8GB RAM**.
- **CPU Training:** Uses Hugging Face `Trainer` with `use_cpu=True`.
- **LoRA (Low-Rank Adaptation):** Only trains ~1.1M parameters instead of the full 1.1B, saving massive amounts of memory.
- **Memory Management:** Small context windows (256) and gradient accumulation (8 steps) prevent OOM (Out of Memory) crashes on 8GB systems.

## Usage

### 1. Harvest Data
Identifies and scrapes clean text from official Linux documentation.
```bash
python3 harvest_linux.py
