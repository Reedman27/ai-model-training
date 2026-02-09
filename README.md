# Linux Wiki AI Trainer (Low-Spec Optimized)

A GPLv3 toolkit for harvesting minimalist Linux documentation and training a TinyLlama-1.1B model. Optimized for low-end hardware (AMD A4-9125 / 8GB RAM).

## 🚀 Setup, Harvesting, Cleaning, and Training (All-in-One)

You need Python 3.10+, pip, and venv installed.  
Install required packages using your system package manager (apt, pacman, dnf, emerge):

- python3
- python3-pip
- python3-venv

## 🔁 Full Workflow

```bash
# create and activate virtual environment
python3 -m venv ia_ai_env
source ia_ai_env/bin/activate

# install dependencies
pip install requests beautifulsoup4 datasets "transformers[torch]" accelerate peft

# harvest documentation (Arch, Alpine, LFS, Crux, Slackware)
python3 harvest_linux.py

# clean harvested data
sed -i 's/[[:space:]]\+/ /g; /\[edit\]/d; /^$/d' data/linux_docs/*.txt

# train the model (CPU-only, LoRA, low-RAM)
python3 train_linux_ai.py

Chat with your AI
After training is complete, use the chat script to test your model:
```bash
