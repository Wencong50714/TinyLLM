## Intro

A lightweight implementation of a Decoder-only language model trained on the TinyStories dataset. The project features custom Triton kernels for optimized performance on NVIDIA GPUs.

## Features
- Transformer-based language model architecture
- Custom Triton kernels for key operations:
  - Softmax
  - RMS Normalization
  - Cross Entropy Loss
  - Rotary Position Embeddings (RoPE)
- Custom tokenizer training using SentencePiece

## Prerequisites

```bash
pip install -r requirements.txt
```

## Usage

```bash
# Download TinyStories dataset and train tokenizer
python train_vocab.py    
  
# Preprocess data
python preprocess.py

# Train Model
python train.py

# Generate text samples using trained model
python sample.py --prompt "your prompt"
```

## Reference

- https://github.com/datawhalechina/tiny-universe/
- https://github.com/unslothai/unsloth
- https://github.com/linkedin/Liger-Kernel