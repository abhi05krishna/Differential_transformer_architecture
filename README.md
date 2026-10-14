# Differential Transformer — Reduced-Scale Implementation and Evaluation

## Overview

This project implements and evaluates a small decoder-only language model inspired by the Differential Transformer.

The main objective is to compare conventional causal self-attention with differential attention, where two attention distributions are combined using a learnable coefficient λ.

Three model variants are evaluated:

1. Standard Transformer
2. Learned-λ Differential Transformer
3. Fixed-λ Differential Transformer

The project also includes a small Needle-in-a-Haystack retrieval experiment and an ablation study comparing learned and fixed λ.

---

## Architecture

The model uses the following components:

- GPT-2 tokenizer
- Token embeddings
- Pre-RMSNorm
- Causal attention
- Rotary Positional Embeddings (RoPE)
- Differential attention
- SwiGLU feed-forward network
- Residual connections
- Final RMSNorm
- Language-model head

The Standard Transformer uses 4 attention heads.

The Differential Transformer uses 2 differential heads, with each differential head containing two query/key attention branches.

---

## Dataset

Dataset: TinyStories

Tokenizer: GPT-2 tokenizer

Vocabulary size: 50,257

Training stories: 25,000

Validation stories: 2,500

Context length: 128 tokens

The validation set is a held-out slice of the TinyStories training split rather than the official validation split.

---

## Experimental Configuration

|      Parameter              | Value    |
|---------------------------- |----------|
| Embedding dimension         | 128      |
| Transformer layers          | 2        |
| Standard attention heads    | 4        |
| Differential attention heads| 2        |
| Context length              | 128      |
| Batch size                  | 16       |
| Learning rate               | 3 × 10⁻⁴ |
| Weight decay                | 0.01     |
| Dropout                     | 0.1      |
| Training steps              | 10,000   |
| Random seed                 | 42       |
| Optimizer                   | AdamW    |

CUDA mixed precision was used when available.

---

## Differential Attention

Differential attention computes two attention distributions:

A₁ = softmax(Q₁K₁ᵀ / √d)

A₂ = softmax(Q₂K₂ᵀ / √d)

and combines them as:

DiffAttn(X) = (A₁ − λA₂)V

The subtraction is intended to suppress common or distracting attention patterns.

---

## Results

### Language Modeling

| Model                  | Validation Loss | Validation PPL |
|------------------------|-----------------|----------------|
| Standard Transformer   | 3.2135          | 24.87          |
| Learned-λ Differential | 2.8688          | 17.62          |
| Fixed-λ Differential   | 2.8559          | 17.39          |

The Differential Transformer variants achieved substantially lower validation loss and perplexity than the Standard Transformer in this reduced-scale experiment.

The fixed-λ variant obtained the lowest validation perplexity in this particular run.

---

## Ablation Study

The learned-λ and fixed-λ Differential Transformer variants were compared to determine whether learning λ provided an advantage.

| Model     | Validation Loss | Validation PPL |
|-----------|-----------------|----------------|
| Learned λ | 2.8688          | 17.62          |
| Fixed λ   | 2.8559          | 17.39          |

The fixed-λ model performed slightly better on validation perplexity in this run.

Therefore, learning λ did not provide an improvement for this particular experimental configuration.

---

## Needle-in-a-Haystack Retrieval

A synthetic retrieval task containing 30 examples was used.

There were:

- 10 early-position examples
- 10 middle-position examples
- 10 late-position examples
- 8 candidate answers per example

### Retrieval Accuracy

| Position | Standard | Learned λ | Fixed λ |
|----------|----------|-----------|---------|
| Early    | 0/10     | 2/10      | 1/10    |
| Middle   | 2/10     | 3/10      | 3/10    |
| Late     | 1/10     | 1/10      | 1/10    |
| Overall  | 3/30     | 6/30      | 5/30    |

Overall accuracy:

- Standard Transformer: 10.00%
- Learned-λ Differential: 20.00%
- Fixed-λ Differential: 16.67%

The learned-λ model obtained the highest retrieval accuracy in this small synthetic experiment.

---

## Limitations

This project is a reduced-scale educational implementation and does not reproduce the scale of the original Differential Transformer experiments.

Important limitations include:

- Small 128-dimensional model
- Only 2 Transformer layers
- 25,000 training stories
- 2,500 validation stories
- Single random seed
- Small synthetic retrieval dataset
- Only 30 retrieval examples
- Eight candidates per retrieval example
- Validation uses a held-out slice of the training split
- Packed token blocks may cross story boundaries

Therefore, the results should be interpreted as an implementation and experimental study rather than a reproduction of the original large-scale paper.

---

## Reproducibility

The main experiment uses random seed 42.

The project was implemented using PyTorch and was trained using CUDA when available.

The repository contains the implementation, experiment scripts/notebook, result visualizations, and project report.

---

