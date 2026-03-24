# Data Lab

## Lab 1: LLM Data Pipeline

This lab is the LLM Data Pipeline lab in Data Lab (Lab 1). It walks through preparing a causal language modeling dataset end-to-end inside a notebook.

### What's here

- `Lab1.ipynb`: installs deps, downloads a trimmed IMDB reviews split, runs quick EDA, tokenizes with `roberta-base`, chunks into fixed-length blocks, and batches with a manual collator for causal LM.

### How to run

1. Open `Lab1.ipynb` in VS Code or Jupyter.
2. Run the first cell to install `transformers`, `datasets`, and `torch` into the active kernel.
3. Execute the remaining cells top-to-bottom to load IMDB, view EDA (length stats, label balance), tokenize, group into blocks, and inspect a sample batch.

### How it works

- Uses a small, shuffled slice of IMDB reviews for quick iterations.
- Tokenizes text with `roberta-base`, setting EOS as the pad token.
- Runs a brief EDA cell to show length stats and label balance before processing.
- Packs token IDs and attention masks into `BLOCK_SIZE`-sized sequences for causal LM training.
- Builds a PyTorch `DataLoader` with a manual collate to stack ids/masks and clone labels for next-token prediction.
