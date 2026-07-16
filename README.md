# CEM Semantic Task Validation (Sec. III-E)

Code and evaluation script for the semantic task validation experiment
(Sec. III-E) of the letter

> K.-H. Lee, H.-H. Choi, and J.-R. Lee, "Contrastive Embedding Multiplexing for
> Multi-User Semantic Communication Systems," submitted to *IEEE Signal
> Processing Letters* (manuscript SPL-47318-2026).

Contrastive embedding multiplexing (CEM) multiplexes several users in one
shared embedding space: a user-specific positional mask assigns each user a
soft subspace, and an InfoNCE contrastive objective (trained jointly with the
reconstruction loss) drives the channel-corrupted receiver-side
representations of different users toward near-orthogonality. This repository
verifies that the symbol-level gains carry over to a practical semantic task,
namely text transmission scored by BLEU.

## What the experiment does

`cem_text.py` transmits English sentences from the Europarl corpus with
`U = 8` users through the CEM pipeline (user-specific masking -> shared
Transformer encoder -> 1/U superposition -> Rayleigh fading + AWGN ->
masked-query cross-attention decoding) and reports corpus-averaged sentence
BLEU-4 (add-one smoothing on the higher n-gram precisions) on a held-out
5% test split.

- Vocabulary: the 22,000 most frequent lowercase words (+ PAD/UNK)
- Token length `T = 32`, embedding dimension `d = 128`, projection
  dimension 64, temperature 0.1
- Two configurations are trained under an identical protocol
  (8,000 steps, AdamW, per-batch SNR drawn uniformly from 0-25 dB):
  the CE scheme (mask only, `lambda = 0`) and the proposed CE + NCE scheme
  (`lambda = 0.01`)

## How to run

1. Download the English side of the French-English Europarl v7 corpus from
   <https://www.statmt.org/europarl/> and place it at
   `data/europarl-v7.fr-en.en`.
2. Run:

   ```bash
   python cem_text.py        # trains both configurations, then evaluates BLEU at 10 and 20 dB
   ```

   Results are written to `results_bleu.csv`.

## Expected results (single seed)

| Scheme | BLEU @ 10 dB | BLEU @ 20 dB |
|---|---|---|
| CE (mask only) | 0.117 | 0.118 |
| CE + NCE (proposed) | **0.156** | **0.163** |

The CE scheme's BLEU is flat in SNR, indicating an interference-limited
regime; the contrastive term alleviates it, so embedding-level separation
translates into semantic-level recovery (a 33-38% relative BLEU gain).

`results/results_bleu.csv` contains the numbers reported in the letter.

## Requirements

- Python >= 3.10, PyTorch >= 2.0 (CUDA or Apple MPS optional; CPU works)

## License

MIT (see `LICENSE`).
