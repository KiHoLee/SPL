# CEM Semantic Task Validation (Sec. III-E)

Code and evaluation script for the semantic task validation experiment
(Sec. III-E) of the letter

> K.-H. Lee, H.-H. Choi, and J.-R. Lee, "Contrastive Embedding Multiplexing for
> Multi-User Semantic Communication Systems," submitted to *IEEE Signal
> Processing Letters* (manuscript SPL-48226-2026).

Contrastive embedding multiplexing (CEM) multiplexes several users in one
shared embedding space: a user-specific positional mask assigns each user a
soft subspace, and an InfoNCE contrastive objective (trained jointly with the
reconstruction loss) drives the channel-corrupted receiver-side
representations of different users toward near-orthogonality. This repository
verifies that the symbol-level gains carry over to a practical semantic task,
namely text transmission scored by BLEU.

## What the experiment does

`cem_text.py` transmits English sentences from the Europarl corpus with
`U = 4` users (letter configuration) through the CEM pipeline (user-specific masking -> shared
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
  (`lambda = 0.001`, the operating value adopted in the letter)

## How to run

1. Download the English side of the French-English Europarl v7 corpus from
   <https://www.statmt.org/europarl/> and place it at
   `data/europarl-v7.fr-en.en`.
2. Run:

   ```bash
   python cem_text.py --users 4 --lam 0        # CE scheme (letter configuration)
   python cem_text.py --users 4 --lam 0.001    # proposed scheme (letter configuration)
   python cem_text.py                          # previous-configuration U = 8 pair (lambda = 0.01)
   ```

   Results are written to `results_bleu.csv`.

## Expected results (single seed)

| Scheme | BLEU @ 10 dB | BLEU @ 20 dB |
|---|---|---|
| Single-user reference (U = 1, CE) | 0.994 | 0.995 |
| CE (mask only, U = 4) | 0.183 | 0.184 |
| CE + NCE (proposed, U = 4, lambda = 1e-3, letter configuration) | **0.369** | **0.394** |
| CE (mask only, U = 8, d = 128, previous configuration) | 0.117 | 0.118 |
| CE + NCE (U = 8, d = 256, lambda = 1e-3) | 0.456 | 0.491 |

The CE scheme's BLEU is flat in SNR, indicating an interference-limited
regime; the contrastive term alleviates it, so embedding-level separation
translates into semantic-level recovery (at the letter configuration U = 4 the
proposed scheme roughly doubles the BLEU).

`results/results_bleu.csv` contains the numbers reported in the letter.

The single-user interference-free reference can be reproduced with
`python cem_text.py --users 1`. Its near-perfect BLEU shows that the
lower scores at U = 4 and U = 8 come from inter-user interference rather
than from the text model itself.

The absolute BLEU is governed by the embedding capacity relative to the
user load. Doubling the embedding dimension under the adopted weight
(`python cem_text.py --dim 256 --lam 0.001`) raises the U = 8 BLEU from
0.117 to 0.456/0.491, confirming the capacity trend reported in the
response letter.

A no-mask ablation (user-specific masking disabled, contrastive term only)
can be reproduced with `python cem_text.py --include-no-mask`. At the symbol
level this configuration fails entirely (SER pinned near 0.56 at all SNRs;
Sec. III-D of the letter), showing that the mask and the contrastive term
are complementary.

## Requirements

- Python >= 3.10, PyTorch >= 2.0 (CUDA or Apple MPS optional; CPU works)

## License

MIT (see `LICENSE`).
