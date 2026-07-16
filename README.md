# CEM: Contrastive Embedding Multiplexing for Multi-User Semantic Communication

Reference implementation and experiment scripts for the letter

> K.-H. Lee, H.-H. Choi, and J.-R. Lee, "Contrastive Embedding Multiplexing for
> Multi-User Semantic Communication Systems," submitted to *IEEE Signal
> Processing Letters* (manuscript SPL-47318-2026).

CEM multiplexes several users in a single shared embedding space: a
user-specific positional mask assigns each user a soft subspace, and an
InfoNCE contrastive objective (jointly trained with the reconstruction loss)
pushes the channel-corrupted receiver-side representations of different users
toward near-orthogonality.

## Repository layout

| File | Purpose |
|---|---|
| `cem_full.py` | Main model + training/evaluation for the symbol-level experiments (Fig. 2, Fig. 3, Table II of the letter). One run per configuration; per-batch SNR drawn uniformly from 0–25 dB. |
| `cem_text.py` | **Semantic task validation (Sec. III-E)**: text transmission over the Europarl corpus with a 22,000-word vocabulary, scored with sentence-level BLEU-4. |
| `cem_ablation.py` | Stand-alone ablation grid (mask on/off x contrastive on/off, hard binary mask). |
| `postproc.py` | Large-sample SER re-evaluation of statistically thin high-SNR points and inter-user cosine-similarity (rho) measurements (Sec. III-D). |
| `plot_fig2.py`, `plot_fig3.py` | Regenerate Fig. 2 and Fig. 3 from the result CSVs. |
| `results/` | CSV outputs used in the letter (SER grids, BLEU scores, rho measurements, ablation table). |

## Semantic task validation (Sec. III-E)

`cem_text.py` transmits English sentences from the Europarl corpus with
`U = 8` users through the identical CEM pipeline used for the symbol-level
experiments (mask -> shared Transformer encoder -> 1/U superposition ->
Rayleigh fading + AWGN -> masked-query cross-attention decoding) and reports
corpus-averaged sentence BLEU-4 (add-one smoothing on the higher n-gram
precisions) on a held-out 5% test split.

1. Download the English side of the French–English Europarl v7 corpus from
   <https://www.statmt.org/europarl/> and place it at
   `data/europarl-v7.fr-en.en`.
2. Run:

   ```bash
   python cem_text.py            # trains CE and CE+NCE (lambda = 0.01), then evaluates BLEU at 10/20 dB
   ```

Expected results (single seed, ~8,000 steps):

| Scheme | BLEU @ 10 dB | BLEU @ 20 dB |
|---|---|---|
| CE (mask only) | 0.117 | 0.118 |
| CE + NCE (proposed) | **0.156** | **0.163** |

## Symbol-level experiments

```bash
python cem_full.py                    # trains all configurations, writes results_full.csv
python postproc.py                    # high-precision re-evaluation + rho report
python plot_fig2.py && python plot_fig3.py
```

Key defaults (Table I of the letter): `d = 128`, `T = 32`, `V = 4`,
4 encoder / 2 decoder layers, projection dimension 64, temperature 0.1,
AdamW with learning rate 1e-3, contrastive weight `lambda = 1e-2`.

## Requirements

- Python >= 3.10, PyTorch >= 2.0 (CUDA or Apple MPS optional; CPU works)
- `matplotlib`, `pandas` for the plotting scripts

## License

MIT (see `LICENSE`).
