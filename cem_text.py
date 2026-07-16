"""
Semantic task validation (Sec. III-E): transmit English sentences from the
Europarl corpus with U=8 users through the CEM system and score BLEU.

Same architecture/protocol as cem_full.py (pre-norm Transformer, positional
encoding, random-SNR training, AdamW + cosine), with a word-level vocabulary
built from Europarl. Two configurations: CE baseline (soft mask, lam=0) and
proposed (soft mask, lam=1e-2). BLEU-4 with add-one-free cumulative
precision is computed between transmitted and recovered sentences at
SNR = 10 and 20 dB.
"""

import argparse
import collections
import csv
import math
import os
import re
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from cem_full import get_device, sinusoidal_pe

D = 128
T = 32
U = 8
ENC_LAYERS = 4
DEC_LAYERS = 2
NHEAD = 8
TAU = 0.1
SNR_LO, SNR_HI = 0.0, 25.0
PAD, UNK = 0, 1

HERE = os.path.dirname(os.path.abspath(__file__))


# ------------------------------------------------------------ data
def load_sentences(path, max_sent=200_000):
    sents = []
    with open(path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            words = re.findall(r"[a-z']+", line.lower())
            if 4 <= len(words) <= T:
                sents.append(words)
                if len(sents) >= max_sent:
                    break
    return sents


def build_vocab(sents, vocab_size):
    cnt = collections.Counter(w for s in sents for w in s)
    words = [w for w, _ in cnt.most_common(vocab_size - 2)]
    stoi = {w: i + 2 for i, w in enumerate(words)}   # 0=PAD, 1=UNK
    itos = {i: w for w, i in stoi.items()}
    return stoi, itos


def encode_corpus(sents, stoi):
    ids = torch.full((len(sents), T), PAD, dtype=torch.long)
    lens = torch.zeros(len(sents), dtype=torch.long)
    for i, s in enumerate(sents):
        for j, w in enumerate(s):
            ids[i, j] = stoi.get(w, UNK)
        lens[i] = len(s)
    return ids, lens


# ------------------------------------------------------------ model
class CEMText(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.V = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, D, padding_idx=PAD)
        self.register_buffer("pe", sinusoidal_pe(T, D))
        self.masks = nn.Parameter(torch.randn(U, D))
        enc = nn.TransformerEncoderLayer(D, NHEAD, dropout=0.0,
                                         batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(enc, ENC_LAYERS)
        self.prior = nn.Parameter(torch.randn(U, T, D) * 0.02)
        dec = nn.TransformerDecoderLayer(D, NHEAD, dropout=0.0,
                                         batch_first=True, norm_first=True)
        self.decoder = nn.TransformerDecoder(dec, DEC_LAYERS)
        self.fc_out = nn.Linear(D, vocab_size)
        self.proj = nn.Sequential(nn.Linear(D, D), nn.ReLU(), nn.Linear(D, 64))

    def transmit(self, tokens):
        B = tokens.shape[0]
        emb = self.token_embedding(tokens) * self.masks.view(1, U, 1, D)
        emb = emb + self.pe.view(1, 1, T, D)
        s = self.encoder(emb.reshape(B * U, T, D))
        return s.reshape(B, U, T, D).mean(dim=1)

    def channel(self, m, snr_db):
        h = torch.sqrt(torch.randn_like(m) ** 2 + torch.randn_like(m) ** 2) \
            * math.sqrt(0.5)
        faded = h * m
        snr = 10.0 ** (snr_db / 10.0)
        sig_pow = faded.pow(2).mean(dim=-1, keepdim=True)
        return faded + torch.sqrt(sig_pow / snr) * torch.randn_like(faded)

    def receive(self, y):
        B = y.shape[0]
        query = (self.prior * self.masks.view(U, 1, D)) + self.pe.view(1, T, D)
        query = query.unsqueeze(0).expand(B, U, T, D).reshape(B * U, T, D)
        mem = y.unsqueeze(1).expand(B, U, T, D).reshape(B * U, T, D)
        dec = self.decoder(query, mem)
        return self.fc_out(dec).reshape(B, U, T, -1), \
            dec.mean(dim=1).reshape(B, U, D)


def info_nce(za, zb):
    za, zb = F.normalize(za, dim=-1), F.normalize(zb, dim=-1)
    sim = torch.einsum("bud,bvd->buv", za, zb) / TAU
    labels = torch.arange(U, device=za.device).expand(sim.shape[0], U)
    return F.cross_entropy(sim.reshape(-1, U), labels.reshape(-1))


# ------------------------------------------------------------ BLEU
def bleu4(ref, hyp):
    """Sentence BLEU-4 with smoothing (+1 on higher n-gram precisions)."""
    if not hyp:
        return 0.0
    precisions = []
    for n in range(1, 5):
        ref_ngr = collections.Counter(
            tuple(ref[i:i + n]) for i in range(len(ref) - n + 1))
        hyp_ngr = collections.Counter(
            tuple(hyp[i:i + n]) for i in range(len(hyp) - n + 1))
        overlap = sum((ref_ngr & hyp_ngr).values())
        total = max(1, sum(hyp_ngr.values()))
        if n == 1:
            p = overlap / total
        else:
            p = (overlap + 1) / (total + 1)
        precisions.append(max(p, 1e-9))
    bp = math.exp(min(0.0, 1 - len(ref) / max(1, len(hyp))))
    return bp * math.exp(sum(math.log(p) for p in precisions) / 4)


@torch.no_grad()
def eval_bleu(model, data_ids, data_lens, snr_db, batches, batch, device,
              seed=1234):
    torch.manual_seed(seed)
    model.eval()
    scores, n = 0.0, 0
    N = data_ids.shape[0]
    for _ in range(batches):
        idx = torch.randint(0, N, (batch * U,))
        tokens = data_ids[idx].reshape(batch, U, T).to(device)
        lens = data_lens[idx].reshape(batch, U)
        y = model.channel(model.transmit(tokens), snr_db)
        logits, _ = model.receive(y)
        pred = logits.argmax(dim=-1).cpu()
        tok = tokens.cpu()
        for b in range(batch):
            for u in range(U):
                L = int(lens[b, u])
                ref = tok[b, u, :L].tolist()
                hyp = pred[b, u, :L].tolist()
                scores += bleu4(ref, hyp)
                n += 1
    return scores / n


def train_text(lam, data_ids, data_lens, steps, batch, lr, device, seed=42):
    torch.manual_seed(seed)
    V = int(max(data_ids.max().item() + 1, 2))
    model = CEMText(V).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    warmup = min(300, steps // 10)

    def fac(s):
        if s < warmup:
            return s / max(1, warmup)
        p = (s - warmup) / max(1, steps - warmup)
        return 0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * p))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, fac)
    N = data_ids.shape[0]
    model.train()
    t0 = time.time()
    for step in range(1, steps + 1):
        snr_db = SNR_LO + (SNR_HI - SNR_LO) * torch.rand(1).item()
        idx = torch.randint(0, N, (batch * U,))
        tokens = data_ids[idx].reshape(batch, U, T).to(device)
        m = model.transmit(tokens)
        y_a = model.channel(m, snr_db)
        logits, pooled_a = model.receive(y_a)
        loss = F.cross_entropy(logits.reshape(-1, model.V),
                               tokens.reshape(-1), ignore_index=PAD)
        if lam > 0:
            y_b = model.channel(m, snr_db)
            _, pooled_b = model.receive(y_b)
            loss = loss + lam * info_nce(model.proj(pooled_a),
                                         model.proj(pooled_b))
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if step % 1000 == 0 or step == 1:
            print(f"[text lam={lam}] step {step}/{steps} "
                  f"loss={loss.item():.4f} ({time.time()-t0:.0f}s)",
                  flush=True)
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default=os.path.join(HERE, "data",
                                                     "europarl-v7.fr-en.en"))
    ap.add_argument("--vocab-size", type=int, default=22000)
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--eval-batches", type=int, default=40)
    ap.add_argument("--eval-batch", type=int, default=32)
    args = ap.parse_args()

    device = get_device()
    print(f"device={device}", flush=True)
    sents = load_sentences(args.corpus)
    print(f"sentences: {len(sents)}", flush=True)
    stoi, _ = build_vocab(sents, args.vocab_size)
    ids, lens = encode_corpus(sents, stoi)
    n_train = int(len(sents) * 0.95)
    train_ids, train_lens = ids[:n_train], lens[:n_train]
    test_ids, test_lens = ids[n_train:], lens[n_train:]
    print(f"vocab={len(stoi)+2} train={n_train} test={len(sents)-n_train}",
          flush=True)

    rows = []
    for name, lam in [("text_ce", 0.0), ("text_nce0.01", 1e-2)]:
        print(f"=== training {name} ===", flush=True)
        model = train_text(lam, train_ids, train_lens, args.steps,
                           args.batch, args.lr, device)
        torch.save(model.state_dict(), os.path.join(HERE, f"model_{name}.pth"))
        for snr in [10.0, 20.0]:
            b = eval_bleu(model, test_ids, test_lens, snr,
                          args.eval_batches, args.eval_batch, device)
            print(f">>> {name} @ {snr:.0f} dB : BLEU={b:.4f}", flush=True)
            rows.append(dict(run=name, snr_db=snr, BLEU=f"{b:.6f}"))
            with open(os.path.join(HERE, "results_bleu.csv"), "w",
                      newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                w.writeheader()
                w.writerows(rows)
    print("TEXT DONE.", flush=True)


if __name__ == "__main__":
    main()
