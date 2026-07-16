"""
Full regeneration of SPL results with the improved CEM implementation:
  Fig. 2  : lambda sweep at U=8            (SER vs lambda, SNR in {5,10,15,20})
  Fig. 3  : SNR vs SER for U in {4,8,16}   (CE baseline vs CE+NCE, + DE bound)
  Table II: ablation at U=8                (no-mask/CE, mask/CE, no-mask/NCE,
                                            mask/NCE, binary-mask/NCE)
  Sec III-D: mean |inter-user cos| (rho) final values and training trajectory.

Protocol: ONE model per configuration, trained with per-batch random SNR
uniform in [0, 25] dB, evaluated at SNR in {0,5,10,15,20,25} dB.
Architecture follows the SPL manuscript Eqs. (1)-(8); implementation derived
from code/JSAC/transformer.ipynb (multiplicative user masks, shared
Transformer, superposed transmission) with the SPL tx-rx split, sinusoidal
positional encoding, pre-norm Transformer blocks, and no dropout.
"""

import argparse
import csv
import math
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

D = 128
T = 32
V = 4
ENC_LAYERS = 4
DEC_LAYERS = 2
NHEAD = 8
TAU = 0.1
EVAL_SNRS = [0.0, 5.0, 10.0, 15.0, 20.0, 25.0]
SNR_LO, SNR_HI = 0.0, 25.0

# name, users, mask_mode, lambda
RUNS = [
    ("mask_ce_U8",        8,  "soft",   0.0),
    ("mask_nce0.01_U8",   8,  "soft",   1e-2),
    ("no_mask_ce_U8",     8,  "none",   0.0),
    ("no_mask_nce_U8",    8,  "none",   1e-2),
    ("binary_nce_U8",     8,  "binary", 1e-2),
    ("mask_nce0.0001_U8", 8,  "soft",   1e-4),
    ("mask_nce0.001_U8",  8,  "soft",   1e-3),
    ("mask_nce0.1_U8",    8,  "soft",   1e-1),
    ("mask_nce0.0003_U8", 8,  "soft",   3e-4),
    ("mask_nce0.003_U8",  8,  "soft",   3e-3),
    ("mask_nce0.03_U8",   8,  "soft",   3e-2),
    ("mask_nce0.3_U8",    8,  "soft",   3e-1),
    ("mask_ce_U4",        4,  "soft",   0.0),
    ("mask_nce0.01_U4",   4,  "soft",   1e-2),
    ("mask_ce_U16",       16, "soft",   0.0),
    ("mask_nce0.01_U16",  16, "soft",   1e-2),
]


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def sinusoidal_pe(t_len, dim):
    pos = torch.arange(t_len).unsqueeze(1).float()
    div = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
    pe = torch.zeros(t_len, dim)
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe


class CEMModel(nn.Module):
    def __init__(self, num_users, mask_mode):
        super().__init__()
        self.U = num_users
        self.token_embedding = nn.Embedding(V, D)
        self.register_buffer("pe", sinusoidal_pe(T, D))

        if mask_mode == "soft":
            self.masks = nn.Parameter(torch.randn(num_users, D))
        elif mask_mode == "binary":
            m = torch.zeros(num_users, D)
            per = D // num_users
            for u in range(num_users):
                m[u, u * per:(u + 1) * per] = 1.0
            self.register_buffer("masks", m)
        else:
            self.register_buffer("masks", torch.ones(num_users, D))

        enc_layer = nn.TransformerEncoderLayer(
            d_model=D, nhead=NHEAD, dropout=0.0, batch_first=True,
            norm_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=ENC_LAYERS)
        self.prior = nn.Parameter(torch.randn(num_users, T, D) * 0.02)
        dec_layer = nn.TransformerDecoderLayer(
            d_model=D, nhead=NHEAD, dropout=0.0, batch_first=True,
            norm_first=True)
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=DEC_LAYERS)
        self.fc_out = nn.Linear(D, V)
        self.proj = nn.Sequential(nn.Linear(D, D), nn.ReLU(), nn.Linear(D, 64))

    def transmit(self, tokens):
        B, U = tokens.shape[:2]
        emb = self.token_embedding(tokens)
        masked = emb * self.masks.view(1, U, 1, D)                 # Eq. (1)
        masked = masked + self.pe.view(1, 1, T, D)
        s = self.encoder(masked.reshape(B * U, T, D))              # Eq. (2)
        return s.reshape(B, U, T, D).mean(dim=1)                   # Eq. (3)

    def channel(self, m, snr_db):
        h = torch.sqrt(torch.randn_like(m) ** 2 + torch.randn_like(m) ** 2) \
            * math.sqrt(0.5)
        faded = h * m                                              # Eq. (4)
        snr = 10.0 ** (snr_db / 10.0)
        sig_pow = faded.pow(2).mean(dim=-1, keepdim=True)
        return faded + torch.sqrt(sig_pow / snr) * torch.randn_like(faded)

    def receive(self, y):
        B, U = y.shape[0], self.U
        query = (self.prior * self.masks.view(U, 1, D)) + self.pe.view(1, T, D)
        query = query.unsqueeze(0).expand(B, U, T, D).reshape(B * U, T, D)
        mem = y.unsqueeze(1).expand(B, U, T, D).reshape(B * U, T, D)
        dec = self.decoder(query, mem)                             # Eq. (5)
        logits = self.fc_out(dec).reshape(B, U, T, V)
        pooled = dec.mean(dim=1).reshape(B, U, D)
        return logits, pooled


def info_nce(za, zb, tau=TAU):
    U = za.shape[1]
    za = F.normalize(za, dim=-1)
    zb = F.normalize(zb, dim=-1)
    sim = torch.einsum("bud,bvd->buv", za, zb) / tau               # Eq. (7)
    labels = torch.arange(U, device=za.device).expand(sim.shape[0], U)
    return F.cross_entropy(sim.reshape(-1, U), labels.reshape(-1))


def mean_abs_inter_user_cos(pooled):
    U = pooled.shape[1]
    z = F.normalize(pooled, dim=-1)
    sim = torch.einsum("bud,bvd->buv", z, z)
    iu, jv = torch.triu_indices(U, U, offset=1)
    return sim[:, iu, jv].abs().mean().item()


@torch.no_grad()
def evaluate(model, num_users, snr_db, eval_batches, batch, device, seed=1234):
    torch.manual_seed(seed)
    model.eval()
    err, tot, rho_sum = 0, 0, 0.0
    for _ in range(eval_batches):
        tokens = torch.randint(0, V, (batch, num_users, T), device=device)
        m = model.transmit(tokens)
        y = model.channel(m, snr_db)
        logits, pooled = model.receive(y)
        err += (logits.argmax(dim=-1) != tokens).sum().item()
        tot += tokens.numel()
        rho_sum += mean_abs_inter_user_cos(pooled)
    return err / tot, rho_sum / eval_batches


def train_one(name, num_users, mask_mode, lam, steps, batch, lr, device,
              outdir, log_every=1000, seed=42):
    torch.manual_seed(seed)
    model = CEMModel(num_users, mask_mode).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    warmup = min(300, steps // 10)

    def fac(s):
        if s < warmup:
            return s / max(1, warmup)
        p = (s - warmup) / max(1, steps - warmup)
        return 0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * p))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, fac)
    rho_log = []
    model.train()
    t0 = time.time()
    for step in range(1, steps + 1):
        snr_db = SNR_LO + (SNR_HI - SNR_LO) * torch.rand(1).item()
        tokens = torch.randint(0, V, (batch, num_users, T), device=device)
        m = model.transmit(tokens)
        y_a = model.channel(m, snr_db)
        logits, pooled_a = model.receive(y_a)
        loss = F.cross_entropy(logits.reshape(-1, V), tokens.reshape(-1))
        if lam > 0:
            y_b = model.channel(m, snr_db)
            _, pooled_b = model.receive(y_b)
            loss = loss + lam * info_nce(model.proj(pooled_a),
                                         model.proj(pooled_b))
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()

        if step % log_every == 0 or step == 1:
            with torch.no_grad():
                rho = mean_abs_inter_user_cos(pooled_a.detach())
                rng_cpu = torch.get_rng_state()
                rng_mps = (torch.mps.get_rng_state()
                           if device.type == "mps" else None)
                ser10, _ = evaluate(model, num_users, 10.0, 10, 128, device)
                torch.set_rng_state(rng_cpu)
                if rng_mps is not None:
                    torch.mps.set_rng_state(rng_mps)
                model.train()
            rho_log.append((step, rho, ser10))
            print(f"[{name}] step {step}/{steps} loss={loss.item():.4f} "
                  f"rho={rho:.4f} SER10~{ser10:.5f} ({time.time()-t0:.0f}s)",
                  flush=True)

    torch.save(model.state_dict(), os.path.join(outdir, f"model_{name}.pth"))
    with open(os.path.join(outdir, f"rho_traj_{name}.csv"), "w",
              newline="") as f:
        w = csv.writer(f)
        w.writerow(["step", "rho_abs_cos", "ser10_probe"])
        w.writerows(rho_log)
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--eval-batches", type=int, default=150)
    ap.add_argument("--eval-batch", type=int, default=256)
    ap.add_argument("--only", nargs="*", default=None,
                    help="subset of run names")
    ap.add_argument("--out", default="results_full.csv")
    args = ap.parse_args()

    device = get_device()
    outdir = os.path.dirname(os.path.abspath(args.out)) or "."
    os.makedirs(outdir, exist_ok=True)
    print(f"device={device} steps={args.steps} batch={args.batch} "
          f"lr={args.lr}", flush=True)

    rows = []
    out_path = os.path.abspath(args.out)
    if os.path.exists(out_path):          # resume: keep prior rows
        with open(out_path) as f:
            rows = list(csv.DictReader(f))
    done = {r["run"] for r in rows}

    for name, num_users, mask_mode, lam in RUNS:
        if args.only and name not in args.only:
            continue
        if name in done:
            print(f"=== skip {name} (already in CSV) ===", flush=True)
            continue
        print(f"=== training {name} (U={num_users}, mask={mask_mode}, "
              f"lam={lam}) ===", flush=True)
        model = train_one(name, num_users, mask_mode, lam, args.steps,
                          args.batch, args.lr, device, outdir)
        eb = args.eval_batch if num_users <= 8 else args.eval_batch // 2
        for snr in EVAL_SNRS:
            ser, rho = evaluate(model, num_users, snr, args.eval_batches,
                                eb, device)
            n_sym = args.eval_batches * eb * num_users * T
            print(f">>> {name} @ {snr:.0f} dB : SER={ser:.6f} rho={rho:.4f} "
                  f"({n_sym:.2e} sym)", flush=True)
            rows.append(dict(run=name, users=num_users, mask=mask_mode,
                             lam=lam, snr_db=snr, SER=f"{ser:.8f}",
                             rho_abs_cos=f"{rho:.6f}",
                             steps=args.steps, batch=args.batch))
            with open(out_path, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                w.writeheader()
                w.writerows(rows)
    print("ALL DONE.", flush=True)


if __name__ == "__main__":
    main()
