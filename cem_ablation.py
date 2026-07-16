"""
CEM ablation study for SPL revision (Table II + Sec. III-D numbers).

Implements the SPL manuscript architecture (Eqs. (1)-(8)):
  TX : E_u = TokEmb(x_u);  E_u^masked = E_u \odot P_u;  S_u = Encoder(E_u^masked)
       M = (1/U) sum_u S_u
  CH : Y = H \odot M + N   (Rayleigh, unit average power; AWGN at target SNR)
  RX : M_u = E_u^prev \odot P_u (learnable prior);  \hat{E}_u = Decoder(M_u, Y)
  Loss: L_CE + lambda * L_InfoNCE  (views = two independent channel realizations)

Ablation configurations (Table II):
  no_mask_ce      -- P_u = 1 (fixed),        lambda = 0
  mask_ce         -- learned soft mask,      lambda = 0      (CE baseline)
  no_mask_nce     -- P_u = 1 (fixed),        lambda = 0.01
  mask_nce        -- learned soft mask,      lambda = 0.01   (proposed)
  binary_mask_nce -- fixed binary partition, lambda = 0.01   (mask-structure check)

Derived from code/JSAC/transformer.ipynb (shared multiplicative user masks,
shared Transformer, superposed transmission), adapted to the SPL tx-rx split.
"""

import argparse
import csv
import math
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------- settings
U = 8               # users
D = 128             # embedding dim
T = 32              # token length
V = 4               # vocabulary (QPSK-equivalent)
ENC_LAYERS = 4
DEC_LAYERS = 2
NHEAD = 8
LAMBDA_NCE = 1e-2   # contrastive weight (paper-optimized)
TAU = 0.1           # InfoNCE temperature
LR = 1e-4           # AdamW

CONFIGS = {
    "no_mask_ce":      dict(mask="none",   lam=0.0),
    "mask_ce":         dict(mask="soft",   lam=0.0),
    "no_mask_nce":     dict(mask="none",   lam=LAMBDA_NCE),
    "mask_nce":        dict(mask="soft",   lam=LAMBDA_NCE),
    "binary_mask_nce": dict(mask="binary", lam=LAMBDA_NCE),
}


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
    def __init__(self, mask_mode: str):
        super().__init__()
        self.mask_mode = mask_mode
        self.token_embedding = nn.Embedding(V, D)
        self.register_buffer("pe", sinusoidal_pe(T, D))

        if mask_mode == "soft":
            # learned soft mask P_u (Eq. (1)); multiplicative, as in the JSAC code
            self.masks = nn.Parameter(torch.randn(U, D))
        elif mask_mode == "binary":
            # fixed hard partition: D/U coordinates per user
            m = torch.zeros(U, D)
            per = D // U
            for u in range(U):
                m[u, u * per:(u + 1) * per] = 1.0
            self.register_buffer("masks", m)
        else:  # "none"
            self.register_buffer("masks", torch.ones(U, D))

        enc_layer = nn.TransformerEncoderLayer(
            d_model=D, nhead=NHEAD, dropout=0.0, batch_first=True,
            norm_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=ENC_LAYERS)

        # learnable user-specific prior E_u^prev (zero-mean Gaussian init)
        self.prior = nn.Parameter(torch.randn(U, T, D) * 0.02)

        dec_layer = nn.TransformerDecoderLayer(
            d_model=D, nhead=NHEAD, dropout=0.0, batch_first=True,
            norm_first=True)
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=DEC_LAYERS)
        self.fc_out = nn.Linear(D, V)

        # two-layer MLP projection head for InfoNCE
        self.proj = nn.Sequential(nn.Linear(D, D), nn.ReLU(), nn.Linear(D, 64))

    def transmit(self, tokens):
        """tokens: (B, U, T) -> multiplexed M: (B, T, D)  (Eqs. (1)-(3))"""
        B = tokens.shape[0]
        emb = self.token_embedding(tokens)                    # (B, U, T, D)
        masked = emb * self.masks.view(1, U, 1, D)            # Eq. (1)
        masked = masked + self.pe.view(1, 1, T, D)            # positional encoding
        s = self.encoder(masked.reshape(B * U, T, D))         # Eq. (2)
        s = s.reshape(B, U, T, D)
        return s.mean(dim=1)                                  # Eq. (3)

    def channel(self, m, snr_db):
        """Rayleigh fading + AWGN (Eq. (4)); unit average channel power."""
        h = torch.sqrt(torch.randn_like(m) ** 2 + torch.randn_like(m) ** 2) \
            * math.sqrt(0.5)
        faded = h * m
        snr = 10.0 ** (snr_db / 10.0)
        sig_pow = faded.pow(2).mean(dim=-1, keepdim=True)
        noise = torch.sqrt(sig_pow / snr) * torch.randn_like(faded)
        return faded + noise

    def receive(self, y):
        """y: (B, T, D) -> logits: (B, U, T, V), pooled hidden: (B, U, D)"""
        B = y.shape[0]
        query = (self.prior * self.masks.view(U, 1, D)) + self.pe.view(1, T, D)
        query = query.unsqueeze(0).expand(B, U, T, D).reshape(B * U, T, D)
        mem = y.unsqueeze(1).expand(B, U, T, D).reshape(B * U, T, D)
        dec = self.decoder(query, mem)                        # Eq. (5)
        logits = self.fc_out(dec).reshape(B, U, T, V)
        pooled = dec.mean(dim=1).reshape(B, U, D)             # \bar{s}_u
        return logits, pooled


def info_nce(za, zb, tau=TAU):
    """za, zb: (B, U, dz); negatives = other users within the batch item (Eq. (7))."""
    za = F.normalize(za, dim=-1)
    zb = F.normalize(zb, dim=-1)
    sim = torch.einsum("bud,bvd->buv", za, zb) / tau          # (B, U, U)
    labels = torch.arange(U, device=za.device).expand(sim.shape[0], U)
    return F.cross_entropy(sim.reshape(-1, U), labels.reshape(-1))


def mean_abs_inter_user_cos(pooled):
    """pooled: (B, U, D) -> scalar mean |cos| over user pairs (Sec. III-D rho-bar)."""
    z = F.normalize(pooled, dim=-1)
    sim = torch.einsum("bud,bvd->buv", z, z)
    iu, jv = torch.triu_indices(U, U, offset=1)
    return sim[:, iu, jv].abs().mean().item()


def train_one(config_name, snr_db, steps, batch, device, log_every=1000, seed=42):
    torch.manual_seed(seed)
    cfg = CONFIGS[config_name]
    model = CEMModel(cfg["mask"]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    warmup = min(300, steps // 10)

    def lr_at(s):
        if s < warmup:
            return LR * s / max(1, warmup)
        p = (s - warmup) / max(1, steps - warmup)
        return LR * (0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * p)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: lr_at(s) / LR)
    lam = cfg["lam"]
    rho_log = []

    model.train()
    t0 = time.time()
    for step in range(1, steps + 1):
        tokens = torch.randint(0, V, (batch, U, T), device=device)
        m = model.transmit(tokens)
        y_a = model.channel(m, snr_db)
        logits, pooled_a = model.receive(y_a)
        loss = F.cross_entropy(logits.reshape(-1, V), tokens.reshape(-1))  # Eq. (6)

        if lam > 0:
            y_b = model.channel(m, snr_db)                     # second realization
            _, pooled_b = model.receive(y_b)
            za = model.proj(pooled_a)
            zb = model.proj(pooled_b)
            loss = loss + lam * info_nce(za, zb)               # Eq. (8)

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
                ser_probe, _ = evaluate(model, snr_db, 10, 128, device, seed=7)
                torch.set_rng_state(rng_cpu)
                if rng_mps is not None:
                    torch.mps.set_rng_state(rng_mps)
                model.train()
            rho_log.append((step, rho))
            print(f"[{config_name} | {snr_db} dB] step {step}/{steps} "
                  f"loss={loss.item():.4f} rho={rho:.4f} SER~{ser_probe:.4f} "
                  f"({time.time() - t0:.0f}s)", flush=True)
    return model, rho_log


@torch.no_grad()
def evaluate(model, snr_db, eval_batches, batch, device, seed=1234):
    torch.manual_seed(seed)
    model.eval()
    err, tot = 0, 0
    rho_sum, rho_n = 0.0, 0
    for _ in range(eval_batches):
        tokens = torch.randint(0, V, (batch, U, T), device=device)
        m = model.transmit(tokens)
        y = model.channel(m, snr_db)
        logits, pooled = model.receive(y)
        pred = logits.argmax(dim=-1)
        err += (pred != tokens).sum().item()
        tot += tokens.numel()
        rho_sum += mean_abs_inter_user_cos(pooled)
        rho_n += 1
    return err / tot, rho_sum / rho_n


def main():
    global U, LR
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--eval-batches", type=int, default=400)
    ap.add_argument("--eval-batch", type=int, default=256)
    ap.add_argument("--snrs", type=float, nargs="+", default=[10.0, 20.0])
    ap.add_argument("--configs", nargs="+", default=list(CONFIGS.keys()))
    ap.add_argument("--out", default="results_ablation.csv")
    ap.add_argument("--users", type=int, default=U)
    ap.add_argument("--lr", type=float, default=LR)
    args = ap.parse_args()
    U = args.users
    LR = args.lr

    device = get_device()
    print(f"device={device}, steps={args.steps}, batch={args.batch}", flush=True)

    outdir = os.path.dirname(os.path.abspath(args.out))
    os.makedirs(outdir, exist_ok=True)
    rows = []
    for name in args.configs:
        for snr in args.snrs:
            print(f"=== training {name} @ {snr} dB ===", flush=True)
            model, rho_log = train_one(name, snr, args.steps, args.batch, device)
            torch.save(model.state_dict(),
                       os.path.join(outdir, f"model_{name}_{int(snr)}dB.pth"))
            ser, rho = evaluate(model, snr, args.eval_batches, args.eval_batch, device)
            n_sym = args.eval_batches * args.eval_batch * U * T
            print(f">>> {name} @ {snr} dB : SER={ser:.6f} rho={rho:.4f} "
                  f"({n_sym:.2e} symbols)", flush=True)
            rows.append(dict(config=name, snr_db=snr, SER=ser, rho_abs_cos=rho,
                             steps=args.steps, batch=args.batch, n_symbols=n_sym))
            with open(args.out, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                w.writeheader()
                w.writerows(rows)
            # save rho trajectory for the training-dynamics sentence in Sec. III-D
            traj = os.path.join(outdir, f"rho_traj_{name}_{int(snr)}dB.csv")
            with open(traj, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["step", "rho_abs_cos"])
                w.writerows(rho_log)
    print("done.", flush=True)


if __name__ == "__main__":
    main()
