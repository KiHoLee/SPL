"""Post-processing for the full regeneration run:
1) Large-sample re-evaluation of statistically thin high-SNR points
   (models are loaded from the saved checkpoints; no retraining).
2) Inter-user similarity (rho) measurements for Sec. III-D, in two spaces:
   - z-space: projection-head outputs (where InfoNCE operates)
   - encoder-space: per-token masked encoder outputs S_u (transmit side,
     directly related to inter-user interference in the multiplexed signal)
Outputs: results_reeval.csv, rho_report.csv
"""

import csv
import os
import time

import torch
import torch.nn.functional as F

from cem_full import CEMModel, RUNS, T, get_device, mean_abs_inter_user_cos

HERE = os.path.dirname(os.path.abspath(__file__))
RUNMAP = {name: (u, m, lam) for name, u, m, lam in RUNS}

# (run, snr, eval_batches, batch) — sized for >=~1000 expected errors
REEVAL = [
    ("mask_ce_U8",        15.0, 2500, 256),
    ("mask_ce_U8",        20.0, 2500, 256),
    ("mask_ce_U8",        25.0, 2500, 256),
    ("mask_nce0.01_U8",   15.0, 2500, 256),
    ("mask_nce0.01_U8",   20.0, 2500, 256),
    ("mask_nce0.01_U8",   25.0, 2500, 256),
    ("mask_nce0.0001_U8", 15.0, 2500, 256),
    ("mask_nce0.0001_U8", 20.0, 2500, 256),
    ("mask_nce0.001_U8",  15.0, 2500, 256),
    ("mask_nce0.001_U8",  20.0, 2500, 256),
    ("mask_nce0.1_U8",    15.0, 2500, 256),
    ("mask_nce0.1_U8",    20.0, 2500, 256),
    ("binary_nce_U8",     20.0, 2500, 256),
    ("mask_ce_U4",        10.0, 3000, 256),
    ("mask_nce0.01_U4",   10.0, 3000, 256),
]

RHO_RUNS = ["mask_ce_U8", "mask_nce0.01_U8"]
RHO_SNR = 10.0
RHO_BATCHES = 200


def load_model(name, device):
    u, mask, _ = RUNMAP[name]
    model = CEMModel(u, mask).to(device)
    sd = torch.load(os.path.join(HERE, f"model_{name}.pth"),
                    map_location=device, weights_only=True)
    model.load_state_dict(sd)
    model.eval()
    return model, u


@torch.no_grad()
def big_eval(model, num_users, snr_db, batches, batch, device, seed=999):
    torch.manual_seed(seed)
    err, tot = 0, 0
    for _ in range(batches):
        tokens = torch.randint(0, 4, (batch, num_users, T), device=device)
        y = model.channel(model.transmit(tokens), snr_db)
        logits, _ = model.receive(y)
        err += (logits.argmax(dim=-1) != tokens).sum().item()
        tot += tokens.numel()
    return err, tot


@torch.no_grad()
def rho_measure(model, num_users, snr_db, batches, device, seed=555):
    """Returns (rho_z, rho_enc): mean |inter-user cos| in projection space
    and in per-token masked-encoder-output space."""
    torch.manual_seed(seed)
    rho_z_sum, rho_e_sum = 0.0, 0.0
    iu, jv = torch.triu_indices(num_users, num_users, offset=1)
    for _ in range(batches):
        tokens = torch.randint(0, 4, (64, num_users, T), device=device)
        B = tokens.shape[0]
        # encoder-space: per-token S_u before aggregation
        emb = model.token_embedding(tokens) * model.masks.view(1, num_users, 1, -1)
        emb = emb + model.pe.view(1, 1, T, -1)
        s = model.encoder(emb.reshape(B * num_users, T, -1))
        s = s.reshape(B, num_users, T, -1)
        sn = F.normalize(s, dim=-1)
        sim_e = torch.einsum("butd,bvtd->buvt", sn, sn)      # (B,U,U,T)
        rho_e_sum += sim_e[:, iu, jv, :].abs().mean().item()
        # z-space: projection of pooled receiver representation
        m = s.mean(dim=1)
        y = model.channel(m, snr_db)
        _, pooled = model.receive(y)
        z = F.normalize(model.proj(pooled), dim=-1)
        sim_z = torch.einsum("bud,bvd->buv", z, z)
        rho_z_sum += sim_z[:, iu, jv].abs().mean().item()
    return rho_z_sum / batches, rho_e_sum / batches


def main():
    device = get_device()
    print(f"device={device}", flush=True)

    # --- rho measurements first (fast) ---
    rho_rows = []
    for name in RHO_RUNS:
        model, u = load_model(name, device)
        rho_z, rho_e = rho_measure(model, u, RHO_SNR, RHO_BATCHES, device)
        print(f"RHO {name}: z-space={rho_z:.4f} encoder-space={rho_e:.4f}",
              flush=True)
        rho_rows.append(dict(run=name, snr_db=RHO_SNR,
                             rho_z=f"{rho_z:.6f}", rho_enc=f"{rho_e:.6f}"))
    with open(os.path.join(HERE, "rho_report.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rho_rows[0].keys()))
        w.writeheader()
        w.writerows(rho_rows)

    # --- large-sample re-evaluation ---
    rows = []
    for name, snr, batches, batch in REEVAL:
        model, u = load_model(name, device)
        eb = batch if u <= 8 else batch // 2
        t0 = time.time()
        err, tot = big_eval(model, u, snr, batches, eb, device)
        ser = err / tot
        print(f"REEVAL {name} @ {snr:.0f} dB : SER={ser:.3e} "
              f"({err} err / {tot:.2e} sym, {time.time()-t0:.0f}s)",
              flush=True)
        rows.append(dict(run=name, snr_db=snr, SER=f"{ser:.10f}",
                         n_errors=err, n_symbols=tot))
        with open(os.path.join(HERE, "results_reeval.csv"), "w",
                  newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    print("POSTPROC DONE.", flush=True)


if __name__ == "__main__":
    main()
