"""
DL-9 | Convolutional Autoencoder pada MNIST (5.000 citra)
Split 80:10:10 (4.000 train / 500 val / 500 test), stratified per kelas.

Cara pakai:
    pip install torch torchvision scikit-image pandas matplotlib
    python ae_mnist_dl9.py

Output (folder ./hasil_ae):
    metrik/              -> konfigurasi dan metrik evaluasi CSV
    riwayat/             -> loss train/validasi per epoch
    grafik/              -> grafik loss, rekonstruksi, dan perbandingan run
    model/               -> bobot terbaik (val loss terendah)
"""
import os
import time
import platform
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision
from torchvision import datasets, transforms
from skimage.metrics import structural_similarity as ssim_fn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------
# 1. KONFIGURASI RUN  (ubah di sini bila perlu)
# ----------------------------------------------------------------------
KELOMPOK = "K9"
SEED = 42
N_PER_CLASS = 500            # 10 kelas x 500 = 5.000 citra
SPLIT = (0.8, 0.1, 0.1)      # train, val, test
OUT_DIR = "hasil_ae"
MODEL_DIR = os.path.join(OUT_DIR, "model")
METRIC_DIR = os.path.join(OUT_DIR, "metrik")
HISTORY_DIR = os.path.join(OUT_DIR, "riwayat")
PLOT_DIR = os.path.join(OUT_DIR, "grafik")
LOSS_DIR = os.path.join(PLOT_DIR, "loss")
RECON_DIR = os.path.join(PLOT_DIR, "rekonstruksi")
DATA_DIR = "data"

RUNS = [
    # run_id,  latent, lr,    epoch, batch, parameter yang diubah
    dict(run_id="AE-01", latent=16, lr=1e-3, epochs=30, batch=128, changed="Baseline"),
    dict(run_id="AE-02", latent=32, lr=1e-3, epochs=30, batch=128, changed="Latent dimension 16 -> 32"),
    dict(run_id="AE-03", latent=16, lr=1e-4, epochs=30, batch=128, changed="Learning rate 1e-3 -> 1e-4"),
    # Opsional (memperjelas tren latent dim). Hapus baris ini jika cukup 3 run.
    dict(run_id="AE-04", latent=8, lr=1e-3, epochs=30, batch=128, changed="Latent dimension 16 -> 8"),
    dict(run_id="AE-05", latent=16, lr=1e-4, epochs=200, batch=128, changed="lr 1e-4, epoch 200 (uji konvergensi)"),
]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ----------------------------------------------------------------------
# 2. DATA: ambil 5.000 citra (seimbang per kelas) lalu split 80:10:10
# ----------------------------------------------------------------------
def load_data():
    ds = datasets.MNIST(DATA_DIR, train=True, download=True,
                        transform=transforms.ToTensor())  # skala [0,1], shape 1x28x28
    targets = ds.targets.numpy()
    rng = np.random.RandomState(SEED)

    idx_train, idx_val, idx_test = [], [], []
    n_tr = int(N_PER_CLASS * SPLIT[0])
    n_va = int(N_PER_CLASS * SPLIT[1])
    for c in range(10):
        idx = np.where(targets == c)[0]
        idx = rng.choice(idx, N_PER_CLASS, replace=False)
        idx_train += idx[:n_tr].tolist()
        idx_val += idx[n_tr:n_tr + n_va].tolist()
        idx_test += idx[n_tr + n_va:].tolist()

    def to_tensor(indices):
        x = torch.stack([ds[i][0] for i in indices])
        y = torch.tensor([int(targets[i]) for i in indices])
        return x, y

    return to_tensor(idx_train), to_tensor(idx_val), to_tensor(idx_test)


# ----------------------------------------------------------------------
# 3. MODEL: Convolutional Autoencoder
# ----------------------------------------------------------------------
class ConvAE(nn.Module):
    def __init__(self, latent_dim: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1),   # 28 -> 14
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),  # 14 -> 7
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, latent_dim),          # bottleneck
        )
        self.decoder_fc = nn.Sequential(
            nn.Linear(latent_dim, 64 * 7 * 7),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),  # 7 -> 14
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 3, stride=2, padding=1, output_padding=1),   # 14 -> 28
            nn.Sigmoid(),
        )

    def forward(self, x):
        z = self.encoder(x)
        h = self.decoder_fc(z).view(-1, 64, 7, 7)
        return self.decoder(h)


# ----------------------------------------------------------------------
# 4. METRIK
# ----------------------------------------------------------------------
@torch.no_grad()
def predict(model, x, bs=500):
    model.eval()
    outs = []
    for i in range(0, len(x), bs):
        outs.append(model(x[i:i + bs].to(device)).cpu())
    return torch.cat(outs)


def compute_metrics(x, xr):
    """x, xr: tensor N x 1 x 28 x 28 pada rentang [0,1]. Rata-rata per citra."""
    mse_i = ((x - xr) ** 2).flatten(1).mean(1)
    mae_i = (x - xr).abs().flatten(1).mean(1)
    psnr_i = 10 * torch.log10(1.0 / mse_i.clamp_min(1e-12))     # MAX = 1.0
    a, b = x.squeeze(1).numpy(), xr.squeeze(1).numpy()
    ssim_i = [ssim_fn(a[i], b[i], data_range=1.0) for i in range(len(a))]
    return dict(MSE=mse_i.mean().item(), MAE=mae_i.mean().item(),
                PSNR=psnr_i.mean().item(), SSIM=float(np.mean(ssim_i)))


# ----------------------------------------------------------------------
# 5. TRAINING
# ----------------------------------------------------------------------
def run_experiment(cfg, data):
    (xtr, _), (xva, _), (xte, yte) = data
    set_seed(SEED)
    model = ConvAE(cfg["latent"]).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    loss_fn = nn.MSELoss()
    g = torch.Generator().manual_seed(SEED)

    hist, best_val = [], float("inf")
    best_path = os.path.join(MODEL_DIR, f"best_{cfg['run_id']}.pt")
    t0 = time.time()
    for ep in range(1, cfg["epochs"] + 1):
        model.train()
        perm = torch.randperm(len(xtr), generator=g)
        tot = 0.0
        for i in range(0, len(xtr), cfg["batch"]):
            xb = xtr[perm[i:i + cfg["batch"]]].to(device)
            loss = loss_fn(model(xb), xb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * len(xb)
        tr_loss = tot / len(xtr)
        va_loss = loss_fn(predict(model, xva), xva).item()
        hist.append(dict(epoch=ep, train_mse=tr_loss, val_mse=va_loss))
        if va_loss < best_val:
            best_val = va_loss
            torch.save(model.state_dict(), best_path)
        print(f"[{cfg['run_id']}] ep {ep:02d}/{cfg['epochs']}  train={tr_loss:.5f}  val={va_loss:.5f}")
    minutes = (time.time() - t0) / 60

    # evaluasi TEST dengan bobot terbaik (val loss terendah)
    model.load_state_dict(torch.load(best_path, map_location=device))
    xr = predict(model, xte)
    m = compute_metrics(xte, xr)
    print(f"[{cfg['run_id']}] TEST: {m}")

    pd.DataFrame(hist).to_csv(os.path.join(HISTORY_DIR, f"history_{cfg['run_id']}.csv"), index=False)
    plot_loss(cfg, hist)
    plot_recon(cfg, xte, xr, yte)
    return model, m, minutes, n_params, len(xte)


# ----------------------------------------------------------------------
# 6. GRAFIK
# ----------------------------------------------------------------------
def plot_loss(cfg, hist):
    df = pd.DataFrame(hist)
    plt.figure(figsize=(5, 3.5))
    plt.plot(df.epoch, df.train_mse, label="Train")
    plt.plot(df.epoch, df.val_mse, label="Validasi")
    plt.xlabel("Epoch"); plt.ylabel("MSE"); plt.yscale("log")
    plt.title(f"Kurva loss {cfg['run_id']} (latent={cfg['latent']}, lr={cfg['lr']})")
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(LOSS_DIR, f"loss_{cfg['run_id']}.png"), dpi=200)
    plt.close()


def plot_recon(cfg, x, xr, y, n=10):
    # satu contoh per kelas (0-9) agar gambar representatif
    idx = [int((y == c).nonzero()[0]) for c in range(n)]
    fig, ax = plt.subplots(2, n, figsize=(n * 1.1, 3.0))
    for j, i in enumerate(idx):
        ax[0, j].imshow(x[i, 0], cmap="gray"); ax[0, j].axis("off")
        ax[1, j].imshow(xr[i, 0], cmap="gray"); ax[1, j].axis("off")
        ax[0, j].set_title(str(j), fontsize=8)
    fig.text(0.005, 0.70, "Asli", fontsize=8, rotation=90, va="center")
    fig.text(0.005, 0.28, "Rekonstruksi", fontsize=8, rotation=90, va="center")
    plt.suptitle(f"{cfg['run_id']} (latent={cfg['latent']})", fontsize=9)
    plt.tight_layout(rect=(0.02, 0, 1, 1))
    plt.savefig(os.path.join(RECON_DIR, f"recon_{cfg['run_id']}.png"), dpi=200)
    plt.close()


def plot_summary(df):
    fig, ax = plt.subplots(1, 3, figsize=(11, 3.2))
    for a, col in zip(ax, ["MSE", "PSNR (dB)", "SSIM"]):
        a.bar(df["Run ID"], df[col]); a.set_title(col)
        for i, v in enumerate(df[col]):
            a.text(i, v, f"{v:.4f}" if col != "PSNR (dB)" else f"{v:.2f}",
                   ha="center", va="bottom", fontsize=7)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "perbandingan_run.png"), dpi=200)
    plt.close()


# ----------------------------------------------------------------------
# 7. MAIN
# ----------------------------------------------------------------------
def main():
    for output_dir in (MODEL_DIR, METRIC_DIR, HISTORY_DIR, LOSS_DIR, RECON_DIR):
        os.makedirs(output_dir, exist_ok=True)
    set_seed(SEED)
    data = load_data()
    print("Ukuran split (train/val/test):", [len(d[0]) for d in data])

    hw = torch.cuda.get_device_name(0) if torch.cuda.is_available() else platform.processor() or "CPU"
    fw = f"PyTorch {torch.__version__} / torchvision {torchvision.__version__}"

    cfg_rows, res_rows = [], []
    for cfg in RUNS:
        _, m, minutes, n_params, n_test = run_experiment(cfg, data)
        cfg_rows.append({
            "Kelompok": KELOMPOK, "Run ID": cfg["run_id"], "Model": "Convolutional AE",
            "Dataset & versi": "MNIST (torchvision), subset 5.000 citra seimbang 500/kelas",
            "Split train/val/test": "80/10/10 = 4000/500/500 (stratified, seed 42)",
            "Preprocessing": "ToTensor, normalisasi [0,1]",
            "Input shape": "28x28x1",
            "Arsitektur & versi kode":
                f"Enc: Conv(1-32,s2)-Conv(32-64,s2)-FC({cfg['latent']}); "
                f"Dec: FC-ConvT(64-32,s2)-ConvT(32-1,s2)-Sigmoid; {n_params} param; ae_mnist_dl9.py",
            "Latent/noise dim": cfg["latent"], "Kondisi/label": "Tidak berlaku",
            "Epoch": cfg["epochs"], "Batch size": cfg["batch"], "Optimizer": "Adam",
            "Learning rate": cfg["lr"], "Loss": "MSE", "Seed": SEED,
            "Framework & versi": fw, "Hardware": hw,
            "Waktu latih (menit)": round(minutes, 3),
            "Parameter yang diubah": cfg["changed"], "Link kode/log": "isi tautan GitHub/Drive",
        })
        res_rows.append({
            "Kelompok": KELOMPOK, "Run ID": cfg["run_id"], "Model": "AE", "Split evaluasi": "Test",
            "Jumlah sampel": n_test, "MSE": m["MSE"], "MAE": m["MAE"], "PSNR (dB)": m["PSNR"],
            "SSIM": m["SSIM"], "Reconstruction loss": m["MSE"],
            "KL divergence": "", "ELBO / negative ELBO": "", "FID": "", "KID": "",
            "Conditional accuracy": "", "Diversity metric": "",
            "Train time (menit)": round(minutes, 3),
            "Protokol evaluasi / fitur":
                "Data range [0,1]; MSE/MAE/PSNR/SSIM dihitung per citra lalu dirata-rata; "
                "PSNR MAX=1; SSIM skimage (win 7, data_range=1); bobot dengan val loss terendah",
            "Link grafik & citra": "isi tautan", "Catatan": "Satu seed (42)",
        })

    pd.DataFrame(cfg_rows).to_csv(os.path.join(METRIC_DIR, "konfigurasi.csv"), index=False)
    res = pd.DataFrame(res_rows)
    res.to_csv(os.path.join(METRIC_DIR, "hasil_metrik.csv"), index=False)
    plot_summary(res)
    print("\nRingkasan hasil TEST:\n", res[["Run ID", "MSE", "MAE", "PSNR (dB)", "SSIM"]].to_string(index=False))
    print(f"\nSemua file tersimpan di: {os.path.abspath(OUT_DIR)}")


if __name__ == "__main__":
    main()
