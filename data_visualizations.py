# Side-by-side 2D embeddings: PCA vs Kernel PCA vs UMAP vs t-SNE vs PyTorch Denoising AE
# pip install ucimlrepo umap-learn torch scikit-learn matplotlib

import numpy as np
import matplotlib.pyplot as plt
from ucimlrepo import fetch_ucirepo

import umap
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.decomposition import KernelPCA

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split


# -------------------------
# Load + preprocess
# -------------------------
wine = fetch_ucirepo(id=109)

X_df = wine.data.features.copy()
y_df = wine.data.targets.copy()

y = y_df.iloc[:, 0].astype(str).values
X = X_df.values.astype(np.float32)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X).astype(np.float32)

le = LabelEncoder()
y_enc = le.fit_transform(y)

X_t = torch.from_numpy(X_scaled)


# -------------------------
# PCA
# -------------------------
pca = PCA(n_components=2, random_state=42)
Z_pca = pca.fit_transform(X_scaled)


# -------------------------
# Kernel PCA
# -------------------------
# RBF is the most common kernel for nonlinear structure.
# gamma=None lets sklearn pick 1 / n_features by default.
kpca = KernelPCA(
    n_components=2,
    kernel="rbf",
    gamma=None,
    fit_inverse_transform=False,
    eigen_solver="auto",
    random_state=42
)
Z_kpca = kpca.fit_transform(X_scaled)


# -------------------------
# UMAP
# -------------------------
Z_umap = umap.UMAP(
    n_components=2,
    n_neighbors=15,
    min_dist=0.1,
    random_state=42
).fit_transform(X_scaled)


# -------------------------
# t-SNE
# -------------------------
Z_tsne = TSNE(
    n_components=2,
    perplexity=30,
    init="pca",
    learning_rate="auto",
    random_state=42
).fit_transform(X_scaled)


# -------------------------
# PyTorch Denoising Autoencoder (val + early stopping + BN + dropout + weight decay)
# -------------------------
class DenoisingAE(nn.Module):
    def __init__(self, input_dim, latent_dim=2, p_drop=0.2):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(p_drop),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(p_drop),

            nn.Linear(64, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim),
        )

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat


def train_dae_get_latents(
    X_t,
    latent_dim=2,
    batch_size=32,
    lr=1e-3,
    weight_decay=1e-4,
    noise_std=0.05,
    max_epochs=2000,
    patience=60,
    seed=42,
):
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds = TensorDataset(X_t)

    n = len(ds)
    n_val = max(1, int(0.2 * n))
    n_train = n - n_val
    train_ds, val_ds = random_split(
        ds,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(seed)
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=False)

    model = DenoisingAE(input_dim=X_t.shape[1], latent_dim=latent_dim, p_drop=0.2).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()

    best_val = float("inf")
    best_state = None
    bad_epochs = 0

    for epoch in range(1, max_epochs + 1):
        model.train()
        for (xb,) in train_loader:
            xb = xb.to(device)

            if noise_std > 0:
                xb_in = xb + noise_std * torch.randn_like(xb)
            else:
                xb_in = xb

            opt.zero_grad()
            xb_hat = model(xb_in)
            loss = loss_fn(xb_hat, xb)
            loss.backward()
            opt.step()

        model.eval()
        val_loss = 0.0
        count = 0
        with torch.no_grad():
            for (xb,) in val_loader:
                xb = xb.to(device)
                xb_hat = model(xb)
                val_loss += loss_fn(xb_hat, xb).item() * xb.size(0)
                count += xb.size(0)
        val_loss /= count

        if val_loss < best_val - 1e-6:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1

        if epoch % 100 == 0 or epoch == 1:
            print(f"epoch {epoch:4d} | val loss {val_loss:.6f}")

        if bad_epochs >= patience:
            print(f"Early stopping at epoch {epoch} (best val loss {best_val:.6f})")
            break

    model.load_state_dict(best_state)
    model.eval()

    with torch.no_grad():
        Z = model.encoder(X_t.to(device)).cpu().numpy()

    return Z


Z_ae = train_dae_get_latents(
    X_t,
    latent_dim=2,
    batch_size=32,
    lr=1e-3,
    weight_decay=1e-4,
    noise_std=0.05,
    max_epochs=2000,
    patience=60,
    seed=42
)


# -------------------------
# Plot (side-by-side)
# -------------------------
fig, axes = plt.subplots(1, 5, figsize=(24, 5))

sc0 = axes[0].scatter(Z_pca[:, 0], Z_pca[:, 1], c=y_enc, s=20, alpha=0.85)
axes[0].set_title(f"PCA (2D)\nExplained var = {pca.explained_variance_ratio_.sum():.2f}")
axes[0].set_xlabel("PC1")
axes[0].set_ylabel("PC2")

sc1 = axes[1].scatter(Z_kpca[:, 0], Z_kpca[:, 1], c=y_enc, s=20, alpha=0.85)
axes[1].set_title("Kernel PCA (RBF, 2D)")
axes[1].set_xlabel("KPCA 1")
axes[1].set_ylabel("KPCA 2")

sc2 = axes[2].scatter(Z_umap[:, 0], Z_umap[:, 1], c=y_enc, s=20, alpha=0.85)
axes[2].set_title("UMAP (2D)")
axes[2].set_xlabel("Dim 1")
axes[2].set_ylabel("Dim 2")

sc3 = axes[3].scatter(Z_tsne[:, 0], Z_tsne[:, 1], c=y_enc, s=20, alpha=0.85)
axes[3].set_title("t-SNE (2D)")
axes[3].set_xlabel("Dim 1")
axes[3].set_ylabel("Dim 2")

sc4 = axes[4].scatter(Z_ae[:, 0], Z_ae[:, 1], c=y_enc, s=20, alpha=0.85)
axes[4].set_title("Denoising AE (2D)\nval + early stopping")
axes[4].set_xlabel("Latent 1")
axes[4].set_ylabel("Latent 2")

handles, _ = sc0.legend_elements()
labels = le.inverse_transform(np.arange(len(handles)))
fig.legend(handles, labels, title="target", loc="center right")

plt.tight_layout(rect=[0, 0, 0.93, 1])
plt.show()
