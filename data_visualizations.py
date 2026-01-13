# Side-by-side 2D embeddings: UMAP vs t-SNE vs PyTorch Autoencoder
# - Excludes target from embeddings
# - Colors points by target
#
# pip install ucimlrepo umap-learn torch scikit-learn matplotlib

import numpy as np
import matplotlib.pyplot as plt
from ucimlrepo import fetch_ucirepo

import umap
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.manifold import TSNE

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


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
# PyTorch Autoencoder (2D latent)
# -------------------------
class AutoEncoder(nn.Module):
    def __init__(self, input_dim, latent_dim=2):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim),
        )

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AutoEncoder(input_dim=X_scaled.shape[1], latent_dim=2).to(device)

loader = DataLoader(TensorDataset(X_t), batch_size=32, shuffle=True)
opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
loss_fn = nn.MSELoss()

epochs = 300
for epoch in range(epochs):
    model.train()
    for (xb,) in loader:
        xb = xb.to(device)
        opt.zero_grad()
        loss = loss_fn(model(xb), xb)
        loss.backward()
        opt.step()

model.eval()
with torch.no_grad():
    Z_ae = model.encoder(X_t.to(device)).cpu().numpy()


# -------------------------
# Plot (side-by-side)
# -------------------------
fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))

sc0 = axes[0].scatter(Z_umap[:, 0], Z_umap[:, 1], c=y_enc, s=20, alpha=0.85)
axes[0].set_title("UMAP (2D)")
axes[0].set_xlabel("Dim 1")
axes[0].set_ylabel("Dim 2")

sc1 = axes[1].scatter(Z_tsne[:, 0], Z_tsne[:, 1], c=y_enc, s=20, alpha=0.85)
axes[1].set_title("t-SNE (2D)")
axes[1].set_xlabel("Dim 1")
axes[1].set_ylabel("Dim 2")

sc2 = axes[2].scatter(Z_ae[:, 0], Z_ae[:, 1], c=y_enc, s=20, alpha=0.85)
axes[2].set_title("PyTorch Autoencoder (2D latent)")
axes[2].set_xlabel("Latent 1")
axes[2].set_ylabel("Latent 2")

# One shared legend (based on same classes)
handles, _ = sc0.legend_elements()
labels = le.inverse_transform(np.arange(len(handles)))
fig.legend(handles, labels, title="target", loc="center right")

plt.tight_layout(rect=[0, 0, 0.9, 1])
plt.show()
