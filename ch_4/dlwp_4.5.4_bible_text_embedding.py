import re, math, random, os
from collections import Counter
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# ---------- charting helpers (matplotlib only) ----------
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# -------------------------
# Config
# -------------------------
TEXT_PATH   = "../data/erv.txt"          # path to your Bible text file
EMBED_DIM   = 200                # 50–300 is common; larger = more nuance
WINDOW      = 10                  # context window (symmetric)
MIN_COUNT   = 5                  # drop very rare tokens to shrink vocab
LR          = 3e-3               # learning rate
EPOCHS      = 20                  # raise for better quality
BATCH_SIZE  = 8192
NEG_K       = 5                  # negatives per positive
SEED        = 7

# Reproducibility (CPU-only here)
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# -------------------------
# Load + tokenize
# -------------------------
def load_text(path):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

def tokenize(text):
    # lower, remove verse numbers and punctuation-heavy stuff
    # (keeps letters and digits; tweak as desired)
    text = text.lower()
    text = re.sub(r"\[[^\]]*\]", " ", text)        # remove [brackets] if present
    text = re.sub(r"\d+:\d+", " ", text)           # remove verse refs like 3:16
    text = re.sub(r"[^a-z0-9\s\-']", " ", text)    # keep a-z, 0-9, hyphen, apostrophe
    toks = [t for t in text.split() if t]
    return toks

print(f"Loading {TEXT_PATH} ...")
raw = load_text(TEXT_PATH)
tokens = tokenize(raw)
print(f"Total tokens: {len(tokens):,}")

# -------------------------
# Build vocab (with min_count)
# -------------------------
counts = Counter(tokens)
vocab = [w for w,c in counts.items() if c >= MIN_COUNT]
vocab.sort()
word2idx = {w:i for i,w in enumerate(vocab)}
idx2word = {i:w for w,i in word2idx.items()}
V = len(vocab)
print(f"Vocab size (min_count={MIN_COUNT}): {V:,}")

# Filter tokens to vocab
tokens = [w for w in tokens if w in word2idx]

# -------------------------
# Make skip-gram (center, context) pairs
# -------------------------
pairs = []
for i, w in enumerate(tokens):
    c = word2idx[w]
    start = max(0, i - WINDOW)
    end   = min(len(tokens), i + WINDOW + 1)
    for j in range(start, end):
        if j == i: continue
        wj = tokens[j]
        pairs.append((c, word2idx[wj]))

pairs_np = np.array(pairs, dtype=np.int64)
print(f"Training pairs: {len(pairs_np):,}")

# -------------------------
# Negative sampling distribution (unigram^0.75)
# -------------------------
freq = np.array([counts[idx2word[i]] for i in range(V)], dtype=np.float64)
q = freq ** 0.75
q /= q.sum()
cum = np.cumsum(q)

def sample_neg(batch_size, K):
    u = np.random.rand(batch_size * K)
    return np.searchsorted(cum, u).reshape(batch_size, K)

# -------------------------
# SGNS model
# -------------------------
class SGNS(nn.Module):
    def __init__(self, V, D):
        super().__init__()
        self.in_embed  = nn.Embedding(V, D)
        self.out_embed = nn.Embedding(V, D)
        init_range = 0.5 / D
        with torch.no_grad():
            self.in_embed.weight.uniform_(-init_range, init_range)
            self.out_embed.weight.zero_()
    def forward(self, center, pos, neg):
        v = self.in_embed(center)                    # [B, D]
        u_pos = self.out_embed(pos)                  # [B, D]
        u_neg = self.out_embed(neg)                  # [B, K, D]
        pos_score = torch.sum(v * u_pos, dim=1)      # [B]
        neg_score = torch.bmm(u_neg, v.unsqueeze(2)).squeeze(2)  # [B, K]
        loss = -(torch.log(torch.sigmoid(pos_score) + 1e-9).mean()
                 + torch.log(torch.sigmoid(-neg_score) + 1e-9).mean())
        return loss

model = SGNS(V, EMBED_DIM).to(device)
opt = optim.Adam(model.parameters(), lr=LR)

# -------------------------
# Train
# -------------------------
num_batches = (len(pairs_np) + BATCH_SIZE - 1) // BATCH_SIZE
for ep in range(EPOCHS):
    np.random.shuffle(pairs_np)
    total = 0.0
    for b in range(num_batches):
        batch = pairs_np[b*BATCH_SIZE:(b+1)*BATCH_SIZE]
        if len(batch) == 0: continue
        center = torch.tensor(batch[:,0], dtype=torch.long, device=device)
        pos    = torch.tensor(batch[:,1], dtype=torch.long, device=device)
        neg    = torch.tensor(sample_neg(len(batch), NEG_K), dtype=torch.long, device=device)

        opt.zero_grad(set_to_none=True)
        loss = model(center, pos, neg)
        loss.backward(); opt.step()
        total += loss.item() * len(batch)
    print(f"epoch {ep+1}/{EPOCHS}  avg loss={total/len(pairs_np):.4f}")

# -------------------------
# Inspect: nearest neighbors
# -------------------------
def embedding_matrix(m):
    with torch.no_grad():
        return m.in_embed.weight.detach().cpu().numpy()

def cosine_neighbors(word, mat, k=10):
    if word not in word2idx: return []
    idx = word2idx[word]
    X = mat / (np.linalg.norm(mat, axis=1, keepdims=True) + 1e-9)
    sims = X @ X[idx]
    order = np.argsort(-sims)[:k+1]
    return [(idx2word[i], float(sims[i])) for i in order if i != idx]

E = embedding_matrix(model)

for w in ["god","lord","jesus","moses","david","israel","egypt","faith","love","sin","kingdom"]:
    if w in word2idx:
        nbrs = ", ".join([f"{nw}({sim:.2f})" for nw, sim in cosine_neighbors(w, E, k=8)])
        print(f"{w:>8} → {nbrs}")
    else:
        print(f"{w:>8} → <not in vocab (min_count={MIN_COUNT})>")

def _existing(words, word2idx):
    return [w for w in words if w in word2idx]

def plot_pca_words(selected_words, E, word2idx, title="Word embeddings (PCA to 2D)"):
    """Scatter-plot a PCA(2) projection of the selected words."""
    words = _existing(selected_words, word2idx)
    if not words:
        print("No selected words are in the vocabulary."); return
    idxs = [word2idx[w] for w in words]
    X = E[idxs]
    Z = PCA(2).fit_transform(X)

    plt.figure(figsize=(6,5))
    plt.scatter(Z[:,0], Z[:,1])
    for (x,y,w) in zip(Z[:,0], Z[:,1], words):
        plt.text(x+0.02, y+0.02, w)
    plt.title(title)
    plt.xlabel("PC1"); plt.ylabel("PC2")
    plt.tight_layout()
    plt.show()

def top_neighbors(anchor, E, word2idx, idx2word, k=12):
    """Return (word, cosine) pairs for the k nearest neighbors of anchor."""
    if anchor not in word2idx:
        print(f"'{anchor}' not in vocab."); return []
    X = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-9)
    a = word2idx[anchor]
    sims = X @ X[a]
    order = np.argsort(-sims)
    out = []
    for i in order:
        if i == a: continue
        out.append((idx2word[i], float(sims[i])))
        if len(out) >= k: break
    return out

def plot_neighbors_pca(anchor, E, word2idx, idx2word, k=12):
    """PCA(2) of anchor + its neighbors to see the mini-cluster."""
    nbrs = top_neighbors(anchor, E, word2idx, idx2word, k)
    if not nbrs: return
    words = [anchor] + [w for w,_ in nbrs]
    idxs  = [word2idx[w] for w in words]
    X = E[idxs]
    Z = PCA(2).fit_transform(X)

    plt.figure(figsize=(6,5))
    plt.scatter(Z[:,0], Z[:,1])
    for (x,y,w) in zip(Z[:,0], Z[:,1], words):
        plt.text(x+0.02, y+0.02, w)
    plt.title(f"Nearest neighbors of '{anchor}' (PCA to 2D)")
    plt.xlabel("PC1"); plt.ylabel("PC2")
    plt.tight_layout()
    plt.show()

def plot_neighbor_bars(anchor, E, word2idx, idx2word, k=12):
    """Simple bar chart of cosine similarity for the top-k neighbors."""
    nbrs = top_neighbors(anchor, E, word2idx, idx2word, k)
    if not nbrs: return
    words, sims = zip(*nbrs)
    y = np.arange(len(words))

    plt.figure(figsize=(7,4))
    plt.barh(y, sims)
    plt.yticks(y, words)
    plt.gca().invert_yaxis()
    plt.xlabel("cosine similarity")
    plt.title(f"Top-{k} neighbors of '{anchor}'")
    plt.tight_layout()
    plt.show()


# -------------------------
# Optional: save for later / TensorBoard Projector
# -------------------------
OUT_DIR = "bible_embeddings"
os.makedirs(OUT_DIR, exist_ok=True)
np.save(os.path.join(OUT_DIR, "embeddings.npy"), E)
with open(os.path.join(OUT_DIR, "vocab.txt"), "w", encoding="utf-8") as f:
    for w in vocab: f.write(w + "\n")
print(f"Saved embeddings to {OUT_DIR}/embeddings.npy and vocab to {OUT_DIR}/vocab.txt")

# A few Bible-ish themes
people    = ["god","lord","jesus","moses","david","abraham","isaac","jacob","joseph","paul","peter"]
places    = ["israel","jerusalem","egypt","babylon","zion","galilee","rome"]
concepts  = ["faith","love","grace","truth","sin","righteousness","law","covenant","kingdom","spirit"]

# PCA clusters (will skip words not in vocab)
plot_pca_words(people,   E, word2idx, title="People cluster (PCA)")
plot_pca_words(places,   E, word2idx, title="Places cluster (PCA)")
plot_pca_words(concepts, E, word2idx, title="Concepts cluster (PCA)")

# Anchor-driven neighbor views
plot_neighbors_pca("jesus", E, word2idx, idx2word, k=15)
plot_neighbor_bars("jesus", E, word2idx, idx2word, k=15)

plot_neighbors_pca("israel", E, word2idx, idx2word, k=15)
plot_neighbor_bars("israel", E, word2idx, idx2word, k=15)

plot_neighbors_pca("faith", E, word2idx, idx2word, k=15)
plot_neighbor_bars("faith", E, word2idx, idx2word, k=15)


