# mini_sgns.py
import re, math, random
from collections import Counter
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# ---- Reproducibility
seed = 7
random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

# ---- Tiny but structured corpus (repeat patterns for stronger signal)
corpus = """
the king ruled the kingdom and the queen ruled the castle
the queen and the king were wise and kind
royalty like king queen prince and princess live in a castle

the man worked in the field while the woman cared for the children
the woman spoke to the man and the child played with a toy

apples and bananas are fruit an apple and a banana are tasty fruit
fruit like apple banana orange and mango are sweet fruit

a dog chased a cat the cats climbed while the dogs barked animals run
paris is the capital of france london is the capital of the uk
the river runs by the road near the bank of the river
"""

def tokenize(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return [t for t in text.split() if t]

tokens = tokenize(corpus)

# ---- Vocabulary
counts = Counter(tokens)
vocab = sorted(counts.keys())
word2idx = {w:i for i,w in enumerate(vocab)}
idx2word = {i:w for w,i in word2idx.items()}
V = len(vocab)

# ---- Skip-gram pairs (center → context)
window = 2
pairs = []
for i, w in enumerate(tokens):
    c = word2idx[w]
    for j in range(max(0, i-window), min(len(tokens), i+window+1)):
        if j == i: continue
        pairs.append((c, word2idx[tokens[j]]))

# ---- Negative sampling distribution (unigram^0.75)
freq = np.array([counts[idx2word[i]] for i in range(V)], dtype=np.float64)
q = freq ** 0.75; q /= q.sum(); cum = np.cumsum(q)
def sample_neg(batch_size, K):
    u = np.random.rand(batch_size*K)
    return np.searchsorted(cum, u).reshape(batch_size, K)

# ---- SGNS model
class SGNS(nn.Module):
    def __init__(self, V, D):
        super().__init__()
        self.in_embed = nn.Embedding(V, D)
        self.out_embed = nn.Embedding(V, D)
        init_range = 0.5 / D
        with torch.no_grad():
            self.in_embed.weight.uniform_(-init_range, init_range)
            self.out_embed.weight.zero_()
    def forward(self, center, pos, neg):
        v = self.in_embed(center)                           # [B, D]
        u_pos = self.out_embed(pos)                         # [B, D]
        u_neg = self.out_embed(neg)                         # [B, K, D]
        pos_score = torch.sum(v * u_pos, dim=1)             # [B]
        neg_score = torch.bmm(u_neg, v.unsqueeze(2)).squeeze(2)  # [B, K]
        loss = -(torch.log(torch.sigmoid(pos_score) + 1e-9).mean()
                 + torch.log(torch.sigmoid(-neg_score) + 1e-9).mean())
        return loss

# ---- Train
device = torch.device("cpu")
D = 50
model = SGNS(V, D).to(device)
opt = optim.Adam(model.parameters(), lr=3e-3)

pairs_np = np.array(pairs, dtype=np.int64)
B, K, epochs = 256, 5, 12
for ep in range(epochs):
    np.random.shuffle(pairs_np)
    tot = 0.0
    for i in range(0, len(pairs_np), B):
        batch = pairs_np[i:i+B]
        center = torch.tensor(batch[:,0], dtype=torch.long, device=device)
        pos    = torch.tensor(batch[:,1], dtype=torch.long, device=device)
        neg    = torch.tensor(sample_neg(len(batch), K), dtype=torch.long, device=device)
        opt.zero_grad(set_to_none=True)
        loss = model(center, pos, neg)
        loss.backward(); opt.step()
        tot += loss.item() * len(batch)
    print(f"epoch {ep+1}/{epochs}  avg loss={tot/len(pairs_np):.4f}")

# ---- Inspect: nearest neighbors by cosine similarity
def embedding_matrix(m):
    with torch.no_grad():
        return m.in_embed.weight.detach().cpu().numpy()

def cosine_neighbors(word, mat, k=6):
    if word not in word2idx: return []
    idx = word2idx[word]
    X = mat / (np.linalg.norm(mat, axis=1, keepdims=True) + 1e-9)
    sims = X @ X[idx]
    order = np.argsort(-sims)[:k+1]
    return [(idx2word[i], float(sims[i])) for i in order if i != idx]

E = embedding_matrix(model)
probe = ["king","queen","man","woman","apple","banana","fruit",
         "paris","france","dog","cat","castle","capital"]
print("\nNearest neighbors (cosine sim):\n")
for w in probe:
    print(f"{w:>8} → " + ", ".join([f"{nw}({sim:.2f})" for nw,sim in cosine_neighbors(w, E, k=6)]))

# ---- 2-D PCA visualization of selected words
selected = ["king","queen","prince","princess","castle",
            "apple","banana","orange","mango","fruit",
            "paris","france","london","uk","capital",
            "dog","cat","dogs","cats","animals"]
sel_idx = [word2idx[w] for w in selected if w in word2idx]
labels  = [idx2word[i] for i in sel_idx]
Z = PCA(2).fit_transform(E[sel_idx])

plt.figure(figsize=(6,5))
plt.scatter(Z[:,0], Z[:,1])
for (x,y,t) in zip(Z[:,0], Z[:,1], labels):
    plt.text(x+0.03, y+0.03, t)
plt.title("Word embeddings (PCA to 2-D)")
plt.xlabel("PC1"); plt.ylabel("PC2")
plt.tight_layout()
plt.show()

