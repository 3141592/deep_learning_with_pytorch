import torch, torch.nn as nn, torch.optim as optim

# 1. Toy vocab + data
pairs = [(0, 1), (1, 2), (2, 3)]  # (center, context) word indices

# 2. Model: 2 embedding tables (skip-gram)
class SGNS(nn.Module):
    def __init__(self, vocab, dim):
        super().__init__()
        self.in_embed = nn.Embedding(vocab, dim)
        self.out_embed = nn.Embedding(vocab, dim)
    def forward(self, c, pos, neg):
        v = self.in_embed(c)               # center
        u_pos = self.out_embed(pos)        # positive
        u_neg = self.out_embed(neg)        # negative
        pos_loss = -torch.log(torch.sigmoid((v*u_pos).sum(1))).mean()
        neg_loss = -torch.log(torch.sigmoid(-(v.unsqueeze(1)*u_neg).sum(2))).mean()
        return pos_loss + neg_loss

# 3. Train loop
model = SGNS(vocab=10, dim=8)
opt = optim.Adam(model.parameters(), lr=0.01)
for epoch in range(200):
    loss = model(
        torch.tensor([c for c,_ in pairs]),
        torch.tensor([p for _,p in pairs]),
        torch.randint(0,10,(len(pairs),4))
    )
    opt.zero_grad(); loss.backward(); opt.step()
print(model.in_embed.weight)

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

E = model.in_embed.weight.detach().numpy()
Z = PCA(2).fit_transform(E)
plt.scatter(Z[:,0], Z[:,1])
for i, (x, y) in enumerate(Z):
    plt.text(x+0.02, y+0.02, str(i))
plt.title("Word embeddings in 2-D PCA space")
plt.show()

