import torch
from pprint import pprint

print("# shape [channels, rows, columns]")
img_t = torch.randn(3, 5, 5)
pprint(f"img_t: {img_t}")

weights = torch.tensor([0.2126, 0.7152, 0.0722])
pprint(f"weights: {weights}")

print("# shape [batch, channels, rows, columns]")
batch_t = torch.randn(2, 3, 5, 5)
pprint(f"batch_t: {batch_t}")



