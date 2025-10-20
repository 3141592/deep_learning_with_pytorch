import torch
from pprint import pprint

print("# shape [channels, rows, columns]")
img_t = torch.randn(3, 5, 5)
pprint(f"img_t.shape: {img_t.shape}")
pprint(f"img_t[0]: {img_t[0]}")

weights = torch.tensor([0.2126, 0.7152, 0.0722])
pprint(f"weights.shape: {weights.shape}")
pprint(f"weights[0]: {weights[0]}")

print("# shape [batch, channels, rows, columns]")
batch_t = torch.randn(2, 3, 5, 5)
pprint(f"batch_t.shape: {batch_t.shape}")
pprint(f"batch_t[0]: {batch_t[0]}")

img_gray_naive = img_t.mean(-3)
pprint(f"img_gray_naive.shape: {img_gray_naive.shape}")
pprint(f"img_gray_naive[0]: {img_gray_naive[0]}")

batch_gray_naive = batch_t.mean(-3)
pprint(f"batch_gray_naive.shape: {batch_gray_naive.shape}")
pprint(f"batch_gray_naive[0]: {batch_gray_naive[0]}")

img_named = img_t.refine_names(..., 'channels', 'rows', 'columns')
pprint(f"img_named.shape: {img_named.shape}")
pprint(f"img_named[0]: {img_named[0]}")
pprint(f"img_named[0][0]: {img_named[0][0]}")

batch_named = img_t.refine_names(..., 'channels', 'rows', 'columns')
pprint(f"batch_named.shape: {batch_named.shape}")
pprint(f"batch_named[0]: {batch_named[0]}")
pprint(f"batch_named[0][0]: {img_named[0][0]}")

