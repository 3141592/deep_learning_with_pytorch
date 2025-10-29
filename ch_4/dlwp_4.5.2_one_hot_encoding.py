import torch
from pprint import pprint

pprint("# 4.5.1 Converting text to numbers")
with open('../data/1342-0.txt', encoding='utf8') as f:
  text = f.read()

pprint("# 4.5.2 One-hot-encoding characters")
lines = text.split('\n')
line = lines[200]
pprint(line)

letter_t = torch.zeros(len(line), 128) # 128 hardcoded due to the limits of ASCII
pprint(f"letter_t.shape: {letter_t.shape}")
pprint(f"letter_t: {letter_t}")

pprint("The text uses directional double-quotes, which are not valid ASCII, so the screen them out here.")
for i, letter in enumerate(line.lower().strip()):
  letter_index = ord(letter) if ord(letter) < 128 else 0
  letter_t[i][letter_index] = 1

pprint(f"letter_t: {letter_t}")
pprint(f"letter_t[2]: {letter_t[2]}")

