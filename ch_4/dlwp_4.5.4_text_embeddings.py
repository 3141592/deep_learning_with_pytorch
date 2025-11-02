import torch
from pprint import pprint

def clean_words(input_str):
  # ctrl+shift+U 201D, 201C for directional double-quotes
  punctuation = '.,;:"!?“”_-'
  word_list = input_str.lower().replace('\n', ' ').split()
  word_list = [word.strip(punctuation) for word in word_list]
  return word_list

pprint("# 4.5.1 Converting text to numbers")
with open('../data/1342-0.txt', encoding='utf8') as f:
  text = f.read()

print()
pprint("# 4.5.2 One-hot-encoding characters")
lines = text.split('\n')
line = lines[200]
pprint(line)

print()
pprint("# 4.5.3 One-hot encoding whole words")
words_in_line = clean_words(line)
pprint(f"words_in_line: {words_in_line}")

pprint("Mapping of words to indexes")
word_list = sorted(set(clean_words(text)))
pprint(f"word_list[:15]: {word_list[:15]}")
pprint(f"word_list[:15]: {word_list[:15]}")

word2index_dict = {word: i for (i, word) in enumerate(word_list)}

# Use list comprehension to get the first 5 items in the dictionary and format them for pprint
first_items = {k: word2index_dict[k] for k in list(word2index_dict)[:15]}
pprint(f"First elements of word2index_dict: {first_items}")

pprint("One-hot encode sentence")
word_t = torch.zeros(len(words_in_line), len(word2index_dict))
for i, word in enumerate(words_in_line):
  word_index = word2index_dict[word]
  word_t[i][word_index] = 1
  print('{:2} {:4} {}'.format(i, word_index, word))

pprint(f"word_t.shape: {word_t.shape}")

print()
pprint("# 4.5.4 Text embeddings")




