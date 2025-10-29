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

pprint("# 4.5.2 One-hot-encoding characters")
lines = text.split('\n')
line = lines[200]
pprint(line)

pprint("# 4.5.3 One-hot encoding whole words")
words_in_line = clean_words(line)
pprint(f"words_in_line: {words_in_line}")

pprint("Mapping of words to indexes")
word_list = sorted(set(clean_words(text)))
pprint(f"word_list: {word_list}")


