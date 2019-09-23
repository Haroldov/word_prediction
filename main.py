#!/usr/bin/env python3
import numpy as np
from RNNmod import *
import random

with open('input_text.txt', 'r', encoding="utf-8") as f:
    txt = f.read()
print(type(txt))
txt = txt.replace("'s", '').replace("'", '')
sentences = txt.split('. ')
formattedSentences = []
words = []
for sentence in sentences[:-1]:
    tmpSen = sentence.replace(',', '').replace('.', '')
    formattedSentences.append(tmpSen.split(' '))
    words += sentence.replace(',', '').replace('.', '').split(" ")
formattedSentences.append(sentences[-1].replace('\n', '').replace('.', '').split(" "))
words += sentences[-1].replace('\n', '').replace(',', '').replace('.', '').split(" ")
words += ['\n']
words = list(set(words))
wrd2ix = {wrd:i for i, wrd in enumerate(sorted(words))}
ix2wrd = {i:wrd for i, wrd in enumerate(sorted(words))}

for s in formattedSentences:
    print(s)
parameters = model(formattedSentences, ix2wrd, wrd2ix,
                   dino_names=1, vocab_size=len(words))
while True:
    string = input("Enter a sentence: ")
    string = string.replace('\n', '').replace('.', '').split(" ")
    print()
    print(" ".join(string), end=" ")
    infer(string, parameters, wrd2ix, ix2wrd)
    print()
