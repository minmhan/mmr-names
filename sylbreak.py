import os, sys, codecs
import fileinput
import re
import myparser

tokens = ['^', '#']
with open("data/processed-text/test.txt", "r", encoding="utf-8") as file:
    for line in file.readlines():
        p = myparser.MyParser()
        print(p.syllable("^" +
                         line.strip()))
        tokens.extend(p.syllable(line.strip()))


token_to_id = {k:v for v,k in enumerate(set(tokens))}
for k,v in token_to_id.items():
    print(k,v)




