import nltk
# nltk.download('punkt')
from nltk.tokenize import word_tokenize
from nltk import FreqDist
import numpy as np


def main():
    f = open("data/10_15_2008.txt", mode='r',
             encoding='utf-8-sig').read().lower()
    # tokenize
    tokens = word_tokenize(f)
    print(f'Tokens:\n{tokens[:10]}')
    freq = FreqDist(tokens)
    print(freq.most_common(10))
    

if __name__ == "__main__":
    main()