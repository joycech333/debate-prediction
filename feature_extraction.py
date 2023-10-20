import nltk
# nltk.download('punkt')
from nltk.tokenize import word_tokenize
from nltk.text import Text
import numpy as np


def main():
    f = open("data/10_15_2008.txt").read().lower()
    tokens = word_tokenize(f)
    print(f'Tokens:\n{tokens[:10]}')
    bag_of_words = {word: tokens.count(word) for word in set(tokens)}
    print(f'Bag of words:\n{list(bag_of_words.items())[:10]}')


if __name__ == "__main__":
    main()