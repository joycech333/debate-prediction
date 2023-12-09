import numpy as np
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import pickle
import os
import re

files = [f.name for f in os.scandir("congressional_speeches")]

'''
Takes in a single line of text, returns tokenized vector of cleaned words from line.
Remove punctuations; numbers; English stopwords; tokenize and lowercase.
'''
def preprocess_text(line):
    cleaned_line = re.sub(r'[^\w\s]', '', line)
    cleaned_line = re.sub(r'\d+', '', cleaned_line) 
    tokens = word_tokenize(cleaned_line.lower())
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word not in stop_words]
    return filtered_tokens

# We want speech level splits: split by | delimiter
def tokenized_small_speeches(files):
    tokenized_speeches = []
    for file_path in files[:3]:
        print(f"reading {file_path}")
        file = open(f'congressional_speeches/{file_path}', 'r', encoding='utf-8', errors='ignore').read()
        speeches = file.split('|')
        for speech in speeches: 
            processed_speech = preprocess_text(speech)
            if len(processed_speech) > 4:
                tokenized_speeches.append(processed_speech)
    return tokenized_speeches

congress_tokens = tokenized_small_speeches(files)

with open('ctokens_1.pickle', 'wb') as file:
    pickle.dump(congress_tokens, file)

with open('ctokens_1.pickle', 'rb') as file:
    congress_tokens = pickle.load(file)
print(congress_tokens)
