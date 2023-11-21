import util
import numpy as np
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import re

file_path = 'data/pres/10_07_2008.txt'
speaker_lines = util.split_speakers(file_path)

# Takes in a single line of text, returns tokenized vector of cleaned words from line
def preprocess_text(line):
    # Remove punctuation
    cleaned_line = re.sub(r'[^\w\s]', '', line)
    # Remove numbers
    cleaned_line = re.sub(r'\d+', '', cleaned_line) 
    # Tokenize and lowercase
    tokens = word_tokenize(line.lower())
    # Remove English stopwords
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word not in stop_words]
    return filtered_tokens

# FIX THIS: gets all individual speaker lines from speaker_lines dictionary. 
total_lines = []
for speaker in speaker_lines:
    for line in speaker_lines[speaker]:
        total_lines.append(line)

# Preprocess all the lines of text in the debate
processed_data = [preprocess_text(line) for line in total_lines]

# Train the Word2Vec model with skip-grams
model = Word2Vec(sentences=processed_data, vector_size=100, window=2, sg=1, epochs=50)

# Accessing word vectors
word_vectors = model.wv

# Get similar words to given target words (TEST THINGS HERE)
target_words = ['economy', 'future', 'americans', 'great', 'countries']

# Find similar words for each target word
for word in target_words:
    similar_words = model.wv.most_similar(word, topn=5)
    print(f"Words similar to '{word}': {similar_words}")
