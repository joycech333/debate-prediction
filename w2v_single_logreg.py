import util
import numpy as np
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import re
import json

file_path = 'data/pres/10_07_2008.txt'
# files = ['data/pres/09_26_2008.txt', 'data/pres/10_07_2008.txt', 'data/pres/10_15_2008.txt', 'data/vp/10_02_2008.txt']
speaker_lines = util.split_speakers(file_path)

# Takes in a single line of text, returns tokenized vector of cleaned words from line
def preprocess_text(line):
    # Remove punctuation; numbers; tokenize and lowercase; remove English stopwords
    cleaned_line = re.sub(r'[^\w\s]', '', line)
    cleaned_line = re.sub(r'\d+', '', cleaned_line) 
    tokens = word_tokenize(line.lower())
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word not in stop_words]
    return filtered_tokens

def create_word2vec_models(speaker_lines):
    # speaker_lines = {}

    # for file_path in files:
    #     speaker = util.split_speakers(file_path)
    #     if speaker not in speaker_lines:
    #         speaker_lines[speaker] = []
    #     speaker_lines[speaker].extend(util.split_speakers(file_path)[speaker])

    # Initialize a dictionary to hold Word2Vec models for each speaker
    speaker_models = {}

    # Get each speaker's total lines (list)
    for speaker, total_lines in speaker_lines.items():
        # Preprocess all the lines of text for this speaker
        processed_data = [preprocess_text(line) for line in total_lines]
        # Train the Word2Vec model with skip-grams
        model = Word2Vec(sentences=processed_data, vector_size=100, window=2, sg=1, epochs=50)
        # Store this speaker's Word2Vec model in the dictionary
        speaker_models[speaker] = model
    return speaker_models
    # # Accessing word vectors for each speaker
    # for speaker, model in speaker_models.items():
    #     word_vectors = model.wv 
    #     print(f"Word vectors for {speaker}: {word_vectors}")

'''
Logistic regression
'''
num_speakers = len(speaker_lines)
# Get speaker_matrices from word_to_vec
speaker_matrices = create_word2vec_models(speaker_lines)
# Make an array of feature vectors (compressed word2vec matrices)
embeddings = []
# Takes in matrix word2vec embeddings
for speaker, model in speaker_matrices.items():
    word_vectors = model.wv

    # Get all word vectors for this speaker
    all_vectors = [word_vectors[word] for word in word_vectors.key_to_index]

    # Create a feature vector by averaging all word vectors
    feature_vector = np.mean(all_vectors, axis=0)
    embeddings.append(feature_vector)

embeddings = np.array(embeddings)
# Get the winning speaker and losing speaker labels: FIX THIS HARD CODED RIGHT NOW
labels = np.array([0, 0, 1, 0])

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(embeddings, labels, test_size=0.2, random_state=42)

# Initialize and train logistic regression model
logistic_regression = LogisticRegression()
logistic_regression.fit(X_train, y_train)

# Evaluate the model
accuracy = logistic_regression.score(X_test, y_test)
print(f"Logistic Regression Accuracy: {accuracy}")
