import util
import numpy as np
import re
import json
from nltk import ngrams
import gensim.downloader as api
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfTransformer

files = ['data/pres/09_26_2008.txt', 'data/pres/10_07_2008.txt', 'data/pres/10_15_2008.txt', 'data/vp/10_02_2008.txt']

'''
Get labels from winners file. Since this is associated with each file (debate), 
we need to build the labels vector as we read each file from files.
Note that this code can be integrated with the function make_speakers_dict,
but is separate for readability.
'''
def get_winners(files):
    labels = []
    with open('data/ground_truths.json') as json_file:
        winners = json.load(json_file)

    for file_path in files:
        speakers_lines = util.split_speakers(file_path)
        for speaker in speakers_lines:
            # Janky way to fix the discrepancy between file names in the path and ground_truths.json
            winner = winners[file_path[-14:]]
            # If this speaker is the winner of the debate, represent as a 1 in labels vector
            if speaker.lower() == winner.lower():
                labels.append(1)
            else:
                labels.append(0)
    return np.array(labels)

get_winners(files)

'''
Remove punctuation, numbers, make lowercase
'''
def preprocess_text(line):
    cleaned_line = re.sub(r'[^\w\s]', '', line)
    cleaned_line = re.sub(r'\d+', '', cleaned_line) 
    cleaned_line = cleaned_line.lower()
    return cleaned_line

'''
Create a common vocabulary to standardize feature vector lengths
Concatenate all speaker lines across all files
'''
all_transcripts = []
for file_path in files:
    speakers_lines = util.split_speakers(file_path)
    for speaker, lines in speakers_lines.items():
        # Join all the lines for that particular speaker
        speaker_lines = ' '.join([preprocess_text(line) for line in lines])
        all_transcripts.append(speaker_lines)

# Initialize CountVectorizer for trigrams
ngram_range = (3, 3)  # Specifies trigrams
count_vectorizer = CountVectorizer(ngram_range=ngram_range)

# Create train/test split here from all_transcripts
labels = get_winners(files)
train_transcripts, test_transcripts, y_train, y_test = train_test_split(all_transcripts, labels, test_size=0.2, random_state=42)
X_train = count_vectorizer.fit_transform(train_transcripts).toarray()
X_test = count_vectorizer.transform(test_transcripts).toarray()

'''
Naive Bayes
'''
# def naivebayes(features, labels):

nb_classifier = GaussianNB()
nb_classifier.fit(X_train, y_train)

# Predict on the training set
predictions_train = nb_classifier.predict(X_train)
# Calculate accuracy for training set
accuracy_train = accuracy_score(y_train, predictions_train)
print("Training Accuracy:", accuracy_train)

# Predict on the test set
predictions_test = nb_classifier.predict(X_test)
# Calculate accuracy for test set
accuracy_test = accuracy_score(y_test, predictions_test)
print("Test Accuracy:", accuracy_test)
