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
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
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

labels = get_winners(files)
X = count_vectorizer.fit_transform(all_transcripts).toarray()

'''
Naive Bayes
'''
# def naivebayes(features, labels):

nb_classifier = GaussianNB()

# Define the number of folds
num_folds = 10

# Perform 10-fold cross-validation and calculate accuracy
kfold = KFold(n_splits=num_folds, shuffle=True, random_state=42)
accuracy = cross_val_score(nb_classifier, X, labels, cv=kfold, scoring='accuracy')

# Print the accuracy for each fold and the mean accuracy
print("Accuracy for each fold:", accuracy)
print("Mean accuracy:", accuracy.mean())
