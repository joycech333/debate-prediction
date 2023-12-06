import util
import os
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

files = [f.name for f in os.scandir("scraped-data/transcripts")]

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
    participants = util.PARTICIPANTS[file_path]
    speakers_lines = util.split_speakers(f'scraped-data/transcripts/{file_path}', True)
    for speaker, lines in speakers_lines.items():
        # Exclude non-participants
        if speaker.upper() in participants:
            # Join all the lines for that particular speaker
            speaker_lines = ' '.join([preprocess_text(line) for line in lines])
            all_transcripts.append(speaker_lines)
    
# Initialize CountVectorizer for trigrams
ngram_range = (3, 3)  # Specifies trigrams
count_vectorizer = CountVectorizer(ngram_range=ngram_range)

labels = util.ys_by_speaker()
X = count_vectorizer.fit_transform(all_transcripts).toarray()

'''
Naive Bayes
'''
nb_classifier = GaussianNB()

# Define the number of folds
num_folds = 10

# Perform 10-fold cross-validation and calculate accuracy
kfold = KFold(n_splits=num_folds, shuffle=True, random_state=42)
accuracy = cross_val_score(nb_classifier, X, np.ravel(labels), cv=kfold, scoring='accuracy')

# Print the accuracy for each fold and the mean accuracy
print("Accuracy for each fold:", accuracy)
print("Mean accuracy:", accuracy.mean())
