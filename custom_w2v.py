import util
import pickle
import nltk
import os
import re
from gensim.models import KeyedVectors
from gensim.models import Word2Vec
import numpy as np
import gensim.downloader as api
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
# Remember to run pip install xgboost
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold

# Sample debate transcripts (replace this with your actual debate transcripts)
files = [f.name for f in os.scandir("scraped-data/transcripts")]
model_path = "./custom_word2vec.model"

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

# We want speech level splits: use split_speakers
def tokenized_small_speeches(files):
    tokenized_input = []
    for file_path in files:
        speaker_dict = util.split_speakers(f'scraped-data/transcripts/{file_path}')
        for speaker, lines in speaker_dict.items():
            for line in lines: 
                processed_line = preprocess_text(line)
                if len(processed_line) > 4:
                    tokenized_input.append(processed_line)
    return tokenized_input

# Check if the model is already saved
if not os.path.exists(model_path):
    # Train the Word2Vec model. Min_count 5 to get rid of rare words
    model = Word2Vec(sentences=tokenized_small_speeches(files), vector_size=300, window=5, min_count=5, workers=4, epochs=10)
    # Save the loaded model
    model.save(model_path)
else:
    # Load the saved model
    model = Word2Vec.load(model_path)  

# debate_tokens = tokenized_small_speeches(files)
# with open('debate_tokens.pickle', 'wb') as file:
#     pickle.dump(debate_tokens, file)

'''
Takes in all debate files, returns a dictionary containing words spoken by each speaker. 
Keys = names of speakers.
Values = list of all tokenized & processed words from that person's speech, including duplicates
(tokenized concatenation of all words the speaker spoke). e.g. ['word1', 'word2', 'word3']
(Casual explanation: each speaker has one huge list of all the words they spoke)
'''
def make_speakers_dict(files):
    speakers_words = {}

    # Using integers as keys to allow duplicate speakers
    i = 0
    for file_path in files:
        participants = util.PARTICIPANTS[file_path]
        speakers_lines = util.split_speakers(f'scraped-data/transcripts/{file_path}', True)
        
        for speaker, lines in speakers_lines.items():
            if speaker.upper() in participants:
                word_tokens = []
                for line in lines:
                    word_tokens.extend(preprocess_text(line))
                speakers_words[i] = word_tokens
                i += 1

    return speakers_words
'''
Test make_speakers_dict
'''
# dict = make_speakers_dict(files)
# print(dict.keys())
# print(list(dict.values())[0])


'''
Input: dictionary of tokenized speakers' words.
Using the pre-trained Word2Vec model, get embeddings of every word for every speaker.
Takes mean of the word embeddings to calculate feature vector for that speaker.
Output: numpy array of feature vectors, one for each speaker. These are the Xs.
'''
def make_features(speakers_words):
    vocab = list(model.wv.index_to_key)
    
    features = []
    for speaker, tokens in speakers_words.items():
        speaker_vec = []
        for word in tokens:
            if word in vocab: 
                speaker_vec.append(model.wv[word])
        if speaker_vec:
            speaker_feature = np.mean(speaker_vec, axis=0)
            features.append(np.array(speaker_feature))
    return np.array(features)

'''
Test whether embed worked
'''
# print(make_features(make_speakers_dict(files)).shape)
# make_features(make_speakers_dict(files))

'''
Run models
'''
features = make_features(make_speakers_dict(files))
with open('word2vec_features.pickle', 'wb') as file:
    pickle.dump(features, file)

labels = np.array(util.ys_by_speaker())

# model = LogisticRegression()
model = XGBClassifier(n_estimators=100, random_state=42)
# model = RandomForestClassifier(n_estimators=100, random_state=42)

def cross_val_accuracy(features, labels, model):

    # Perform 10-fold cross-validation
    cv_scores = cross_val_score(model, features, labels.ravel(), cv=10)

    # Print accuracy for each fold
    print("Accuracy for each fold:", cv_scores)

    # Calculate mean accuracy across folds
    mean_accuracy = np.mean(cv_scores)
    std_dev = np.std(cv_scores)
    print(f"Mean Accuracy: {mean_accuracy}, Std Dev: {std_dev}")

cross_val_accuracy(features, labels, model)
