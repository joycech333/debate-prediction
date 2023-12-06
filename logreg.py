"""
bag of words with logreg
"""

import util
import json
import pandas as pd
import sklearn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import train_test_split
import re
import nltk
import os
import numpy as np
# nltk.download('averaged_perceptron_tagger')
# nltk.download('universal_tagset')

def pos_tokenizer(text):
    tokens = nltk.word_tokenize(text)
    pos = nltk.pos_tag(tokens)
    return [x[1] for x in pos]


def pos_tokenizer_uni(text):
    tokens = nltk.word_tokenize(text)
    pos = nltk.pos_tag(tokens, tagset='universal')
    return [x[1] for x in pos]


# only keep words with specific pos
def filter_pos_tokenizer(lines, pos):
    all_filtered = []
    for line in lines:
        tokens = nltk.word_tokenize(line)
        pos = nltk.pos_tag(tokens, tagset='universal')
        filtered = [x[0] for x in pos if x[1] == pos]
        all_filtered.extend(filtered)
    return all_filtered


def pronoun_tokenizer(text):
    pronouns = "|".join(util.PRONOUNS)
    # print(pronouns)
    # print(re.findall(f"\b({pronouns})\b", text))
    # return text
    return re.findall(f"{pronouns}", text)


# one row per speaker per debate
def generate_lines_scores_speaker(filename, allWinners):
    winner = allWinners[filename]['win'].upper()
    loser = allWinners[filename]['lose'].upper()
    draw = allWinners[filename]['draw']

    # ignore files with no ground truth
    if not winner:
        return [], []
    
    participants = util.PARTICIPANTS[filename]
    speaker_lines = util.split_speakers(f'scraped-data/transcripts/{filename}', True)

    all_lines = []
    ys = []
    for speaker in speaker_lines:
        lines = speaker_lines[speaker]

        # if candidate (not a moderator)
        if speaker.upper() in participants:
            all_lines.append('\n'.join(lines))
            # account for ties
            ys.append(speaker == winner or (draw and speaker == loser))
    
    return all_lines, ys


def generate_lines_scores(filename, allWinners):
    winner = allWinners[filename]['win'].upper()
    loser = allWinners[filename]['lose'].upper()
    draw = allWinners[filename]['draw']

    # ignore files with no ground truth
    if not winner:
        return [], []
    
    participants = util.PARTICIPANTS[filename]
    speaker_lines = util.split_speakers(f'scraped-data/transcripts/{filename}', True)

    all_lines = []
    ys = []
    for speaker in speaker_lines:
        lines = speaker_lines[speaker]

        if speaker.upper() in participants:
            all_lines.extend(lines)
            # account for ties
            ys.extend([speaker == winner or (draw and speaker == loser)] * len(lines))
    
    return all_lines, ys

def generate_pronouns_y_speaker(files, winners):
    all_Xs = []
    all_ys = []
    for file_path in files:
        all_lines, ys = generate_lines_scores_speaker(file_path, winners)
        all_Xs.extend(all_lines)
        all_ys.extend(ys)
        
    count_vect = CountVectorizer(tokenizer=pronoun_tokenizer, token_pattern=None)
    count_matrix = count_vect.fit_transform(all_Xs)
    count_array = count_matrix.toarray()
    X = pd.DataFrame(data=count_array, columns=count_vect.get_feature_names_out())
    sums = X.sum(axis=1).replace(0, 1) # no zeros
    normalizedX = X.div(sums, axis=0) # mean across each row (each input line)
    y = pd.DataFrame(all_ys)

    return normalizedX, y


def generate_pronouns_y(files, winners):
    all_Xs = []
    all_ys = []
    for file_path in files:
        all_lines, ys = generate_lines_scores(file_path, winners)
        all_Xs.extend(all_lines)
        all_ys.extend(ys)
        
    count_vect = CountVectorizer(tokenizer=pronoun_tokenizer, token_pattern=None)
    count_matrix = count_vect.fit_transform(all_Xs)
    count_array = count_matrix.toarray()
    X = pd.DataFrame(data=count_array, columns=count_vect.get_feature_names_out())
    sums = X.sum(axis=1).replace(0, 1) # no zeros
    normalizedX = X.div(sums, axis=0) # mean across each row (each input line)
    y = pd.DataFrame(all_ys)

    return normalizedX, y
    

# generate parts of speech vector
def generate_pos_y_speaker(files, winners):
    all_Xs = []
    all_ys = []
    for file_path in files:
        lines, ys = generate_lines_scores_speaker(file_path, winners)
        all_Xs.extend(lines)
        all_ys.extend(ys)
        
    count_vect = CountVectorizer(tokenizer=pos_tokenizer, token_pattern=None)
    count_matrix = count_vect.fit_transform(all_Xs)
    count_array = count_matrix.toarray()
    X = pd.DataFrame(data=count_array, columns=count_vect.get_feature_names_out())
    sums = X.sum(axis=1).replace(0, 1) # no zeros
    X = X.div(sums, axis=0) # mean across each row (each input line)
    y = pd.DataFrame(all_ys)

    return X, y


# generate parts of speech vector
def generate_pos_y(files, winners):
    all_Xs = []
    all_ys = []
    for file_path in files:
        all_lines, ys = generate_lines_scores(file_path, winners)
        all_Xs.extend(all_lines)
        all_ys.extend(ys)
        
    count_vect = CountVectorizer(tokenizer=pos_tokenizer, token_pattern=None)
    count_matrix = count_vect.fit_transform(all_Xs)
    count_array = count_matrix.toarray()
    X = pd.DataFrame(data=count_array, columns=count_vect.get_feature_names_out())
    sums = (X.sum(axis=1)).replace(0, 1) # no zeros
    X = X.div(sums, axis=0) # mean across each row (each input line)
    y = pd.DataFrame(all_ys)

    return X, y


# generate parts of speech vector
def generate_pos_uni_y_speaker(files, winners):
    all_Xs = []
    all_ys = []
    for file_path in files:
        all_lines, ys = generate_lines_scores_speaker(file_path, winners)
        all_Xs.extend(all_lines)
        all_ys.extend(ys)
        
    count_vect = CountVectorizer(tokenizer=pos_tokenizer_uni, token_pattern=None)
    count_matrix = count_vect.fit_transform(all_Xs)
    count_array = count_matrix.toarray()
    X = pd.DataFrame(data=count_array, columns=count_vect.get_feature_names_out())
    sums = X.sum(axis=1).replace(0, 1) # no zeros
    X = X.div(sums, axis=0) # mean across each row (each input line)
    y = pd.DataFrame(all_ys)

    return X, y

# generate parts of speech vector
def generate_pos_uni_y(files, winners):
    all_Xs = []
    all_ys = []
    for file_path in files:
        all_lines, ys = generate_lines_scores(file_path, winners)
        all_Xs.extend(all_lines)
        all_ys.extend(ys)
        
    count_vect = CountVectorizer(tokenizer=pos_tokenizer_uni, token_pattern=None)
    count_matrix = count_vect.fit_transform(all_Xs)
    count_array = count_matrix.toarray()
    X = pd.DataFrame(data=count_array, columns=count_vect.get_feature_names_out())
    sums = X.sum(axis=1).replace(0, 1) # no zeros
    X = X.div(sums, axis=0) # mean across each row (each input line)
    y = pd.DataFrame(all_ys)

    return X, y


# make this not just N
def filter_pos(file, posCode):
    text = open(f'scraped-data/transcripts/{file}').read()
    tokens = nltk.word_tokenize(text.lower())
    pos_tags = nltk.pos_tag(tokens)
    return [word for word, pos in pos_tags if word.upper() not in util.PARTICIPANTS and (pos[0] == posCode)] #  or pos[0] == 'P'


# generate X, y with just the top 5 of a certain pos (ex: t5 nouns)
def generate_X_y_pos_5_speaker(files, winners, posCode):
    filtered_pos = []
    all_ys = []
    all_files = []
    for file in files:
        all_lines, ys = generate_lines_scores_speaker(file, winners)
        all_files.extend(all_lines)
        all_ys.extend(ys)

        filtered_pos.extend(filter_pos(file, posCode))
        
    # print(filtered_pos[:20])
    t5tuples = nltk.FreqDist(filtered_pos).most_common(5)
    t5 = [x[0] for x in t5tuples]
    count_vect = CountVectorizer(vocabulary=t5)
    count_matrix = count_vect.fit_transform(all_files)
    count_array = count_matrix.toarray()
    X = pd.DataFrame(data=count_array, columns=count_vect.get_feature_names_out())
    # print(X[:20])
    sums = X.sum(axis=1).replace(0, 1) # no zeros
    X = X.div(sums, axis=0) # mean across each row (each input line)
    y = pd.DataFrame(all_ys)

    return X, y


# generate X, y with just the top 5 of a certain pos (ex: t5 nouns)
def generate_X_y_pos_5(files, winners, posCode):
    filtered_pos = []
    all_ys = []
    all_files = []
    for file in files:
        all_lines, ys = generate_lines_scores(file, winners)
        all_files.extend(all_lines)
        all_ys.extend(ys)

        filtered_pos.extend(filter_pos(file, posCode))
        
    # print(filtered_pos[:20])
    t5tuples = nltk.FreqDist(filtered_pos).most_common(5)
    t5 = [x[0] for x in t5tuples]
    count_vect = CountVectorizer(vocabulary=t5)
    count_matrix = count_vect.fit_transform(all_files)
    count_array = count_matrix.toarray()
    X = pd.DataFrame(data=count_array, columns=count_vect.get_feature_names_out())
    # print(X[:20])
    sums = X.sum(axis=1).replace(0, 1) # no zeros
    X = X.div(sums, axis=0) # mean across each row (each input line)
    y = pd.DataFrame(all_ys)

    return X, y


def generate_X_y(files, winners):
    all_files = []
    all_ys = []
    for file in files:
        # print(file)
        all_lines, ys = generate_lines_scores(file, winners)
        all_files.extend(all_lines)
        all_ys.extend(ys)
        
    count_vect = CountVectorizer(stop_words='english')
    count_matrix = count_vect.fit_transform(all_files)
    count_array = count_matrix.toarray()
    X = pd.DataFrame(data=count_array, columns=count_vect.get_feature_names_out())
    sums = X.sum(axis=1).replace(0, 1) # no zeros
    X = X.div(sums, axis=0) # mean across each row (each input line)
    y = pd.DataFrame(all_ys)

    return X, y


# generate Xs and ys, one row per speaker per debate
def generate_X_y_speaker(files, winners):
    all_Xs = []
    all_ys = []
    for file in files:
        all_lines, ys = generate_lines_scores_speaker(file, winners)
        all_Xs.extend(all_lines)
        all_ys.extend(ys)
        
    count_vect = CountVectorizer(stop_words='english')
    count_matrix = count_vect.fit_transform(all_Xs)
    count_array = count_matrix.toarray()
    X = pd.DataFrame(data=count_array, columns=count_vect.get_feature_names_out())
    sums = X.sum(axis=1).replace(0, 1) # no zeros
    X = X.div(sums, axis=0) # mean across each row (each input line)
    y = pd.DataFrame(all_ys)

    return X, y


# generate Xs and ys, one row per speaker per debate
def generate_text_speaker(files):
    all_Xs = []
    for file in files:
        all_lines, ys = generate_lines_scores_speaker(file, winners)
        all_Xs.extend(all_lines)
    
    X = pd.DataFrame(data=all_Xs)

    return X


def logreg(X, y):
    # split data
    # x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0) # , random_state=0

    # fit regression
    logisticRegr = LogisticRegressionCV(cv=10, random_state=0, max_iter=1000) # solver='saga', C=10.0
    # logisticRegr.fit(x_train, y_train.values.ravel())
    logisticRegr.fit(X, y.values.ravel())

    # predict
    # predictions = logisticRegr.predict(x_test)
    # print(logisticRegr.coef_)
    # print('sum of coeffs:', np.sum(np.abs(logisticRegr.coef_), axis=1))
    # score = logisticRegr.score(x_test, y_test)
    # print('logreg test score:', score)
    # print("Optimal C:", logisticRegr.C_)
    means = np.mean(logisticRegr.scores_[1], axis=0)
    print("Mean accuracy on 10-fold CV:", means)
    print("Average accuracy:", np.mean(means))


if __name__ == "__main__":
    with open('scraped-data/ground_truths.json') as json_file:
        winners = json.load(json_file)

    # files = ['data/pres/09_26_2008.txt', 'data/pres/10_07_2008.txt', 'data/pres/10_15_2008.txt', 'data/vp/10_02_2008.txt']
    files = [f.name for f in os.scandir("scraped-data/transcripts")]

    # y = pd.read_csv('xys/y_per_speaker.csv')
    # print(1 - y.mean())
    # print('All Logreg by Line')
    # X, y = generate_X_y(files, winners)
    # print('X')
    # print(X.info())
    # print('y')
    # print(y.info())
    # logreg(X, y)

    # X = generate_text_speaker(files)
    # X = pd.read_csv('xys/all_x_text_per_speaker.csv')
    # print(X.info())

    # print('\nBy Speaker per Debate')
    # # X.to_csv('xys/all_x_per_speaker.csv', index=False)
    # X = pd.read_csv('xys/all_x_per_speaker.csv')
    # y = pd.read_csv('xys/y_per_speaker.csv')
    # # y.to_csv('xys/y_per_speaker.csv', index=False)
    # # print('X')
    # # print(X.info())
    # # print('y')
    # # print(y.info())
    # # print('mean:', y.mean())
    # logreg(X, y)


    # print('Pronoun')
    # X, y = generate_pronouns_y_speaker(files, winners)
    # # X.to_csv('xys/pronouns_x_per_speaker.csv', index=False)
    # # y.to_csv('xys/y_per_speaker.csv', index=False)
    # X = pd.read_csv('xys/pronouns_x_per_speaker.csv')
    # y = pd.read_csv('xys/y_per_speaker.csv')
    # logreg(X, y)

    # print('\nReg on Specific PoS (ex: Singular Noun)')
    # # X, y = generate_pos_y_speaker(files, winners)
    # # X.to_csv('xys/pos_x_per_speaker.csv', index=False)
    # X = pd.read_csv('xys/pos_x_per_speaker.csv')
    # y = pd.read_csv('xys/y_per_speaker.csv')
    # # print(X[:20])
    # logreg(X, y)

    # print('\nReg on PoS Family (ex: Noun)')
    # # X, y = generate_pos_uni_y_speaker(files, winners)
    # # X.to_csv('xys/pos_fam_x_per_speaker.csv', index=False)
    # X = pd.read_csv('xys/pos_fam_x_per_speaker.csv')
    # y = pd.read_csv('xys/y_per_speaker.csv')
    # # print(X[:20])
    # logreg(X, y)

    # print('\nNoun')
    # # X, y = generate_X_y_pos_5_speaker(files, winners, 'N')
    # # X.to_csv('xys/t5nouns_x_per_speaker.csv', index=False)
    # X = pd.read_csv('xys/t5nouns_x_per_speaker.csv')
    # y = pd.read_csv('xys/y_per_speaker.csv')
    # # print(X[:20])
    # logreg(X, y)

    # print('\nVerb')
    # # X, y = generate_X_y_pos_5_speaker(files, winners, 'V')
    # # X.to_csv('xys/t5verbs_x_per_speaker.csv', index=False)
    # X = pd.read_csv('xys/t5verbs_x_per_speaker.csv')
    # y = pd.read_csv('xys/y_per_speaker.csv')
    # # print(X[:20])
    # logreg(X, y)

    # print('\nAdj')
    # # X, y = generate_X_y_pos_5_speaker(files, winners, 'J')
    # # X.to_csv('xys/t5adjs_x_per_speaker.csv', index=False)
    # X = pd.read_csv('xys/t5adjs_x_per_speaker.csv')
    # y = pd.read_csv('xys/y_per_speaker.csv')
    # # print(X[:20])
    # logreg(X, y)

    # print('\nAdv')
    # # X, y = generate_X_y_pos_5_speaker(files, winners, 'R')
    # # X.to_csv('xys/t5advs_x_per_speaker.csv', index=False)
    # X = pd.read_csv('xys/t5advs_x_per_speaker.csv')
    # y = pd.read_csv('xys/y_per_speaker.csv')
    # # print(X[:20])
    # logreg(X, y)
