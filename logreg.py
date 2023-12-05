"""
bag of words with logreg
"""

import util
import json
import pandas as pd
import sklearn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import re
import nltk
import os
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


def generate_pronouns_y(files, winners):
    for file_path in files:
        all_lines, ys = generate_lines_scores(file_path, winners)
        
    count_vect = CountVectorizer(tokenizer=pronoun_tokenizer, token_pattern=None)
    count_matrix = count_vect.fit_transform(all_lines)
    count_array = count_matrix.toarray()
    X = pd.DataFrame(data=count_array, columns=count_vect.get_feature_names_out())
    sums = X.sum(axis=1).replace(0, 1) # no zeros
    normalizedX = X.div(sums, axis=0) # mean across each row (each input line)
    y = pd.DataFrame(ys)

    return normalizedX, y
    

# generate parts of speech vector
def generate_pos_y(files, winners):
    for file_path in files:
        all_lines, ys = generate_lines_scores(file_path, winners)
        
    count_vect = CountVectorizer(tokenizer=pos_tokenizer, token_pattern=None)
    count_matrix = count_vect.fit_transform(all_lines)
    count_array = count_matrix.toarray()
    X = pd.DataFrame(data=count_array, columns=count_vect.get_feature_names_out())
    sums = X.sum(axis=1).replace(0, 1) # no zeros
    X = X.div(sums, axis=0) # mean across each row (each input line)
    y = pd.DataFrame(ys)

    return X, y


# generate parts of speech vector
def generate_pos_uni_y(files, winners):
    for file_path in files:
        all_lines, ys = generate_lines_scores(file_path, winners)
        
    count_vect = CountVectorizer(tokenizer=pos_tokenizer_uni, token_pattern=None)
    count_matrix = count_vect.fit_transform(all_lines)
    count_array = count_matrix.toarray()
    X = pd.DataFrame(data=count_array, columns=count_vect.get_feature_names_out())
    sums = X.sum(axis=1).replace(0, 1) # no zeros
    X = X.div(sums, axis=0) # mean across each row (each input line)
    y = pd.DataFrame(ys)

    return X, y


# make this not just N
def filter_pos(file, posCode):
    text = open(f'scraped-data/transcripts/{file}').read()
    tokens = nltk.word_tokenize(text.lower())
    pos_tags = nltk.pos_tag(tokens)
    return [word for word, pos in pos_tags if word.upper() not in util.PARTICIPANTS and (pos[0] == posCode)] #  or pos[0] == 'P'


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
    # sums = X.sum(axis=1).replace(0, 1) # no zeros
    # normalizedX = X.div(sums, axis=0) # mean across each row (each input line)
    y = pd.DataFrame(all_ys)

    return X, y


def logreg(X, y):
    # split data
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0) # , random_state=0

    # fit regression
    logisticRegr = LogisticRegression()
    logisticRegr.fit(x_train, y_train.values.ravel())

    # predict
    # predictions = logisticRegr.predict(x_test)
    print(logisticRegr.coef_)
    score = logisticRegr.score(x_test, y_test)
    print('logreg test score:', score)


if __name__ == "__main__":
    with open('scraped-data/ground_truths.json') as json_file:
        winners = json.load(json_file)

    # files = ['data/pres/09_26_2008.txt', 'data/pres/10_07_2008.txt', 'data/pres/10_15_2008.txt', 'data/vp/10_02_2008.txt']
    files = [f.name for f in os.scandir("scraped-data/transcripts")]

    print('All Logreg')
    X, y = generate_X_y(files, winners)
    logreg(X, y)

    print('Pronoun')
    X, y = generate_pronouns_y(files, winners)
    print(1 - y.mean())
    print(X[:20])
    logreg(X, y)

    print('Reg on Specific PoS (ex: Singular Noun)')
    X, y = generate_pos_y(files, winners)
    print(X[:20])
    logreg(X, y)

    print('Reg on PoS Family (ex: Noun)')
    X, y = generate_pos_uni_y(files, winners)
    print(X[:20])
    logreg(X, y)

    print('Noun')
    X, y = generate_X_y_pos_5(files, winners, 'N')
    print(X[:20])
    logreg(X, y)

    print('Verb')
    X, y = generate_X_y_pos_5(files, winners, 'V')
    print(X[:20])
    logreg(X, y)

    print('Adj')
    X, y = generate_X_y_pos_5(files, winners, 'J')
    print(X[:20])
    logreg(X, y)

    print('Adv')
    X, y = generate_X_y_pos_5(files, winners, 'R')
    print(X[:20])
    logreg(X, y)
