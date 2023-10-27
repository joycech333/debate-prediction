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

def tokenize(files, winner):
    for file_path in files:
        filename = file_path.split('/')[-1]
        winner = winners[filename].upper()
        speaker_lines = util.split_speakers(file_path)

        all_lines = []
        ys = []
        for speaker in speaker_lines:
            lines = speaker_lines[speaker]

            # for 2008, this means the speaker was not the moderator
            if speaker in ["OBAMA", "MCCAIN", "BIDEN", "PALIN"]:
                all_lines.extend(lines)
                ys.extend([speaker == winner] * len(lines))

    count_vect = CountVectorizer(stop_words='english')
    count_matrix = count_vect.fit_transform(all_lines)
    count_array = count_matrix.toarray()
    X = pd.DataFrame(data=count_array, columns=count_vect.get_feature_names_out())
    y = pd.DataFrame(ys)

    # split data
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.25) # , random_state=0

    # fit regression
    logisticRegr = LogisticRegression()
    logisticRegr.fit(x_train, y_train.values.ravel())

    # predict
    # predictions = logisticRegr.predict(x_test)
    score = logisticRegr.score(x_test, y_test)
    print(score)



if __name__ == "__main__":
    with open('data/ground_truths.json') as json_file:
        winners = json.load(json_file)

    files = ['data/pres/09_26_2008.txt', 'data/pres/10_07_2008.txt', 'data/pres/10_15_2008.txt', 'data/vp/10_02_2008.txt']
    speaker_items = tokenize(files, winners)