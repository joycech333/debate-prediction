from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.linear_model import LogisticRegression
import numpy as np
import util
import os
from sklearn.model_selection import KFold, cross_val_score

def process_debate(file_paths):
    X = []
    y = []
    for file_path in file_paths:
        file_date = file_path[25:]
        print(file_date)

        all_speakers = util.split_speakers(file_path)

        for speaker in all_speakers:
            debate_cands = util.PARTICIPANTS[file_date]

            # this means the speaker was the moderator
            if speaker not in debate_cands:
                continue
            sentences = all_speakers[speaker]

            debate_winners = [util.WINNERS[file_date]["win"].upper()]
            
            if util.WINNERS[file_date]["draw"] == "TRUE":
                debate_winners.append(util.WINNERS[file_date]["lose"].upper())

            label = 0
            if speaker in debate_winners:
                label = 1

            cur_sentiment = 0
            for sentence in sentences:
                sid = SentimentIntensityAnalyzer()
                ss = sid.polarity_scores(sentence)
                
                compound = ss["compound"]
                cur_sentiment += compound

            X.append(cur_sentiment / len(sentences))
            y.append(label)

    return X, y





files = [f.name for f in os.scandir("scraped-data/transcripts")]
print(len(files))

#files = ["September_26_2008.txt", "November_28_2007.txt", "September_25_1988.txt", "September_16_2015.txt", "April_26_2007.txt", "October_06_1976.txt"]
#files = ["September_26_2008.txt", "November_28_2007.txt"]

era1, era2, era3 = util.split_by_era()
files = era1

files = util.get_mult_cand_debates()

full_paths = [f'scraped-data/transcripts/{file}' for file in files]

#print(full_paths)

# Process the debate file
X, y = process_debate(full_paths)

X = np.reshape(X, (-1, 1))
y = np.reshape(y, (-1, 1))

print(X)
print(X.shape)
print(y)
print(y.shape)



np.savetxt("sentiment_multcand_data.csv", X, delimiter = ",")
np.savetxt("sentiment_multcand_label.csv", y, delimiter = ",")

assert False

"""
K = 10

k_folds = KFold(n_splits = K)
logisticRegr = LogisticRegression()
scores = cross_val_score(logisticRegr, np.reshape(X, (-1, 1)), y, cv = k_folds)
print(scores)
print(scores.mean())
"""

from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
X = np.reshape(X, (-1, 1))
y_encoded = OrdinalEncoder().fit_transform(np.reshape(y, (-1, 1)))
import xgboost as xgb

X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, random_state=1, stratify=y_encoded)


dtrain_clf = xgb.DMatrix(X_train, y_train, enable_categorical=True)
dtest_clf = xgb.DMatrix(X_test, y_test, enable_categorical=True)

params = {"objective": "multi:softprob", "tree_method": "gpu_hist", "num_class": 5}
n = 1000

results = xgb.cv(
   params, dtrain_clf,
   num_boost_round=n,
   nfold=5,
   metrics=["mlogloss", "auc", "merror"],
)

print(results['test-auc-mean'].max())