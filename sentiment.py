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
#files = ["September_26_2008.txt", "November_28_2007.txt", "September_25_1988.txt", "September_16_2015.txt", "April_26_2007.txt", "October_06_1976.txt"]

full_paths = [f'scraped-data/transcripts/{file}' for file in files]

# Process the debate file
X, y = process_debate(full_paths)

K = 10

k_folds = KFold(n_splits = K)
logisticRegr = LogisticRegression()
scores = cross_val_score(logisticRegr, np.reshape(X, (-1, 1)), y, cv = k_folds)
print(scores)
print(scores.mean())

