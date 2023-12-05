from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import numpy as np

def process_debate(file_paths):
    X = []
    y = []
    for file_path in file_paths:
        print()
        print(file_path)
        all_speakers = {}

        file = open(file_path).read()
        # colons denote when a speaker is speaking
        speeches = file.split(":")

        # loop through all the split text
        prev_speech = speeches[0]
        for speech in speeches[1:]:
            # check if word before colon is a speaker name
            if prev_speech[-1].isupper():
                speaker = prev_speech.split()[-1]

                lines = speech.split("\n")[:-1]
                
                if speaker not in all_speakers:
                    all_speakers[speaker] = []

                for line in lines:
                    all_speakers[speaker].append(line)

            prev_speech = speech


        for speaker in all_speakers:
            # for 2008, this means the speaker was the moderator
            if speaker not in ["OBAMA", "MCCAIN", "BIDEN", "PALIN"]:
                continue
            sentences = all_speakers[speaker]

            label = 0
            if speaker == "OBAMA" or speaker == "BIDEN":
                label = 1

            pos = 0
            neu = 0
            neg = 0
            for sentence in sentences:
                sid = SentimentIntensityAnalyzer()
                ss = sid.polarity_scores(sentence)
                temp = []
                for k in sorted(ss):
                    temp.append(ss[k])
                # understanding compound: https://towardsdatascience.com/social-media-sentiment-analysis-in-python-with-vader-no-training-required-4bc6a21e87b8
                compound = temp[0]
                X.append(compound)
                y.append(label)

                if compound <= -0.05:
                    neg += 1
                elif compound >= 0.05:
                    pos += 1
                else:
                    neu += 1

            
            print(speaker, "pos:" + str(pos), "neu:" + str(neu), "neg:" + str(neg))

    return X, y


# Specify the path to your debate text file
file_paths = ['scraped-data/transcripts/September_26_2008.txt', 'scraped-data/transcripts/October_07_2008.txt', 'scraped-data/transcripts/October_15_2008.txt', 'scraped-data/transcripts/October_02_2008.txt']

# Process the debate file
X, y = process_debate(file_paths)
print(len(X))
print(len(y))

# split data
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0) # , random_state=0

x_train= np.reshape(x_train, (-1, 1))
x_test= np.reshape(x_test, (-1, 1))

# fit regression
logisticRegr = LogisticRegression()
logisticRegr.fit(x_train, y_train)

# predict
# predictions = logisticRegr.predict(x_test)
score = logisticRegr.score(x_test, y_test)
print('logreg on words:', score)
