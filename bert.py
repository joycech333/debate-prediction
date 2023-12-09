from sentence_transformers import SentenceTransformer
import util
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import LogisticRegression

model = SentenceTransformer('all-mpnet-base-v2')

#era2 = util.get_two_cand_debates()


era1, era2, era3 = util.split_by_era()

two_cand = util.get_two_cand_debates()
for era in [two_cand]:
    X = []
    y = []
    full_paths = [f'scraped-data/transcripts/{file}' for file in era]

    for file_path in full_paths:
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
            full_speech = ""
            if speaker in debate_winners:
                label = 1
                for sent in sentences:
                    full_speech += " " + sent
            else:
                for sent in sentences:
                    full_speech += " " + sent


            encoding = model.encode(full_speech)
            
            X.append(encoding)
            y.append(label)

    X = np.array(X)
    print(X.shape)
    print(len(y))

    y = np.reshape(y, (-1, 1))

    print(X)
    print(X.shape)
    print(y)
    print(y.shape)

    np.savetxt("bert_twocand_data.csv", X, delimiter = ",")
    np.savetxt("bert_twocand_label.csv", y, delimiter = ",")


    """
    K = 10

    k_folds = KFold(n_splits = K)
    logisticRegr = LogisticRegression()
    scores = cross_val_score(logisticRegr, X, y, cv = k_folds)
    print(scores)
    print(scores.mean())
    """
            

"""

splits = util.split_speakers("data/transcripts/September_26_2016.txt")
old_splits = util.split_speakers("data/transcripts/October_13_1960.txt")
#clean_splits = util.clean_cand_names(["Obama", "McCain", "John"], splits)

#print(old_splits)

kennedy_sent = []

sentences_obama = old_splits["KENNEDY"]

for elem in sentences_obama:
    elem = elem.strip().split(".")
    for sent in elem:
        sent = sent.strip()
        kennedy_sent.append(sent)

embeddings_obama = model.encode(kennedy_sent)
print(embeddings_obama.shape)

pca = PCA(2)
projected_obama = pca.fit_transform(embeddings_obama)
print(projected_obama.shape)

sentences_mccain = splits["TRUMP"]

embeddings_mccain = model.encode(sentences_mccain)

projected_mccain = pca.fit_transform(embeddings_mccain)
print(projected_mccain.shape)

plt.scatter(projected_obama[:, 0], projected_obama[:, 1], color="blue")
plt.scatter(projected_mccain[:, 0], projected_mccain[:, 1], color="red")
plt.xlabel('component 1')
plt.ylabel('component 2')
plt.show()

"""

