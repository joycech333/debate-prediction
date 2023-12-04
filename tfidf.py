from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize.casual import casual_tokenize
import util

from sklearn.decomposition import SparsePCA
from matplotlib import pyplot as plt

from fbpca import pca

from nltk.corpus import stopwords

stops = stopwords.words('english')

tfidf = TfidfVectorizer()

sent = util.split_speakers("data/transcripts/September_26_2008.txt")
obama_sent = sent["OBAMA"]
mccain_sent = sent["MCCAIN"]

obama_sent_cleaned = []
mccain_sent_cleaned = []

for elem in obama_sent:
    cleaned = ""
    for word in elem:
        if word in stops or word in ["Obama", "McCain", "John"]:
            continue
        else:
            cleaned += word
    obama_sent_cleaned.append(cleaned)

"""
for elem in mccain_sent:
    mccain_sent_str += elem + " "
"""

result_obama = tfidf.fit_transform(raw_documents=obama_sent_cleaned)
print(result_obama.shape)
result_mccain = tfidf.fit_transform(raw_documents=mccain_sent)

pca1 = pca(2)

obama_pca = pca1.fit_transform(result_obama)
mccain_pca = pca1.fit_transform(result_mccain)

plt.scatter(obama_pca[:, 0], obama_pca[:, 1], color="blue")
plt.scatter(mccain_pca[:, 0], mccain_pca[:, 1], color="red")
plt.show()

