from sentence_transformers import SentenceTransformer
import util
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

model = SentenceTransformer('all-mpnet-base-v2')

splits = util.split_speakers("data/transcripts/September_26_2008.txt")
clean_splits = util.clean_cand_names(["Obama", "McCain", "John"], splits)

sentences_obama = clean_splits["OBAMA"]

embeddings_obama = model.encode(sentences_obama)

pca = PCA(2)
projected_obama = pca.fit_transform(embeddings_obama)
print(projected_obama.shape)

sentences_mccain = clean_splits["MCCAIN"]

embeddings_mccain = model.encode(sentences_mccain)

projected_mccain = pca.fit_transform(embeddings_mccain)
print(projected_mccain.shape)

plt.scatter(projected_obama[:, 0], projected_obama[:, 1], color="blue")
plt.scatter(projected_mccain[:, 0], projected_mccain[:, 1], color="red")
plt.xlabel('component 1')
plt.ylabel('component 2')
plt.show()