import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize.casual import casual_tokenize

from sklearn.decomposition import PCA
from matplotlib import pyplot as plt


name_of_data_file = "uciml_spam.csv"


# load csv into a pandas data frame
text_df = pd.read_csv( name_of_data_file, encoding = "ISO-8859-1", engine ='python' )

# inspect
print(text_df.head(2))

name_of_text_data_column = "v2"

# Select Model
tfidf_model = TfidfVectorizer(tokenizer=casual_tokenize)

# Fit Model
tfidf_vectors = tfidf_model.fit_transform( raw_documents=text_df[name_of_text_data_column] ).toarray()

# Inspection:
print(type(tfidf_vectors))
print(tfidf_vectors.shape)
print(text_df[name_of_text_data_column].shape)

raise ValueError

print(text_df[name_of_text_data_column])

pca = PCA(n_components=2)

# Name of Vector Array (Numpy)
name_of_vector_array = tfidf_vectors

# New D2 Dataframe (PCA)
df2d = pd.DataFrame(pca.fit_transform(name_of_vector_array), columns=list('xy'))

print(df2d.shape)


"""
# Plot Data Visualization (Matplotlib)
df2d.plot(kind='scatter', x='x', y='y')
plt.show()
"""