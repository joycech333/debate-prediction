import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers import AdamW
from keras.layers import Dropout
from keras.layers import LeakyReLU
from keras.regularizers import l1
from keras.regularizers import l2
from sklearn.metrics import accuracy_score

import zipfile
import pickle

# zip_file_path = 'ngrams_features.pickle.zip'
# extract_to_directory = 'ngrams'

# with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
#     zip_ref.extractall(extract_to_directory)

# with open('ngrams/ngrams_features.pickle', 'rb') as file:
#     features = pickle.load(file)

# with open('word2vec_features.pickle', 'rb') as file:
#     features = pickle.load(file)

with open('google_features.pickle', 'rb') as file:
  features = pickle.load(file)

# print(features[0])

X = np.array(features)
y = pd.read_csv('y_per_speaker.csv')
y = y.values.ravel()
y = y.astype(int)

n_splits = 10
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)

accuracy_scores = []

for train_index, test_index in skf.split(X, y):
    x_train, x_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    model = Sequential([
        Dense(64, activation='relu', input_shape=(X.shape[1],)),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer=Adam(1e-2), loss='binary_crossentropy', metrics=['accuracy'])

    model.fit(x_train, y_train, epochs=15, batch_size=32, verbose=0)

    # Evaluate the model
    y_pred = model.predict(x_test)
    y_pred = (y_pred >= 0.5)
    accuracy = accuracy_score(y_test, y_pred)
    accuracy_scores.append(accuracy)

# Calculate and print average accuracy across all folds
average_accuracy = np.mean(accuracy_scores)
print(f"Average Accuracy: {average_accuracy}")
