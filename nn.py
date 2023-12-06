import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from sklearn.model_selection import train_test_split
import pandas as pd

def main():
    X = pd.read_csv('xys/all_x_text_per_speaker.csv')
    y = pd.read_csv('xys/y_per_speaker.csv')

    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(x_train)
    total_words = len(tokenizer.word_index) + 1

    sequences = tokenizer.texts_to_sequences(x_train)
    padded_sequences = pad_sequences(sequences)

    # Build the model
    embedding_dim = 16
    model = Sequential([
        Embedding(input_dim=total_words, output_dim=embedding_dim, input_length=padded_sequences.shape[1]),
        LSTM(100),
        Dense(1, activation='sigmoid')
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Train the model
    model.fit(padded_sequences, y_train.values.ravel(), epochs=10)

    test_sequences = tokenizer.texts_to_sequences(x_test)
    padded_test_sequences = pad_sequences(test_sequences)

    # predictions = model.predict(padded_test_sequences)
    # rounded_predictions = np.round(predictions)
    # print(rounded_predictions)

    loss, accuracy = model.evaluate(padded_test_sequences, y_test)
    print(f"Test Loss: {loss}, Test Accuracy: {accuracy}")


if __name__ == "__main__":
    main()