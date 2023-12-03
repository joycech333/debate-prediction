import csv
import numpy as np
import nltk
from nltk.corpus import stopwords

stops = stopwords.words('english')

def load_dataset(tsv_path):
    """Load the spam dataset from a TSV file

    Args:
         csv_path: Path to TSV file containing dataset.

    Returns:
        messages: A list of string values containing the text of each message.
        labels: The binary labels (0 or 1) for each message. A 1 indicates spam.
    """

    messages = []
    labels = []

    with open(tsv_path, 'r', newline='', encoding='utf8') as tsv_file:
        reader = csv.reader(tsv_file, delimiter='\t')

        for label, message in reader:
            messages.append(message)
            labels.append(1 if label == 'win' else 0)

    return messages, np.array(labels)


def get_words(message):
    """Get the normalized list of words from a message string.

    This function should split a message into words, normalize them, and return
    the resulting list. For splitting, you should split on spaces. For normalization,
    you should convert everything to lowercase.

    Args:
        message: A string containing an SMS message

    Returns:
       The list of normalized words from the message.
    """

    # *** START CODE HERE ***
    words = set()
    word_list = []
    message = message.strip().split()

    for m in message:
        m = m.lower()

        # stripping punct made it worse
        """
        if m[-1] == "." or m[-1] == ",":
            m = m[:-1]
        """

        # getting rid of stop words had no effect 
        if m in words or m in stops:
            continue
        else:
            words.add(m)
            word_list.append(m)

    return word_list
    # *** END CODE HERE ***


def create_dictionary(messages):
    """Create a dictionary mapping words to integer indices.

    This function should create a dictionary of word to indices using the provided
    training messages. Use get_words to process each message.

    Rare words are often not useful for modeling. Please only add words to the dictionary
    if they occur in at least five messages.

    Args:
        messages: A list of strings containing SMS messages

    Returns:
        A python dict mapping words to integers.
    """

    # *** START CODE HERE ***

    word_counts = {}

    for m in messages:
        word_list = get_words(m)
        for w in word_list:
            if w in word_counts:
                word_counts[w] += 1
            else:
                word_counts[w] = 1

    word_index = {}
    count = 0

    for w in word_counts:
        if word_counts[w] < 5:
            continue

        word_index[w] = count
        count += 1

    return word_index

    # *** END CODE HERE ***


def transform_text(messages, word_dictionary):
    """Transform a list of text messages into a numpy array for further processing.

    This function should create a numpy array that contains the number of times each word
    of the vocabulary appears in each message. 
    Each row in the resulting array should correspond to each message 
    and each column should correspond to a word of the vocabulary.

    Use the provided word dictionary to map words to column indices. Ignore words that
    are not present in the dictionary. Use get_words to get the words for a message.

    Args:
        messages: A list of strings where each string is an SMS message.
        word_dictionary: A python dict mapping words to integers.

    Returns:
        A numpy array marking the words present in each message.
        Where the component (i,j) is the number of occurrences of the
        j-th vocabulary word in the i-th message.
    """
    # *** START CODE HERE ***

    result = np.zeros([len(messages), len(word_dictionary)])

    for i in range(len(messages)):
        for word in messages[i].strip().split():
            word = word.lower()
            if word in word_dictionary:
                word_index = word_dictionary[word]
                result[i][word_index] += 1

    return result

    # *** END CODE HERE ***


def fit_naive_bayes_model(matrix, labels):
    """Fit a naive bayes model.

    This function should fit a Naive Bayes model given a training matrix and labels.

    The function should return the state of that model.

    Feel free to use whatever datatype you wish for the state of the model.

    Args:
        matrix: A numpy array containing word counts for the training data
        labels: The binary (0 or 1) labels for that training data

    Returns: The trained model
    """

    # *** START CODE HERE ***

    all_phi = np.zeros([len(matrix[0]), 2])

    total_spam = 0
    total_not = 0

    for m in range(len(matrix)):
        for k in range(len(matrix[m])):
            if labels[m] == 0:
                total_not += matrix[m][k]
            else:
                total_spam += matrix[m][k]

    for k in range(len(matrix[0])):
        count_0 = 0
        count_1 = 0
        for m in range(len(matrix)):
            if labels[m] == 0:
                count_0 += matrix[m][k]
            else:
                count_1 += matrix[m][k]
        all_phi[k][0] = (1 + count_0) / (total_not + len(matrix[0]))
        all_phi[k][1] = (1 + count_1) / (total_spam + len(matrix[0]))

    all_true = 0
    for i in range(len(labels)):
        if labels[i] == 1:
            all_true += 1

    phi_y = all_true / len(labels)

    return all_phi, phi_y

    # *** END CODE HERE ***


def predict_from_naive_bayes_model(model, matrix):
    """Use a Naive Bayes model to compute predictions for a target matrix.

    This function should be able to predict on the models that fit_naive_bayes_model
    outputs.

    Args:
        model: A trained model from fit_naive_bayes_model
        matrix: A numpy array containing word counts

    Returns: A numpy array containg the predictions from the model
    """
    # *** START CODE HERE ***
    all_phi = model[0]
    phi_y = model[1]

    preds = np.zeros([len(matrix)])
    
    for m in range(len(matrix)):
        pred_spam = 0
        pred_not = 0
        for i in range(len(matrix[0])):
            if matrix[m][i] != 0:
                pred_spam += np.log(all_phi[i][1]) * matrix[m][i]
                pred_not += np.log(all_phi[i][0]) * matrix[m][i]

        pred_spam += np.log(phi_y)
        pred_not += np.log(1-phi_y)
        if pred_spam > pred_not:
            preds[m] = 1
        else:
            preds[m] = 0

    return preds
    # *** END CODE HERE ***


def get_top_five_naive_bayes_words(model, dictionary):
    """Compute the top five words that are most indicative of the spam (i.e positive) class.

    Ues the metric given in part-c as a measure of how indicative a word is.
    Return the words in sorted form, with the most indicative word first.

    Args:
        model: The Naive Bayes model returned from fit_naive_bayes_model
        dictionary: A mapping of word to integer ids

    Returns: A list of the top five most indicative words in sorted order with the most indicative first
    """
    # *** START CODE HERE ***

    model = model[0]

    indicators = []

    for i in range(len(model)):
        indicators.append((i, np.log(model[i][1]/model[i][0])))

    indicators.sort(key = lambda x: x[1])

    top = indicators[-5:]

    backwards_dict = {}
    for elem in dictionary:
        backwards_dict[dictionary[elem]] = elem

    # NN: noun, JJ: adjective, anything with VB is a verb
    words = dictionary.keys()
    tags = nltk.pos_tag(words)

    pos = {"nouns": [], "adj": [], "vb": []}

    noun_indicators = []
    adj_indicators = []
    vb_indicators = []

    for t in tags:
        if t[1] == "NN":
            pos["nouns"].append(t[0])
        elif t[1] == "JJ":
            pos["adj"].append(t[0])
        elif "VB" in t[1]:
            pos["vb"].append(t[0])

    for ind in indicators:
        if backwards_dict[ind[0]] in pos["nouns"]:
            noun_indicators.append(ind)
        elif backwards_dict[ind[0]] in pos["adj"]:
            adj_indicators.append(ind)
        elif backwards_dict[ind[0]] in pos["vb"]:
            vb_indicators.append(ind)

    noun_indicators.sort(key = lambda x: x[1])
    top_noun = noun_indicators[-5:]
    adj_indicators.sort(key = lambda x: x[1])
    top_adj = adj_indicators[-5:]
    vb_indicators.sort(key = lambda x: x[1])
    top_vb = vb_indicators[-5:]

    result = []
    result_noun = []
    result_adj = []
    result_vb = []
    for i in range(4, -1, -1):
        ind = top[i][0]
        result.append(backwards_dict[ind])
        result_noun.append(backwards_dict[top_noun[i][0]])
        result_adj.append(backwards_dict[top_adj[i][0]])
        result_vb.append(backwards_dict[top_vb[i][0]])

    print("noun", result_noun)
    print("adj", result_adj)
    print("vb", result_vb)

    return result

    # *** END CODE HERE ***


def main():
    train_messages, train_labels = load_dataset('data/pres/train_2008.tsv')
    test_messages, test_labels = load_dataset('data/pres/test_2008.tsv')

    dictionary = create_dictionary(train_messages)

    print('Size of dictionary: ', len(dictionary))
    
    train_matrix = transform_text(train_messages, dictionary)

    test_matrix = transform_text(test_messages, dictionary)

    naive_bayes_model = fit_naive_bayes_model(train_matrix, train_labels)
    
    naive_bayes_predictions = predict_from_naive_bayes_model(naive_bayes_model, test_matrix)

    naive_bayes_accuracy = np.mean(naive_bayes_predictions == test_labels)

    print('Naive Bayes had an accuracy of {} on the testing set'.format(naive_bayes_accuracy))
    
    top_5_words = get_top_five_naive_bayes_words(naive_bayes_model, dictionary)

    print('The top 5 indicative words for Naive Bayes are: ', top_5_words)
    

if __name__ == "__main__":
    main()
