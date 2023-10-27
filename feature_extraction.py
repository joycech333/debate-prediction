import nltk
# nltk.download('punkt')
# nltk.download('stopwords')
from nltk.tokenize import word_tokenize
from nltk import FreqDist
from nltk.corpus import stopwords
import numpy as np
import util
from nltk.tokenize import RegexpTokenizer
import collections

PRONOUNS = ['he', 'him', 'his', 'she', 'her', 'hers', 'i', 'me', 'my', 'you', 'your', 'yours', 'they', 'them', 'theirs', 'it', 'its', 'we', 'us', 'our', 'ours', 'myself', 'yourself', 'himself', 'herself', 'ourselves', 'themselves']

"""
For each speaker, returns:
- dictionary of frequencies of only words
- dictionary of frequencies of words, excluding stopwords
- dictionary of frequencies of words + punctuation
- dictionary of frequencies of pronouns
"""

def tokenize(file_path, suppress = True):
    speaker_items = {}
    speaker_lines = util.split_speakers(file_path)
    tokenizer = RegexpTokenizer(r'\w+') # no punct
    stops = stopwords.words('english')

    for speaker in speaker_lines:
        if not suppress: print(f'\nSpeaker: {speaker}')
        lines = speaker_lines[speaker]
        speech = " ".join(lines).lower()

        # tokenize without punctuation
        word_tokens = tokenizer.tokenize(speech)
        word_freq = FreqDist(word_tokens)
        # if not suppress: print(f'Only Words: {word_freq.most_common(10)}')

        # pronoun tokens
        pronouns = {}
        for pronoun in PRONOUNS:
            pronouns[pronoun] = word_freq[pronoun]
        # Sort the pronouns by frequency in descending order
        top_5_pronouns = sorted(pronouns.items(), key=lambda x: x[1], reverse=True)[:5]
        if not suppress:
            print(f'Top 5 Most Frequent Pronouns: {top_5_pronouns}')

        # tokenize without punctuation, without stopwords
        tokens_nostop = [token for token in word_tokens if token not in stops]
        nostop_freq = FreqDist(tokens_nostop)
        if not suppress: print(f'No Stopwords: {nostop_freq.most_common(10)}')

        # tokenize with punctuation
        all_tokens = word_tokenize(speech)
        all_freq = FreqDist(all_tokens)
        # if not suppress: print(f'With Punctuation: {all_freq.most_common(10)}')

        speaker_items[speaker] = {'t5pronouns': top_5_pronouns, 'word_freq': word_freq, 'pronouns': pronouns, 'all_freq': all_freq, 'nostop_freq': nostop_freq}

    return speaker_items


if __name__ == "__main__":
    # file_path = 'data/pres/09_26_2008.txt'
    files = ['data/pres/09_26_2008.txt', 'data/pres/10_07_2008.txt', 'data/pres/10_15_2008.txt', 'data/vp/10_02_2008.txt']
    all_items = collections.defaultdict(dict)
    for file_path in files:
        speaker_items = tokenize(file_path, True)
        for speaker in speaker_items:
            all_items[speaker].update(speaker_items[speaker])
    
    for speaker in all_items:
        if speaker in ["OBAMA", "MCCAIN", "BIDEN", "PALIN"]:
            print(f'\n{speaker}:\n{all_items[speaker]["t5pronouns"]}')
            
    print('\n')