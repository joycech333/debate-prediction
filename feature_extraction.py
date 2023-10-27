import nltk
# nltk.download('punkt')
# nltk.download('stopwords')
from nltk.tokenize import word_tokenize
from nltk import FreqDist
from nltk.corpus import stopwords
import numpy as np
import util
from nltk.tokenize import RegexpTokenizer

PRONOUNS = ['he', 'him', 'his', 'she', 'her', 'hers', 'I', 'me', 'my', 'you', 'your', 'yours', 'they', 'them', 'theirs', 'it', 'its']

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
        if not suppress: print(f'Only Words: {word_freq.most_common(10)}')

        # pronoun tokens
        pronouns = {}
        for pronoun in PRONOUNS:
            pronouns[pronoun] = word_freq[pronoun]
        if not suppress: print(f'Pronouns: {pronouns}')

        # tokenize without punctuation, without stopwords
        tokens_nostop = [token for token in word_tokens if token not in stops]
        nostop_freq = FreqDist(tokens_nostop)
        if not suppress: print(f'No Stopwords: {nostop_freq.most_common(10)}')

        # tokenize with punctuation
        all_tokens = word_tokenize(speech)
        all_freq = FreqDist(all_tokens)
        if not suppress: print(f'With Punctuation: {all_freq.most_common(10)}')

        speaker_items[speaker] = {'word_freq': word_freq, 'pronouns': pronouns, 'all_freq': all_freq, 'nostop_freq': nostop_freq}

    return speaker_items


if __name__ == "__main__":
    file_path = 'data/pres/09_26_2008.txt'
    speaker_items = tokenize(file_path, False)