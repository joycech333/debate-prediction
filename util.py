"""
File for commonly reused functions.
example usage:
import util
...
util.split_speakers(file_path)
"""
import os
import re

PRONOUNS = ['he', 'him', 'his', 'she', 'her', 'hers', 'i', 'me', 'my', 'you', 'your', 'yours', 'they', 'them', 'theirs', 'it', 'its', 'we', 'us', 'our', 'ours', 'myself', 'yourself', 'himself', 'herself', 'ourselves', 'themselves']

"""
Generates a dict of file --> set of participants.
"""
def get_participants():
    files = [f.name for f in os.scandir("scraped-data/participants")]
    participants = {}
    for file in files:
        path = f'scraped-data/participants/{file}'
        participants[file] = []
        with open(path) as f:
            for line in f:
                matches = re.findall(r'\b(\w+)\s*\(', line)
                for match in matches:
                    participants[file].append(match.upper())
    
    print(participants)
    return participants

PARTICIPANTS = get_participants()

"""
Splits the speech at a given txt file_path by speaker
in the form of a dict:
speaker --> list of lines
"""
def split_speakers(file_path):
    file = open(file_path).read()
    all_speakers = {}
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
                if line:
                    all_speakers[speaker].append(line)

        prev_speech = speech

    return all_speakers

# clean a sentence of candidate names
def clean_cand_names(names, split_dict):
    for speaker in split_dict:
        clean_sent = []
        sentences = split_dict[speaker]
        for sent in sentences:
            for name in names:
                sent = sent.replace(name, "")
            clean_sent.append(sent)

        split_dict[speaker] = clean_sent
    return split_dict

# get the data in the format required by the naive bayes classifier
def create_data_tsv(all_dict, file_path, winner, loser):
    file = open(file_path, "w")

    for d in all_dict:
        for speaker in d:
            if speaker == winner:
                for sent in d[speaker]:
                    file.write("win" + "\t" + sent + "\n")
            if speaker == loser:
                for sent in d[speaker]:
                    file.write("lose" + "\t" + sent + "\n")


if __name__ == "__main__":
    # file_paths = ['data/pres/09_26_2008.txt', 'data/pres/10_07_2008.txt', 'data/pres/10_15_2008.txt']
    # split_dicts = []
    # for file_path in file_paths:
    #     split_dict = split_speakers(file_path)
    #     split_dicts.append(clean_cand_names(["Obama", "McCain", "John"], split_dict))

    # create_data_tsv([split_dicts[0], split_dicts[1]], "data/pres/train_2008.tsv", "OBAMA", "MCCAIN")
    # create_data_tsv([split_dicts[2]], "data/pres/test_2008.tsv", "OBAMA", "MCCAIN")

    # print(PARTICIPANTS)
    pass