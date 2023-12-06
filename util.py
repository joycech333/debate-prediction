"""
File for commonly reused functions.
example usage:
import util
...
util.split_speakers(file_path)
"""
import os
import re
import json

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
    
    return participants


def get_winners():
    with open('scraped-data/ground_truths.json') as json_file:
        return json.load(json_file)


PARTICIPANTS = get_participants()
WINNERS = get_winners()
FILES = [f.name for f in os.scandir("scraped-data/transcripts")]

def split_by_era():
    pre2000 = []
    early2000 = []
    post2010 = []

    for f in FILES:
        year = int(f[-8:-4])

        if year <= 2000:
            pre2000.append(f)
        elif year <= 2010:
            early2000.append(f)
        else:
            post2010.append(f)

    return pre2000, early2000, post2010


# be careful and make sure your X's generate with the exact same number of rows (exception handling)
def generate_ys_per_line():
    all_ys = []
    for filename in FILES:
        winner = WINNERS[filename]['win'].upper()
        loser = WINNERS[filename]['lose'].upper()
        draw = WINNERS[filename]['draw']

        # ignore files with no ground truth
        if not winner:
            continue
        
        participants = PARTICIPANTS[filename]
        speaker_lines = split_speakers(f'scraped-data/transcripts/{filename}', True)

        speaker_ys = []
        for speaker in speaker_lines:

            if speaker.upper() in participants:
                # account for ties
                won = (speaker == winner or (draw and speaker == loser))
                speaker_ys.extend([won] * len(speaker_lines[speaker]))
        
            all_ys.extend(speaker_ys)

    return all_ys


def ys_by_speaker():
    winners = []
    for filename in FILES:
        winner = WINNERS[filename]['win'].upper()
        loser = WINNERS[filename]['lose'].upper()
        draw = WINNERS[filename]['draw']

        # ignore files with no ground truth
        if not winner:
            continue
        
        participants = PARTICIPANTS[filename]
        speaker_lines = split_speakers(f'scraped-data/transcripts/{filename}', True)

        for speaker in speaker_lines:

            if speaker.upper() in participants:
                # account for ties
                won = (speaker == winner or (draw and speaker == loser))
                winners.append([won])

    return winners


"""
Splits the speech at a given txt file_path by speaker
in the form of a dict:
speaker --> list of lines

To get all lines for a speaker into a single string, can do '\n'.join(speakers[speaker])
"""
def split_speakers(file_path, escape=False):
    file = open(file_path).read()
    all_speakers = {}
    # colons denote when a speaker is speaking
    speeches = file.split(":")

    # loop through all the split text
    prev_speech = speeches[0]
    for speech in speeches[1:]:
        # check if word before colon is a speaker name
        if prev_speech and prev_speech[-1].isupper():
            speaker = prev_speech.split()[-1]

            lines = speech.split("\n")[:-1]
            
            if speaker not in all_speakers:
                all_speakers[speaker] = []

            for line in lines:
                if line:
                    if escape:
                        participants = PARTICIPANTS[file_path.split('/')[-1]]
                        for name in participants:
                            line = line.replace(name, "").replace(name.title(), "")
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
def create_data_tsv(file_paths, out_file):
    file = open(out_file, "w")

    just_participants = set()

    for file_path in file_paths:
        file_date = file_path[25:]

        all_speakers = split_speakers(file_path)

        for speaker in all_speakers:
            debate_cands = PARTICIPANTS[file_date]

            just_participants.update(debate_cands)

            # this means the speaker was the moderator
            if speaker not in debate_cands:
                continue
            sentences = all_speakers[speaker]

            debate_winners = [WINNERS[file_date]["win"].upper()]
            
            if WINNERS[file_date]["draw"] == "TRUE":
                debate_winners.append(WINNERS[file_date]["lose"].upper())

            full_speech = ""
            if speaker in debate_winners:
                for sent in sentences:
                    full_speech += " " + sent
                    
                file.write("win" + "\t" + full_speech + "\n")
            else:
                for sent in sentences:
                    full_speech += sent
                    
                file.write("lose" + "\t" + full_speech + "\n")

    return just_participants


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
