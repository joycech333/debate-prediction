"""
File for commonly reused functions.
"""

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


if __name__ == "__main__":
    # file_paths = ['data/pres/09_26_2008.txt']
    file_paths = ['data/pres/09_26_2008.txt', 'data/pres/10_07_2008.txt', 'data/pres/10_15_2008.txt', 'data/vp/10_02_2008.txt']
    # for file_path in file_paths:
    #     split_speakers(file_path)