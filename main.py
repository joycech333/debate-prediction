import numpy as np
import csv
import matplotlib.pyplot as plt
import json

# count words for an individual speech
def count_words(speech):
    words = speech.split()
    return len(words)


def process_debate(file_path):
    all_speakers = {}
    times = {}

    file = open(file_path).read()
    # colons denote when a speaker is speaking
    speeches = file.split(":")

    # loop through all the split text
    prev_speech = speeches[0]
    for speech in speeches[1:]:
        # check if word before colon is a speaker name
        if prev_speech[-1].isupper():
            speaker = prev_speech.split()[-1]
            if speaker not in all_speakers:
                all_speakers[speaker] = 0
                times[speaker] = 0
            # add the word count to their speech
            all_speakers[speaker] += count_words(speech)
            # add an instance of talking to the speaker
            times[speaker] += 1
        prev_speech = speech
    return (all_speakers, times)


def main(file_path):
    print(f"\nFile: {file_path}")

    # Process the debate file
    speech_counts, times = process_debate(file_path)

    # Print word counts for each participant
    for participant in speech_counts:
        total = speech_counts[participant]
        freq = times[participant]
        print(f"{participant} spoke {total} words.")
        print(f"{participant} spoke {freq} times.")
        print(f"On average, {participant} spoke {total / freq} words per speech.")


if __name__ == "__main__":
    # file_paths = ['data/pres/09_26_2008.txt']
    file_paths = ['data/pres/09_26_2008.txt', 'data/pres/10_07_2008.txt', 'data/pres/10_15_2008.txt', 'data/vp/10_02_2008.txt']
    for file_path in file_paths:
        main(file_path)