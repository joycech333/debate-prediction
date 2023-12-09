import os
import util

files = [f.name for f in os.scandir("scraped-data/transcripts")]
full_paths = [f'scraped-data/transcripts/{file}' for file in files]

shortest_words = -1
shortest_date = ""

longest_words = -1
longest_date = ""

for file_path in full_paths:
    file_date = file_path[25:]
    print(file_date)

    all_speakers = util.split_speakers(file_path)
    count = 0
    for speaker in all_speakers:
        debate_cands = util.PARTICIPANTS[file_date]

        # this means the speaker was the moderator
        if speaker not in debate_cands:
            continue
        sentences = all_speakers[speaker]

        
        for sent in sentences:
            count += len(sent.strip().split())

        if shortest_words == -1 or count < shortest_words:
            shortest_words = count
            shortest_date = file_date

        if longest_words == -1 or count > longest_words:
            longest_words = count
            longest_date = file_date        


print(shortest_words, shortest_date, longest_words, longest_date)