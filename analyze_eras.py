import util
era1, era2, era3 = util.split_by_era()

for era in [era1, era2, era3]:
    print("NEW ERA")
    full_paths = [f'scraped-data/transcripts/{file}' for file in era]

    for file_path in full_paths:
        file_date = file_path[25:]
        print(file_date)

        all_speakers = util.split_speakers(file_path)

        for speaker in all_speakers:
            debate_cands = util.PARTICIPANTS[file_date]

            # this means the speaker was the moderator
            if speaker not in debate_cands:
                continue
            print(speaker, len(all_speakers[speaker]))