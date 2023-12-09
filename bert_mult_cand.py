import util
import numpy as np

era1, era2, era3 = util.split_by_era()
eras_list = [era1, era2, era3]

mult_cand = util.get_mult_cand_debates()

X = []

era1_embed = np.loadtxt("data/bert_era1_data.csv", delimiter=",")
era2_embed = np.loadtxt("data/bert_era2_data.csv", delimiter=",")
era3_embed = np.loadtxt("data/bert_era3_data.csv", delimiter=",")

print(era1_embed.shape, era2_embed.shape, era3_embed.shape)

era_embeds = [era1_embed, era2_embed, era3_embed]

for i in range(1, 4):
    era = eras_list[i-1]
    counter = 0
    for j in range(len(era)):
        #file_date = file_path[25:]
        #print(file_date)

        all_speakers = util.split_speakers("scraped-data/transcripts/"+era[j])

        for speaker in all_speakers:
            debate_cands = util.PARTICIPANTS[era[j]]

            # this means the speaker was the moderator
            if speaker not in debate_cands:
                continue
            #sentences = all_speakers[speaker]
            if era[j] in mult_cand:
                X.append(np.array(era_embeds[i-1][counter]))

            counter += 1
            print(counter)

X = np.array(X)
print(X.shape)

np.savetxt("bert_multcand_data.csv", X, delimiter = ",")