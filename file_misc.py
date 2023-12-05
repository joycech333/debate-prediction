import os
import json

with open('scraped-data/ground_truths.json', 'rt') as f:
    with open('scraped-data/ground_truths2.json', 'w') as fout:
        for line in f:
            fout.write(line.replace('True', '\"TRUE\"').replace('False', '\"FALSE\"').replace(": ,", ": \"\","))

# with open('scraped-data/ground_truths.json', 'r+') as f:
#     for line in f:
#         f.write(line.replace('True', '\"TRUE\"'))

# names = []
# for f in os.scandir("scraped-data/transcripts"):
#     names.append(f.name)

# files = [f.name for f in os.scandir("scraped-data/transcripts")]
# print(files)

# inOrder = sorted(names, key=lambda s: s[-8:])
# for f in inOrder:
#     print(f'"{f}": {{\n "win": ,\n "lose": ,\n "draw": \n}},')