import util

all = util.ys_by_speaker()

losers = 0
for elem in all:
    if elem == [False]:
        losers += 1

print(losers/len(util.ys_by_speaker()))
