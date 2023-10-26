from nltk.sentiment.vader import SentimentIntensityAnalyzer

def process_debate(file_paths):
    for file_path in file_paths:
        print()
        print(file_path)
        all_speakers = {}

        file = open(file_path).read()
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
                    all_speakers[speaker].append(line)

            prev_speech = speech


        for speaker in all_speakers:
            sentences = all_speakers[speaker]

            pos = 0
            neu = 0
            neg = 0
            for sentence in sentences:
                sid = SentimentIntensityAnalyzer()
                ss = sid.polarity_scores(sentence)
                temp = []
                for k in sorted(ss):
                    temp.append(ss[k])
                # understanding compound: https://towardsdatascience.com/social-media-sentiment-analysis-in-python-with-vader-no-training-required-4bc6a21e87b8
                compound = temp[0]

                if compound <= -0.05:
                    neg += 1
                elif compound >= 0.05:
                    pos += 1
                else:
                    neu += 1

            print(speaker, "pos: " + str(pos), "neu: " + str(neu), "neg: " + str(neg))


# Specify the path to your debate text file
file_paths = ['data/pres/09_26_2008.txt', 'data/pres/10_07_2008.txt', 'data/pres/10_15_2008.txt', 'data/vp/10_02_2008.txt']

# Process the debate file
process_debate(file_paths)