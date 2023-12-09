# train/valid/test: 80/10/10
import numpy as np

eras = ["era1", "era2", "era3"]
train_data = []
valid_data = []
test_data = []

train_label = []
valid_label = []
test_label = []

for era in eras:
    all_data = np.loadtxt("bert_" + era + "_data.csv", delimiter=",")
    all_label = np.loadtxt("bert_" + era + "_label.csv", delimiter=",")

