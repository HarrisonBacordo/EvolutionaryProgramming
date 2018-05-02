import numpy as np
import NeuralNetwork as nn
import csv

fname = 'ass2DataFiles/part1/iris.data'


def process_data(file):
    feats = list()
    labls = list()
    with open(file, 'rt') as csvfile:
        lines = csv.reader(csvfile)
        for line in lines:
            if line:
                feats.append(line[:-1])
                labls.append(line[-1])
        print(feats)
        print(labls)
        return np.array(feats).astype(float), np.array(labls)


features, labels = process_data(fname)
for i in range(10):
    indx = np.random.permutation(len(features))
    features = np.vstack((features, features[indx]))
    labels = np.hstack((labels, labels[indx]))
labels = nn.one_hot(labels)
net = nn.NeuralNetwork([4, 5, 3], "sigmoid")
net.train(features, labels, 1000)
