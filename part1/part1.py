import numpy as np
import NeuralNetwork as nn
import csv

fname = '../ass2DataFiles/part1/iris.data'


def process_data(file):
    feats = list()
    labls = list()
    with open(file, 'rt') as csvfile:
        lines = csv.reader(csvfile)
        for line in lines:
            if line:
                feats.append(line[:-1])
                labls.append(line[-1])
        return np.array(feats).astype(float), np.array(labls)


initfeats, initlabels = process_data(fname)
for i in range(8):
    print("\n\nITR ", i+1)
    testlabels = nn.one_hot(initlabels)
    features, labels = nn.clone_and_shuffle(initfeats, initlabels, 7)
    labels = nn.one_hot(labels)
    net = nn.NeuralNetwork([4, 5, 3], "sigmoid")
    net.train(features, labels, 1000, momentum=0.8, learning_rate=0.05)
    net.test(initfeats, testlabels)
