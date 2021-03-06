import numpy as np
import math


class NeuralNetwork:
    def __init__(self, structure, activation):
        """
        Creates skeleton of neural network
        :param structure: a list that gives the number of layers (length of list) and the number of units in those
        layers (numbers in list).
        :param activation: type of activation function to use throughout neural network
        """
        self.lr = None
        self.initial_lr = None
        self.lr_drop = None
        self.epochs_drop = None
        self.momentum = None
        self.prev_deltas = list()
        self.structure = structure
        self.bias = list()
        self.nn = list()
        self.activation = activation
        for i in range(len(self.structure) - 1):
            self.nn.append(np.random.uniform(-1, 1, (self.structure[i+1], self.structure[i])))
            self.prev_deltas.append(np.zeros((self.structure[i+1], self.structure[i])))
            self.bias.append(np.ones(self.structure[i+1])[:, None])

    def train(self, features_list, labels, epochs, batch_size=20, learn_decay=False, learning_rate=0.1, momentum=0.8,
              lr_drop=0.5, epochs_drop=10):
        """
        Trains the neural network on the given data
        :param features_list: the input data for the neural network
        :param labels: the correct answers for each of the features
        :param epochs: amount of batch_sizes to do before halting training
        :param batch_size: amount of guesses to do before each backprop
        :param learn_decay: whether learn decay should be implemented
        :param learning_rate: the rate at which to attempt to optimize the model
        :param momentum: momentum of the neural network
        :param lr_drop: rate of drop on learning rate decay
        :param epochs_drop: how often to drop the learning rate
        :return:
        """
        self.lr = learning_rate
        self.initial_lr = learning_rate
        self.lr_drop = lr_drop
        self.epochs_drop = epochs_drop
        self.momentum = momentum
        for i, features in enumerate(features_list):
            # store the nn's guess and a list of each layer's values
            guess, states = self.feed_forward(features)
            # calculate cost and compute the gradient
            if self.structure[-1] != 1:
                labels[i] = labels[i][:, None]
            cost = np.atleast_2d(np.subtract(labels[i], guess))
            obj_cost = np.mean(np.abs(cost))
            # print("COST: ", obj_cost)
            self.compute_gradient(cost, states)
            if learn_decay:
                self.lr = self.step_decay(i)

    def feed_forward(self, features):
        """
        feeds the features forward through the NN
        :param features: features to feed through the neural network
        :return: the resulting guess (in the form of a vector), along with the state of the NN
        """
        sigm = np.vectorize(sigmoid)
        states = list()
        state = np.array(features)[:, None]
        states.append(state)
        for i, weights in enumerate(self.nn):
            state = np.dot(weights, state)
            state = np.add(state, self.bias[i])
            state = sigm(state)
            states.append(state)
        return state, states

    def compute_gradient(self, costs, states):
        """
        Implements SGD on the costs that are passed in
        :param costs: avg costs of this mini batch
        :param states: current state of the neural network
        :return: TODO Unsure of this right now
        """
        #     #     #     #      OUTPUT LAYER     #     #     #     #
        # calculate output gradient
        dsig = np.vectorize(derivsig)  # vectorize the derivsig function
        # grab hidden layers, and output layer
        istate = states[0]
        hstate = states[1:-1]
        ostate = states[-1]
        # get the gradient of the output state.
        # multiply it by costs vector and learning rate
        ogradient = dsig(ostate)
        ogradient = np.multiply(costs, ogradient)
        ogradient = np.multiply(ogradient, self.lr)
        # transpose the the hlayer connected to the outlayer.
        hidden_t = np.transpose(hstate[-1])
        # multiply the output gradient by the transposed hidden layer.
        # This is the change to be applied to the output weights.
        weight_ho_deltas = np.dot(ogradient, hidden_t)

        self.bias[-1] = np.add(self.bias[-1], ogradient)
        self.nn[-1] = np.add(self.nn[-1], weight_ho_deltas)
        self.nn[-1] = np.add(self.nn[-1], self.momentum * self.prev_deltas[-1])
        self.prev_deltas[-1] = weight_ho_deltas

        #     #     #     #     HIDDEN LAYER     #     #     #     #
        hcosts = costs  # stores the cost of the -> layer
        hcount = len(hstate)
        for i in reversed(range(len(self.nn)-1)):
            # calculate hidden gradient

            # gets the errors for this layer based on the dot product of the transposed -> layer's weights
            # and the error costs of that layer
            whl_t = np.transpose(self.nn[i+1])
            hcosts = whl_t.dot(hcosts)
            # get the gradients of this layer
            hgradients = dsig(hstate[i])
            # multiply this layer's gradients by the cost of the -> layer.
            hgradients = np.multiply(hgradients, hcosts)
            # multiply the above by the learning rate scalar
            hgradients = np.multiply(hgradients, self.lr)

            # transpose the <- layer's state
            if i == 0:
                hidden_t = np.transpose(istate)
            else:
                hidden_t = np.transpose(hstate[hcount-1])

            # multiply this layer's gradients by the transposed <- layer
            # This is the change to be applied to this layer's weights.
            weight_hh_deltas = np.dot(hgradients, hidden_t)
            self.bias[i] = np.add(self.bias[i], hgradients)
            self.nn[i] = np.add(self.nn[i], weight_hh_deltas)
            self.nn[i] = np.add(self.nn[i], self.momentum * self.prev_deltas[i])
            self.prev_deltas[i] = weight_hh_deltas
            hcount -= 1

    def step_decay(self, epoch):
        lr = self.initial_lr * pow(self.lr_drop,
                                   math.floor((1+epoch) / self.epochs_drop))
        return lr

    def test(self, features, labels):
        """
        Tests the network on the given features and labels, outputting a summary of the results
        :param features: features to be tested on
        :param labels: labels to be tested on
        """
        correct = 0
        obj_cost = 0  # represents the sum of the mean cost of each feed-forward process
        for i, feature in enumerate(features):
            guess, _ = self.feed_forward(feature)
            # check if guess was correct. If so, add 1 to the number correct
            if np.argmax(guess) == list(labels[i]).index(1):
                correct += 1
            # subtract expected - actual to get the cost of each node
            cost = np.subtract(labels[i][:, None], guess)
            # get mean of the costs of each node to get average cost for this iteration, add this to the obj_cost
            obj_cost += np.mean(np.abs(cost))

        # print results
        print("---------TEST SUMMARY----------")
        print("NUMBER CORRECT: ",  correct, "OUT OF", len(features))
        print("AVERAGE COST: ", obj_cost/len(features))


def sigmoid(x):
    """
    implements sigmoid on num
    :param x: number to implement sigmoid on
    :return: result from sigmoid calculation
    """
    return 1 / (1 + np.exp(-x))


def derivsig(y):
    """
    implements derivative of sigmoid
    :param y: number to implement derivsig on
    :return: result from derivsig calculation
    """
    return y * (1 - y)


def one_hot(labels):
    """
    Converts the passed in labels into one hot format
    :param labels: labels to be converted
    :return: nparray of labels in one hot format
    """
    # get number of unique labels
    unique = list(set(labels))
    numclasses = len(unique)
    one_hots = list()
    for label in labels:
        onehot = np.zeros(numclasses)
        onehot[unique.index(label)] = 1
        one_hots.append(onehot)
    return one_hots


def shuffle(features, labels):
    """
    Shuffles the given dataset based off of numpy's permutation function
    :param features: features to shuffle
    :param labels: labels to be shuffled
    :return: the shuffled features and labels
    """
    indx = np.random.permutation(len(features))
    return features[indx], labels[indx]


def clone_and_shuffle(features, labels, cycles):
    """
    Duplicates the dataset a given number of cycles, and shuffles the resulting
    dataset on each function. This is mainly called when you don't have a sufficient
    amount of data to train a neural network on, so you repeatedly duplicate and
    shuffle the same data until you have a satisfactory amount
    :param features: features to clone and shuffle
    :param labels: labels to clone and shuffle
    :param cycles: number of times to clone and shuffle
    :return: the cloned and shuffled features and labels
    """
    feats = features
    labls = labels
    for i in range(cycles):
        indx = np.random.permutation(len(feats))
        feats = np.vstack((feats, feats[indx]))
        labls = np.hstack((labls, labls[indx]))
    return feats, labls
