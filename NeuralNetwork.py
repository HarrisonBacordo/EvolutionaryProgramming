import numpy as np


class NeuralNetwork:
    def __init__(self, structure, activation):
        """
        Creates skeleton of neural network
        :param structure: a list that gives the number of layers (length of list) and the number of units in those
        layers (numbers in list).
        :param activation: type of activation function to use throughout neural network
        """
        self.lr = None
        self.structure = structure
        self.bias = list()
        self.nn = list()
        self.activation = activation
        for i in range(len(self.structure) - 1):
            self.nn.append(np.random.uniform(-1, 1, (self.structure[i+1], self.structure[i])))
            self.bias.append(np.ones(self.structure[i+1])[:, None])

    def train(self, features_list, labels, epochs, batch_size=20, learning_rate=0.1):
        """
        Trains the neural network on the given data
        :param features_list: the input data for the neural network
        :param labels: the correct answers for each of the features
        :param epochs: amount of batch_sizes to do before halting training
        :param batch_size: amount of guesses to do before each backprop
        :param learning_rate: the rate at which to attempt to optimize the model
        :return:
        """
        self.lr = learning_rate
        for i, features in enumerate(features_list):
            # store the nn's guess and a list of each layer's values
            guess, states = self.feed_forward(features)
            # calculate cost and compute the gradient
            if self.structure[-1] != 1:
                labels[i] = labels[i][:, None]
            cost = np.atleast_2d(np.subtract(labels[i], guess))
            obj_cost = np.mean(np.abs(cost))
            # print("\n\nACTUAL: \n", labels[i], "\nGUESS\n", guess, "\nCOST: ", obj_cost)
            self.compute_gradient(cost, states)

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

        #     #     #     #     HIDDEN LAYER     #     #     #     #
        hcost = costs  # stores the cost of the -> layer
        hcount = len(hstate)
        for i in reversed(range(len(self.nn)-1)):
            # calculate hidden gradient

            # gets the errors for this layer based on the -> layer errort layer
            who_t = np.transpose(self.nn[i+1])
            hcost = who_t.dot(hcost)
            # get the gradients of this layer
            # multiply this layer's gradients by the cost of the -> layer.
            # multiply the above by the learning rate scalar
            hgradients = dsig(hstate[i])
            hgradients = np.multiply(hgradients, hcost)
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
            hcount -= 1

    def test(self, features, labels):
        _sum = 0
        obj_cost = 0
        for i, feature in enumerate(features):
            guess, _ = self.feed_forward(feature)
            if np.argmax(guess) == list(labels[i]).index(1):
                _sum += 1
            cost = np.subtract(labels[i][:, None], guess)
            obj_cost += np.mean(np.abs(cost))
        print("---------TEST SUMMARY----------")
        print("NUMBER CORRECT: ",  _sum, "OUT OF", len(features))
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
    indx = np.random.permutation(len(features))
    return features[indx], labels[indx]


def clone_and_shuffle(features, labels, cycles):
    feats = features
    labls = labels

    for i in range(cycles):
        indx = np.random.permutation(len(feats))
        feats = np.vstack((feats, feats[indx]))
        labls = np.hstack((labls, labls[indx]))
    return feats, labls
