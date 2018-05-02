import operator
import NeuralNetwork
import random
import math
import itertools
import numpy as np
from deap import creator, base, tools, gp, algorithms


def prep_data(fname):
    """
    Formats data and puts it into a list for the script to read off of
    :param fname: directory to file
    :return: list of the formatted data
    """
    with open(fname, 'r') as file:
        featureslabels = list()
        for line in file:
            line = line.replace("?", "-1").strip().split(",")
            featureslabels.append(list(map(int, line[1:])))
        perc = round(len(featureslabels) * 0.8)
        training = featureslabels[:perc]
        test = featureslabels[perc:]
    print(training)
    print(test)

    return training, test


def protected_div(left, right):
    """
    Executes division with a protection from division by zero error
    :param left: left side of operation
    :param right: right side of operation
    :return: result of operation
    """
    try:
        return left / right
    except ZeroDivisionError:
        return 1


def square(x):
    """
    Squares the argument
    :param x: number to square
    :return: squared number
    """
    return x*x


def evalClassif(individual, features):
    """
    evaluates the individual based on the given points using mean-squared error
    :param individual: individual to be evaluated
    :param points: points to test individual on
    :return: the mean-squared error of the individual
    """
    f = toolbox.compile(expr=individual)
    result = 0
    for feature in features:
        if feature[9] == 4:
            label = True
        else:
            label = False
        if bool(f(*feature[:9])) == label:
            result += 1
    return result,


# Initialise fitness and individual classes
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)

# Initialise toolbox
toolbox = base.Toolbox()

# set operators that can be used in tree.
pset = gp.PrimitiveSetTyped("MAIN", itertools.repeat(float, 9), bool, "IN")
pset.addPrimitive(operator.add, [float, float], float)
pset.addPrimitive(operator.sub, [float, float], float)
pset.addPrimitive(operator.mul, [float, float], float)
pset.addPrimitive(protected_div, [float, float], float)


# logic operators
# Define a new if-then-else function
def if_then_else(input, output1, output2):
    if input:
        return output1
    else:
        return output2


pset.addPrimitive(operator.lt, [float, float], bool)
pset.addPrimitive(operator.le, [float, float], bool)
pset.addPrimitive(operator.gt, [float, float], bool)
pset.addPrimitive(operator.ge, [float, float], bool)
pset.addPrimitive(operator.eq, [float, float], bool)
pset.addPrimitive(if_then_else, [bool, float, float], float)

# terminals
pset.addEphemeralConstant("rand100", lambda: random.random() * 100, float)
pset.addTerminal(False, bool)
pset.addTerminal(True, bool)


def main(datafile):
    training, test = prep_data(datafile)

    random.seed(10)

    # Initialise evolutionary computations
    toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=2)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("compile", gp.compile, pset=pset)
    toolbox.register("evaluate", evalClassif, features=training)
    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("mate", gp.cxOnePoint)
    toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
    toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)
    toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))
    toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))

    # Initialise the statistic measurement to be shown throughout evolutionary process
    stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
    stats_size = tools.Statistics(len)
    mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
    mstats.register("avg", np.mean)
    mstats.register("std", np.std)
    mstats.register("min", np.min)
    mstats.register("max", np.max)

    # create population and hall of fame
    pop = toolbox.population(n=300)
    hof = tools.HallOfFame(1)

    # execute algorithm and print out results
    pop, log = algorithms.eaSimple(pop, toolbox, 1, 0.65, 300, stats=mstats,
                                   halloffame=hof, verbose=True)
    print("\n\n" + str(hof.keys[0]) + "-->" + str(hof.items[0]))
    return pop, mstats, hof


if __name__ == '__main__':
    main('ass2DataFiles/part3/breast-cancer-wisconsin.data')
