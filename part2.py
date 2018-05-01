import operator
import random
import math
import numpy as np
from deap import creator, base, tools, gp, algorithms


def prep_data(fname):
    """
    Formats data and puts it into a list for the script to read off of
    :param fname: directory to file
    :return: list of the formatted data
    """
    with open(fname, 'r') as file:
        values = list()
        for line in file:
            values.append(' '.join(line.split()).split(' '))
        del values[1], values[0]
    return values


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


def evalsymbreg(individual, points):
    """
    evaluates the individual based on the given points using mean-squared error
    :param individual: individual to be evaluated
    :param points: points to test individual on
    :return: the mean-squared error of the individual
    """
    f = toolbox.compile(expr=individual)
    # Calculate mean-squared error
    diff = 0
    for point in points:
        diff += (f(float(point[0])) - float(point[1]))**2
    return diff/len(points),


# Initialise fitness and individual classes
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

# Initialise toolbox
toolbox = base.Toolbox()

# set operators that can be used in tree.
pset = gp.PrimitiveSet("MAIN", 1)
pset.renameArguments(ARG0='x')
pset.addPrimitive(operator.add, 2)
pset.addPrimitive(operator.sub, 2)
pset.addPrimitive(operator.mul, 2)
pset.addPrimitive(operator.neg, 1)
pset.addPrimitive(protected_div, 2)
pset.addPrimitive(square, 1)
pset.addPrimitive(math.cos, 1)
pset.addPrimitive(math.sin, 1)
pset.addEphemeralConstant("rand101", lambda: random.randint(-11, 11))


def main(datafile):
    data = prep_data(datafile)

    # Initialise evolutionary computations
    toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=2)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("compile", gp.compile, pset=pset)
    toolbox.register("evaluate", evalsymbreg, points=data)
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
    pop, log = algorithms.eaSimple(pop, toolbox, 1, 0.05, 100, stats=mstats,
                                   halloffame=hof, verbose=True)
    print("\n\n" + str(hof.keys[0]) + "-->" + str(hof.items[0]))
    return pop, mstats, hof


if __name__ == '__main__':
    main('ass2DataFiles/part2/regression.txt')
