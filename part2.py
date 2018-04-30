import operator
import numpy as np
from deap import creator, base, tools, gp, algorithms


def prep_data(fname):
    with open(fname, 'r') as file:
        values = list()
        for line in file:
            values.append(' '.join(line.split()).split(' '))
        del values[1], values[0]
    return values


def protected_div(left, right):
    try:
        return left / right
    except ZeroDivisionError:
        return 1


def square(x):
    return x**2


def evalsymbreg(individual, points):
    f = toolbox.compile(expr=individual)
    # Calculate mean-squared error
    diff = 0
    for point in points:
        diff += (f(float(point[0])) - float(point[1]))**2
    return diff/len(points),


# Initialise fitness and individual classes
creator.create("FitnessMin", base.Fitness, weights=(-1.0, 1))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

# set operators that can be used in tree
pset = gp.PrimitiveSet("MAIN", arity=1)
pset.renameArguments(ARG0="x")
pset.addPrimitive(operator.add, 2)
pset.addPrimitive(operator.sub, 2)
pset.addPrimitive(operator.mul, 2)
pset.addPrimitive(operator.neg, 1)
pset.addPrimitive(protected_div, 2)
pset.addPrimitive(square, 1)

toolbox = base.Toolbox()
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=10)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)


def main(datafile):
    data = prep_data(datafile)

    # Initialise evolutionary computations
    toolbox.register("evaluate", evalsymbreg, points=data)
    toolbox.register("mate", tools.cxOnePoint)
    toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
    toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)
    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))
    toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))

    stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
    stats_size = tools.Statistics(len)
    mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
    mstats.register("avg", np.mean)
    mstats.register("std", np.std)
    mstats.register("min", np.min)
    mstats.register("max", np.max)

    pop = toolbox.population(n=100)
    hof = tools.HallOfFame(1)

    pop, log = algorithms.eaSimple(pop, toolbox, 1, 0.8, 50, stats=mstats,
                                   halloffame=hof, verbose=True)
    print("\n\n" + str(hof.keys[0]) + "-->" + str(hof.items[0]))
    return pop, mstats, hof


if __name__ == '__main__':
    main('ass2DataFiles/part2/regression.txt')