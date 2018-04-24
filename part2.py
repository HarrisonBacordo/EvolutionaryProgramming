import operator
import random
from deap import creator, base, tools, gp

CROSSOVER_PROBABILITY = 0.90
MUTATION_PROBABILITY = 0.05


def evaluateInd(individual):
    return eval(individual)

# set operators that can be used in tree
pset = gp.PrimitiveSet("MAIN", arity=1)
pset.addPrimitive(operator.add, 2)
pset.addPrimitive(operator.sub, 2)
pset.addPrimitive(operator.mul, 2)
pset.addPrimitive(operator.truediv, 2)

# Initialise fitness and individual classes
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin,
               pset=pset)


toolbox = base.Toolbox()
# Initialise evolutionary computations
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.2)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("evaluate", evaluateInd)
# Initialise initial population of 100 using half and half
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=5, max_=10)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
pop = toolbox.population(n=100)
print(pop)

for g in range(100):
    # Select next generation of individuals
    offspring = toolbox.select(pop, len(pop))
    # Clone the selected individuals
    offspring = list(map(toolbox.clone, offspring))

    # Apply crossover on the offspring
    for child1, child2 in zip(offspring[::2], offspring[1::2]):
        if random.random() < CROSSOVER_PROBABILITY:
            toolbox.mate(child1, child2)
            del child1.fitness.values
            del child2.fitness.values

    # Apply mutation on the offspring
    for mutant in offspring:
        if random.random() < MUTATION_PROBABILITY:
            toolbox.mutate(mutant)
            del mutant.fitness.values

    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    # Replace population with offspring
    pop[:] = offspring



# create individual
ind1 = toolbox.individual()
# clone and mutate individual, delete the fitness values
mutant = toolbox.clone(ind1)
gp.mutUniform(ind1)
del mutant.fitness.values
print(evaluateInd(ind1))
