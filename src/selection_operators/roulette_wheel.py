import numpy as np
from copy import deepcopy
class RouletteWheel():

    def __init__(self):
        pass

    def selection(self, population, num_survivors):
        '''Function that applies the Roulette Wheel Selection to select
        the parents for the next generation. The same parent canbe chosen more than once.
        '''
        parent_population = []
        fitness = []
        for individual in population:
            fitness.append(individual.fitness)


        # This creates the segment between 0 and 1 where each peasant occupies a space proportional to its fitness
        def makeWheel():
            wheel = []
            fitness_sum = deepcopy(fitness)
            fitness_sum = np.array(fitness_sum)
            total = np.sum(fitness_sum)
            init_point = 0
            for i in range(len(fitness)):
                f = fitness_sum[i] / total
                to_add = (init_point, init_point + f, i)
                wheel.append(to_add)
                init_point += f
            return wheel

        wheel = makeWheel()
        for i in range(num_survivors):
            r = np.random.rand()
            for j in range(len(wheel)):
                if wheel[j][0] <= r < wheel[j][1]:
                    parent_population.append(population[wheel[j][2]])

        return parent_population