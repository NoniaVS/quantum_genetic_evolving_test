import numpy as np
from copy import deepcopy
class RouletteWheel():

    def __init__(self):
        pass

    def selection(self, population, num_survivors):
        '''Function that applies the Stochastic Universal Sampling to select
        the parents for the next generation
        [https://www.researchgate.net/publication/228963181_Modelling_of_a_stochastic_universal_sampling_selection_operator_in_genetic_algorithms_using_generalized_nets]
        '''
        parent_population = []
        fitness = []
        print('NUM SURVIVORS', num_survivors)
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
                f = fitness[i] / total
                wheel.append((init_point, init_point + f, i))
                init_point += f
            return wheel

        wheel = makeWheel()
        print(wheel)
        exit()
        for i in range(num_survivors):
            r = np.random.rand()

            for j in range(len(wheel)):
                if wheel[j][0] <= r < wheel[j][1]:
                    parent_population.append(population[wheel[j][2]])

        return parent_population