import numpy as np

class StochasticUnivSampler():

    def __init__(self):
        pass

    def selection(self, population, num_survivors):
        '''Function that applies the Stochastic Universal Sampling to select
        the parents for the next generation
        [https://www.researchgate.net/publication/228963181_Modelling_of_a_stochastic_universal_sampling_selection_operator_in_genetic_algorithms_using_generalized_nets]
        '''
        parent_population = []
        fitness = []
        for individual in population:
            fitness.append(individual.fitness)


        # This creates the segment between 0 and 1 where each peasant occupies a space proportional to its fitness
        def makeWheel():
            wheel = []
            total = np.sum(fitness)
            init_point = 0
            for i in range(len(fitness)):
                f = fitness[i] / total
                wheel.append((init_point, init_point + f, i))
                init_point += f
            return wheel

        # Here we generate the random position of the first pointer
        r = np.random.rand()

        # Then create the rest N pointers, separated between them with equal distances
        space_between_pointers = (1 - r)/(num_survivors+1)
        point = r
        pointers = [point]
        for i in range(num_survivors-1):
            point = point + space_between_pointers
            pointers.append(point)

        # Create the wheel and store the chosen peasants in parent_population
        wheel = makeWheel()
        for i in range(len(pointers)):
            for j in range(len(wheel)):
                if wheel[j][0] <= pointers[i] < wheel[j][1]:
                    parent_population.append(population[wheel[j][2]])

        return parent_population