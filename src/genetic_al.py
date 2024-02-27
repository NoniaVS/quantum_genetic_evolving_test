import numpy as np
import pennylane as qml
from copy import deepcopy

class GeneticAlgorithm():

    def __init__(self, dna_features, population_features, peasant, fitness_function, crossover, mutator, selection, print_evolution = False):
        '''Constructor of GeneticAlgorithm class
        '''

        self.__dna_features = dna_features
        self.__population_features = population_features
        self.__peasant = peasant
        self.__fitness_function = fitness_function
        self.__crossover = crossover
        self.__mutator = mutator
        self.__selection = selection
        self.__print_evolution = print_evolution

        # Population data
        self.__population = self.__generate_population()
        self.__evolution_fitness = None


    def __generate_population(self):
        '''Generates the complete list of Peasants

        Returns
        -------
        list
            List of Peasants
        '''

        pop_size = self.__population_features['peasant_number']
        population = []
        for ii in range(pop_size):
            peasant = self.__peasant(dna_features = self.__dna_features)
            population.append(peasant)

        return population


    def evolve_population(self):
        '''Function to evolve the first population ultil convergence or max
        generations'''

        n_generations = self.__population_features['generations']
        survival_probability = self.__population_features['surv_prop']
        target_matrix = self.__population_features['target_matrix']
        num_wires = self.__dna_features['number_wires']
        n_population = len(self.__population)
        self.__evolution_fitness = np.zeros((n_generations, n_population))

        for generation in range(n_generations):

            # Compute fitness for generation
            for peasant in self.__population:
                self.__fitness_function.evaluate_peasant(peasant, target_matrix, num_wires)

            # Sort population by their fitness and print the dna of the best
            self.__population.sort(key = lambda x: x.fitness, reverse=True)
            if self.__print_evolution:
                print(f"Generation {generation} : DNA {self.__population[0].dna} : Fitness {self.__population[0].fitness}")

            # Save generational fitness
            generation_fitness = np.zeros(n_population)
            for enum, peasant in enumerate(self.__population):
                generation_fitness[enum] = peasant.fitness

            self.__evolution_fitness[generation, :] = generation_fitness
            if np.abs(self.__population[0].fitness) > 0.999999:
                break

            #Select parents using Stochastic universal sampling [https://doi.org/10.1109/EH.2002.1029883]
            num_survivors = int(len(self.__population) * survival_probability)
            parent_population = self.__selection.selection(population=self.__population, num_survivors=num_survivors)
            self.__population = []
            # Generate offsprings from parents
            for enum in range(n_population):
                par_1, par_2 = np.random.choice(parent_population, 2)
                # Make the crossover
                dna = self.__crossover.generate_dna(par_1, par_2)
                # Mutate the offspring
                dna = self.__mutator.mutation_arb_change(dna, num_wires)
                dna = self.__mutator.mutation_add_gate(dna, num_wires)
                dna = self.__mutator.mutation_del_gate(dna)
                peasant =self.__peasant(dna_features = par_1.dna_features, dna = dna)
                self.__population.append(peasant)

        #self.__evolution_fitness = self.__evolution_fitness[:generation, :]



