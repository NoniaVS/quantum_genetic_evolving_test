import numpy as np
import pennylane as qml
from copy import deepcopy
import random
import scipy
from .tools import function_to_minimize, function_to_minimize_matrix
from src.tools import list_to_gates

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
        mutation_probability = self.__population_features['mutation_probability']
        mutation_probability_add = self.__population_features['mutation_probability_add']
        mutation_probability_del = self.__population_features['mutation_probability_del']
        target_matrix = self.__population_features['target_matrix']
        num_wires = self.__dna_features['number_wires']
        n_population = len(self.__population)
        self.__evolution_fitness = np.zeros((n_generations, n_population))
        fitness_evolution = []
        for generation in range(n_generations):
            print('--------------------------------------')
            print('GENERATION NUMBER', generation)
            print('--------------------------------------')


            # Generate offsprings from parents
            print('DOING CROSSOVER')
            pair_production = deepcopy(self.__population)
            def pop_random(lst):
                idx = random.randrange(0, len(lst))
                return lst.pop(idx)

            list_couples = []
            while pair_production:
                rand1 = pop_random(pair_production)
                rand2 = pop_random(pair_production)
                pair = rand1, rand2
                list_couples.append(pair)
            for i in range(len(list_couples)):
                dna = self.__crossover.generate_dna(list_couples[i][0], list_couples[i][1])
                peasant = self.__peasant(dna_features = list_couples[i][0].dna_features, dna = dna)
                self.__population.append(peasant)
                dna = self.__crossover.generate_dna(list_couples[i][1], list_couples[i][0])
                peasant = self.__peasant(dna_features=list_couples[i][0].dna_features, dna=dna)
                self.__population.append(peasant)


            print('MUTATING POPULATION')
            for i in range(len(self.__population)):
                r = np.random.rand()
                # There exists a probability of the mutation not happening
                if r > mutation_probability:
                    continue
                else:
                    mutated_dna = self.__mutator.mutation_arb_change(self.__population[i].dna, num_wires)
                    self.__population[i].dna = mutated_dna

            for i in range(len(self.__population)):
                r = np.random.rand()
                # There exists a probability of the mutation not happening
                if r > mutation_probability_add:
                    continue
                else:
                    mutated_dna = self.__mutator.mutation_add_gate(self.__population[i].dna, num_wires)
                    self.__population[i].dna = mutated_dna

            for i in range(len(self.__population)):
                r = np.random.rand()
                # There exists a probability of the mutation not happening
                if r > mutation_probability_del:
                    continue
                else:
                    mutated_dna = self.__mutator.mutation_del_gate(self.__population[i].dna)
                    self.__population[i].dna = mutated_dna





            print('OPTIMIZING PARAMETERS')

            for i in range(len(self.__population)):
                init_params = []
                '''
                if generation == 0:
                    num_params = 0
                    for k in range(len(self.__population[i].dna)):
                        if self.__population[i].dna[k]['gate_name'] != 'CNOT':
                            num_params += 1
                    init_params = np.zeros(num_params)
                else:
                '''
                for k in range(len(self.__population[i].dna)):
                    if self.__population[i].dna[k]['gate_name'] != 'CNOT':
                        init_params.append(self.__population[i].dna[k]['rotation_param'])
                if len(init_params) == 0:
                    continue
                else:
                    res = scipy.optimize.minimize(fun = function_to_minimize_matrix, x0= init_params, args = (self.__population[i], target_matrix, num_wires),
                                              method= 'SLSQP')

                self.__population[i].fitness = 1 - res.fun

                new_params = res.x
                count = 0
                for k in range(len(self.__population[i].dna)):
                    if self.__population[i].dna[k]['gate_name'] != 'CNOT':
                        a = self.__population[i].dna
                        a[k]['rotation_param'] = new_params[count]
                        self.__population[i].dna = a

                        count += 1





            # Compute fitness for generation
            for peasant in self.__population:
                self.__fitness_function.evaluate_peasant(peasant, target_matrix, num_wires)

            # Sort population by their fitness and print the dna of the best
            self.__population.sort(key=lambda x: x.fitness, reverse=True)
            if self.__print_evolution:
                print(
                    f"Generation {generation} : DNA {self.__population[0].dna} : Fitness {self.__population[0].fitness}")

                print('THE FITNESS FOR THE FOLLOWING CIRCUIT IS:', self.__population[0].fitness)
                gates = list_to_gates(self.__population[0].dna)
                dev = qml.device("default.qubit", wires=num_wires)

                @qml.qnode(dev)
                def circuit():
                    for i in range(num_wires):
                        qml.Identity(wires=i)
                    for gate in gates:
                        qml.apply(gate)
                    return qml.state()

                # fig, ax = qml.draw_mpl(circuit)()
                # fig.show()
                drawer = qml.draw(circuit)
                print(drawer())

                fitness_evolution.append(self.__population[0].fitness)
                print('+++++++++')
                print('TOP 3')
                for i in range(3):
                    gates = list_to_gates(self.__population[i].dna)
                    dev = qml.device("default.qubit", wires=num_wires)

                    @qml.qnode(dev)
                    def circuit():
                        for i in range(num_wires):
                            qml.Identity(wires=i)
                        for gate in gates:
                            qml.apply(gate)
                        return qml.state()

                    # fig, ax = qml.draw_mpl(circuit)()
                    # fig.show()
                    drawer = qml.draw(circuit)
                    print(drawer())




            if np.abs(self.__population[0].fitness) > 0.98:
                print('ARRIVED TO THE DESIRED FIDELITY, THE FINAL CIRCUIT HAS THE FOLLOWING FORM:')
                gates = list_to_gates(self.__population[0].dna)
                dev = qml.device("default.qubit", wires=num_wires)

                @qml.qnode(dev)
                def circuit():
                    for i in range(num_wires):
                        qml.Identity(wires=i)
                    for gate in gates:
                        qml.apply(gate)
                    return qml.state()

                # fig, ax = qml.draw_mpl(circuit)()
                # fig.show()
                drawer = qml.draw(circuit)
                print(drawer())
                print('STATE', circuit())

                print('WITH A FIDELITY OF:', self.__population[0].fitness)
                break

            print('DOING SELECTION')
            final_population = self.__selection.selection(self.__population, num_survivors= self.__population_features['num_surv_per_gen'])

            self.__population = deepcopy(final_population)
            if generation == n_generations - 1:
                print('ARRIVED TO THE MAXIMUM GENERATIONS, THE FINAL CIRCUIT HAS THE FOLLOWING FORM:')
                gates = list_to_gates(self.__population[0].dna)
                dev = qml.device("default.qubit", wires=num_wires)

                @qml.qnode(dev)
                def circuit():
                    for i in range(num_wires):
                        qml.Identity(wires=i)
                    for gate in gates:
                        qml.apply(gate)
                    return qml.state()

                # fig, ax = qml.draw_mpl(circuit)()
                # fig.show()
                drawer = qml.draw(circuit)
                print(drawer())

                print('WITH A FIDELITY OF:', self.__population[0].fitness)


        return self.__population[0].dna, fitness_evolution




