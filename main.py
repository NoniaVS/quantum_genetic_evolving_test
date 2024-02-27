import numpy as np
import pennylane as qml

from src.genetic_al import GeneticAlgorithm
from src.tools import list_to_gates
from src.peasant import SimplePeasant
from src.mutators import OneGenMutator
from src.crossovers import OnePointCrossover
from src.fitness_functions import ErrorFunction
from src.selection_operators import StochasticUnivSampler, BestOnes


if __name__ == '__main__':

    if True:

        def circuit():
            qml.PauliX(0)
            qml.Hadamard(1)

        target = qml.matrix(circuit)()  #circuit_matrix is a numpy.ndarray


        possible_gates = ['Identity', 'PauliY', 'PauliZ','Hadamard', 'Phase', 'CNOT']


        dna_feat = {'input' : possible_gates,
                    'number_wires' : 2}

        population_feat = {'peasant_number' : 100,
                           'generations' : 200,
                           'surv_prop' : 1,     #High survival probability does not work
                           'target_matrix' : target}


        GA = GeneticAlgorithm(dna_features=dna_feat,
                              population_features=population_feat,
                              peasant=SimplePeasant,
                              fitness_function=ErrorFunction(),
                              crossover=OnePointCrossover(),
                              mutator=OneGenMutator(mutation_probability=0.7, add_probability = 0.6, del_probability = 0.4, possible_gates=possible_gates),
                              selection= BestOnes(),
                              print_evolution = True)

        GA.evolve_population()









        '''
        dev = qml.device('default.qubit', wires = 2)
        @qml.qnode(dev)
        def circuit():
            for gates in list_1:
                qml.apply(gates)
            return qml.expval(qml.PauliZ(0))

        fig, ax = qml.draw_mpl(circuit)()
        fig.show()
        '''




