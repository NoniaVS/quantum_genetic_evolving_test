import numpy as np
import pennylane as qml
from src.tools import list_to_gates

class ErrorFunctionMatrix():

    def __init__(self):
        pass


    def evaluate_peasant(self, _peasant, target_matrix, num_wires):
        '''Functon to compute the fitness of the DNA of a peasant.
        The fitness operator is defined in: https://doi.org/10.1109/EH.2002.1029883 but with lambda adapted

        Parameters
        ----------
        _peasant : Peasant type
            Peasant to be evaluated
        '''

        #Obtain the matrix of the circuit
        gates = list_to_gates(_peasant.dna)
        def circuit():
            for i in range(num_wires):
                qml.Identity(wires=i)
            for gate in gates:
                qml.apply(gate)

        circuit_matrix = qml.matrix(circuit)()  #circuit_matrix is a numpy.ndarray
        error = np.sum(np.abs(circuit_matrix - target_matrix))
        _peasant.fitness = 1/(1+error)


