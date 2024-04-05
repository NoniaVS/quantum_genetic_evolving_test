import numpy as np
import pennylane as qml
from src.tools import list_to_gates

class ErrorFunctionState():

    def __init__(self):
        pass


    def evaluate_peasant(self, _peasant, target_state, num_wires):
        '''Functon to compute the fitness of the DNA of a peasant.
        The fitness operator is defined in:

        Parameters
        ----------
        _peasant : Peasant type
            Peasant to be evaluated
        '''

        #Obtain the matrix of the circuit
        gates = list_to_gates(_peasant.dna)
        dev = qml.device("default.qubit", wires=num_wires)

        @qml.qnode(dev)
        def circuit():
            for i in range(num_wires):
                qml.Identity(wires=i)
            for gate in gates:
                qml.apply(gate)
            return qml.state()

        #fig, ax = qml.draw_mpl(circuit)()
        #fig.show()
        #drawer = qml.draw(circuit)
        #print(drawer())

        circuit_state = circuit()
        #print((circuit_state))

        state0 = qml.math.dm_from_state_vector(target_state)
        state1 = qml.math.dm_from_state_vector(circuit_state)
        fidelity = qml.math.fidelity(state0, state1)

        _peasant.fitness = fidelity


