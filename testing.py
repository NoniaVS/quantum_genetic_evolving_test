import pennylane as qml
from openfermionpyscf import run_pyscf
from src.genetic_al import GeneticAlgorithm
from src.tools import list_to_gates
from src.peasant import SimplePeasant
from src.mutators import OneGenMutator
from src.crossovers import OnePointCrossover
from src.fitness_functions import ErrorFunctionState, ErrorFunctionMatrix
from src.selection_operators import StochasticUnivSampler, BestOnes, RouletteWheel
from openfermion import MolecularData
import numpy as np
import matplotlib.pyplot as plt
from math import pi

resulting_fidelities = []
angle = []
for i in np.arange(0, 2 * np.pi, 0.1):
    angle.append(i)

    dev = qml.device("default.qubit", wires=7)
    @qml.qnode(dev)
    def circuit_target():
        qml.Hadamard(0)
        qml.PauliX(1)
        qml.Hadamard(3)
        qml.Hadamard(4)
        qml.PauliX(6)
        qml.CNOT([0, 1])
        qml.CNOT([1, 2])
        qml.CNOT([2, 3])
        qml.CNOT([3, 4])
        qml.CNOT([4, 5])
        qml.CNOT([5, 6])
        qml.RZ(i, 6)
        qml.CNOT([5, 6])
        qml.CNOT([4, 5])
        qml.CNOT([3, 4])
        qml.CNOT([2, 3])
        qml.CNOT([1, 2])
        qml.CNOT([0, 1])
        qml.Hadamard(0)
        qml.adjoint(qml.PauliX(1))
        qml.Hadamard(3)
        qml.Hadamard(4)
        qml.adjoint(qml.PauliX(6))
        return qml.state()

    state_target = circuit_target()
    @qml.qnode(dev)
    def circuit_equivalent():
        qml.RX(i, 3)
        qml.CNOT([3, 4])
        qml.CNOT([3, 0])
        return qml.state()

    state_equivalent = circuit_equivalent()


    state0 = qml.math.dm_from_state_vector(state_target)
    state1 = qml.math.dm_from_state_vector(state_equivalent)
    fidelity = round(qml.math.fidelity(state0, state1), 5)
    resulting_fidelities.append(fidelity)
    print(fidelity)

plt.plot(angle,resulting_fidelities)
plt.ylabel('Fidelities')
plt.xlabel('Angle')
plt.show()





