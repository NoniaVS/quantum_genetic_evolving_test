import pennylane as qml
from openfermionpyscf import run_pyscf
from src.genetic_al import GeneticAlgorithm
from src.tools import list_to_gates
from src.peasant import SimplePeasant
from src.mutators import OneGenMutator
from src.crossovers import OnePointCrossover
from src.fitness_functions import ErrorFunctionState
from src.selection_operators import StochasticUnivSampler, BestOnes, RouletteWheel
from openfermion import MolecularData
from pennylane import numpy as np
'''
#Define the molecule

symbols = ["H", "H"]
geometry =np.array([[0.0000, 0.0000, 0.0000],[ 0.0000,  0.0000, 1.5000]])
H, qubits = qml.qchem.molecular_hamiltonian(symbols, geometry, method="pyscf")
print(type(qml.TrotterProduct(H, time = 1, order= 1)))
exit()

#Obtain trotterized circuit
dev = qml.device("default.qubit")
@qml.qnode(dev)
def circuit():
    qml.TrotterProduct(H, time = 1, order= 1)
    return qml.state()
fig, ax = qml.draw_mpl(circuit)()
fig.show()
'''

dev = qml.device("default.qubit", wires=4)

@qml.qnode(dev)
def circuit():
    qml.PauliX(0)
    qml.Hadamard(1)
    qml.Hadamard(2)
    qml.Hadamard(3)
    qml.CNOT([0,1])
    qml.CNOT([1, 2])
    qml.CNOT([2, 3])
    qml.RZ(0.5,3)
    qml.CNOT([2, 3])
    qml.CNOT([1, 2])
    qml.CNOT([0,1])
    qml.adjoint(qml.PauliX(0))
    qml.Hadamard(1)
    qml.Hadamard(2)
    qml.adjoint(qml.PauliX(3))
    return qml.state()

target = circuit()
print(target)

#possible_gates = ['Identity', 'PauliY', 'PauliZ', 'PauliX', 'Hadamard','Phase', 'CNOT']
possible_gates = ['RX', 'RY', 'RZ', 'CNOT']

dna_feat = {'input' : possible_gates,
            'number_wires' : 4,
            'number_gates_init': 10}

population_feat = {'peasant_number' : 50,               #MUST be even
                   'generations' : 50,
                   'num_surv_per_gen' : 42,
                   'target_matrix' : target,
                   'mutation_probability': 0.1}


GA = GeneticAlgorithm(dna_features=dna_feat,
                      population_features=population_feat,
                      peasant=SimplePeasant,
                      fitness_function=ErrorFunctionState(),
                      crossover=OnePointCrossover(),
                      mutator=OneGenMutator(possible_gates),
                      selection= StochasticUnivSampler(),
                      print_evolution = True)

result = GA.evolve_population()



'''
dev = qml.device('default.qubit', wires = 2)
@qml.qnode(dev)
def circuit():
    qml.CNOT([0, 1])
    qml.Hadamard(1)
    return qml.expval(qml.PauliZ(0))

fig, ax = qml.draw_mpl(circuit)()
fig.show()
'''


list_1 = list_to_gates(result)
dev = qml.device('default.qubit',2)
@qml.qnode(dev)
def circuit():
    for gates in list_1:
        qml.apply(gates)
    return qml.expval(qml.PauliZ(0))

#fig, ax = qml.draw_mpl(circuit)()
#fig.show()






