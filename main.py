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
from pennylane import numpy as np
import matplotlib.pyplot as plt
from math import pi
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
'''
@qml.qnode(dev)
def circuit():
    qml.PauliX(0)
    qml.Hadamard(1)
    qml.Hadamard(2)
    qml.PauliX(3)
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
'''

@qml.qnode(dev)
def circuit():
    qml.PauliX(0)
    qml.Hadamard(2)
    qml.CNOT([0,1])
    qml.PauliZ(1)
    qml.CNOT([1,2])
    qml.PauliY(3)
    qml.CNOT([0,3])
    return qml.state()


print('---------------------------------------------')
print('THE TARGET CIRCUIT TO REPRODUCE IS:')
print(qml.draw(circuit)())


target = qml.matrix(circuit)()
#print(target)
#print(type(target))




'''
PRUEBA DE FUNCIONAMIENTO DEL CÁLCULO DE FIDELIDAD
possible_gates = ['RX', 'RY', 'RZ', 'CNOT']

dna_feat = {'input' : possible_gates,
            'number_wires' : 4,
            'number_gates_init': 5}


peasant = SimplePeasant(dna_features= dna_feat, dna =[{'target_qubit': 0, 'gate_name': 'RX', 'rotation_param': pi}, {'target_qubit': 1, 'gate_name': 'Hadamard'}, {'target_qubit': 2, 'gate_name': 'Hadamard'}, {'target_qubit': 3, 'gate_name': 'RX', 'rotation_param': pi},
                                                      {'target_qubit': 1, 'gate_name': 'CNOT', 'control_qubit': 0}, {'target_qubit': 2, 'gate_name': 'CNOT', 'control_qubit': 1}, {'target_qubit': 3, 'gate_name': 'CNOT', 'control_qubit': 2},
                                                      {'target_qubit': 3, 'gate_name': 'RZ', 'rotation_param': 0.5}, {'target_qubit': 3, 'gate_name': 'CNOT', 'control_qubit': 2}, {'target_qubit': 2, 'gate_name': 'CNOT', 'control_qubit': 1},
                                                      {'target_qubit': 1, 'gate_name': 'CNOT', 'control_qubit': 0}, {'target_qubit': 0, 'gate_name': 'RX', 'rotation_param': pi}, {'target_qubit': 1, 'gate_name': 'Hadamard'}, {'target_qubit': 2, 'gate_name': 'Hadamard'},
                                                      {'target_qubit': 3, 'gate_name': 'RX', 'rotation_param': pi}])
ErrorFunctionMatrix().evaluate_peasant(peasant,target,4)
print(peasant.fitness)
'''


'''
target = qml.numpy.tensor([0,1,1,0,1,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
print(type(target))
'''


#possible_gates = ['Identity', 'PauliY', 'PauliZ', 'PauliX', 'Hadamard','Phase', 'CNOT']
#possible_gates = ['RX', 'RZ', 'CNOT', 'Hadamard']
possible_gates = ['PauliX', 'CNOT', 'Hadamard', 'PauliZ']

dna_feat = {'input' : possible_gates,
            'number_wires' : 4,
            'number_gates_init': 2}

population_feat = {'peasant_number' : 300,               #MUST be even
                   'generations' : 100,
                   'num_surv_per_gen' : 200,             #MUST be even
                   'target_matrix' : target,
                   'mutation_probability': 0.2,
                   'mutation_probability_add': 0.2,
                   'mutation_probability_del': 0.1}


GA = GeneticAlgorithm(dna_features=dna_feat,
                      population_features=population_feat,
                      peasant=SimplePeasant,
                      fitness_function=ErrorFunctionMatrix(),
                      crossover=OnePointCrossover(),
                      mutator=OneGenMutator(possible_gates),
                      selection= RouletteWheel(),
                      print_evolution = True)

result = GA.evolve_population()


plt.plot(result[1])
plt.ylabel('Fitness')
plt.xlabel('Generation')
plt.show()

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






