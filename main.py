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

dev = qml.device("default.qubit", wires=7)
'''
#Excitation 1
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


#Excitation 2
@qml.qnode(dev)
def circuit():
    qml.Hadamard(0)
    qml.PauliX(1)
    qml.Hadamard(3)
    qml.Hadamard(4)
    qml.PauliX(6)
    qml.CNOT([0,1])
    qml.CNOT([1, 2])
    qml.CNOT([2, 3])
    qml.CNOT([3, 4])
    qml.CNOT([4, 5])
    qml.CNOT([5, 6])
    qml.RZ(0.9,6)
    qml.CNOT([5, 6])
    qml.CNOT([4, 5])
    qml.CNOT([3, 4])
    qml.CNOT([2, 3])
    qml.CNOT([1, 2])
    qml.CNOT([0,1])
    qml.Hadamard(0)
    qml.adjoint(qml.PauliX(1))
    qml.Hadamard(3)
    qml.Hadamard(4)
    qml.adjoint(qml.PauliX(6))
    return qml.state()


'''
#Excitation 3
@qml.qnode(dev)
def circuit():
    qml.PauliX(0)
    qml.Hadamard(1)
    qml.Hadamard(2)
    qml.CNOT([0,1])
    qml.CNOT([1, 2])
    qml.RZ(0.8,2)
    qml.CNOT([1, 2])
    qml.CNOT([0,1])
    qml.adjoint(qml.PauliX(0))
    qml.Hadamard(1)
    qml.Hadamard(2)
    return qml.state()
'''





'''
@qml.qnode(dev)
def circuit():
    qml.Hadamard(0)
    qml.CNOT([0,1])
    return qml.state()
'''

print('---------------------------------------------')
print('THE TARGET CIRCUIT TO REPRODUCE IS:')
print(qml.draw(circuit)())


target = qml.matrix(circuit)()
#target = circuit()
print(target)

#print(type(target))




'''
#PRUEBA DE FUNCIONAMIENTO DEL CÁLCULO DE FIDELIDAD
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





#PRUEBA DE FUNCIONAMIENTO DEL CÁLCULO DE FIDELIDAD 2
possible_gates = ['RX', 'RY', 'RZ', 'CNOT']

dna_feat = {'input' : possible_gates,
            'number_wires' : 7,
            'number_gates_init': 2}


#peasant = SimplePeasant(dna_features= dna_feat, dna =[{'target_qubit': 3, 'gate_name': 'RX', 'rotation_param': 0.9}, {'target_qubit': 4, 'gate_name': 'CNOT', 'control_qubit': 3}, {'target_qubit': 0, 'gate_name': 'CNOT', 'control_qubit': 3}, {'target_qubit': 6, 'gate_name': 'CNOT', 'control_qubit': 1}, {'target_qubit': 0, 'gate_name': 'CNOT', 'control_qubit': 6}, {'target_qubit': 1, 'gate_name': 'CNOT', 'control_qubit': 5}])
peasant = SimplePeasant(dna_features= dna_feat, dna =[{'target_qubit': 3, 'gate_name': 'RX', 'rotation_param': 0.9}, {'target_qubit': 4, 'gate_name': 'CNOT', 'control_qubit': 3}, {'target_qubit': 0, 'gate_name': 'CNOT', 'control_qubit': 3} ])

#peasant = SimplePeasant(dna_features= dna_feat, dna =[{'target_qubit': 1, 'gate_name': 'RX', 'rotation_param': -0.8}, {'target_qubit': 2, 'gate_name': 'CNOT', 'control_qubit': 1}])


result = ErrorFunctionMatrix().evaluate_peasant(peasant,target,7)
print(peasant.fitness)
for i in range(len(result)):
    for j in range(len(target)):
        dif1 = np.abs(round(np.sum(np.abs(target[j, :] - result[:, i])),2))
        dif2 = np.abs(round(np.sum(np.abs(target[j, :] - np.conjugate(result[:, i]))),3))
        if dif1 == 0:
            print('Row', j+1, 'from target matrix matches with column', i+1, 'from equivalent circuit matrix.')
            break
        if dif2 == 0:
            print('Row', j+1, 'from target matrix matches with column', i+1, 'from equivalent circuit matrix.')
            break



exit()









'''
target = qml.numpy.tensor([0,1,1,0,1,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
print(type(target))
'''


#possible_gates = ['Identity', 'PauliY', 'PauliZ', 'PauliX', 'Hadamard','Phase', 'CNOT']
possible_gates = ['RX', 'RZ', 'CNOT', 'RY']
#possible_gates = ['PauliX', 'CNOT', 'Hadamard', 'PauliZ']

dna_feat = {'input' : possible_gates,
            'number_wires' : 7,
            'number_gates_init': 2}

population_feat = {'peasant_number' : 200,               #MUST be even
                   'generations' : 150,
                   'num_surv_per_gen' : 100,             #MUST be even
                   'target_matrix' : target,
                   'mutation_probability': 0.3,
                   'mutation_probability_add': 0.3,
                   'mutation_probability_del': 0.1}


GA = GeneticAlgorithm(dna_features=dna_feat,
                      population_features=population_feat,
                      peasant=SimplePeasant,
                      fitness_function=ErrorFunctionState(),
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






