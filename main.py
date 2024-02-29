import pennylane as qml
from openfermionpyscf import run_pyscf
from src.genetic_al import GeneticAlgorithm
from src.tools import list_to_gates
from src.peasant import SimplePeasant
from src.mutators import OneGenMutator
from src.crossovers import OnePointCrossover
from src.fitness_functions import ErrorFunction
from src.selection_operators import StochasticUnivSampler, BestOnes
from openfermion import MolecularData
from pennylane import numpy as np

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
    qml.TrotterProduct(H, time = 1, order= 1).compute_decomposition
    return qml.state()
fig, ax = qml.draw_mpl(circuit)()
fig.show()
exit()





possible_gates = ['Identity', 'PauliY', 'PauliZ', 'PauliX', 'Hadamard','Phase', 'CNOT']


dna_feat = {'input' : possible_gates,
            'number_wires' : 2}

population_feat = {'peasant_number' : 100,
                   'generations' : 200,
                   'surv_prop' : 1,
                   'target_matrix' : target}


GA = GeneticAlgorithm(dna_features=dna_feat,
                      population_features=population_feat,
                      peasant=SimplePeasant,
                      fitness_function=ErrorFunction(),
                      crossover=OnePointCrossover(),
                      mutator=OneGenMutator(mutation_probability=0.7, add_probability = 0.6, del_probability = 0.4, possible_gates=possible_gates),
                      selection= StochasticUnivSampler(),
                      print_evolution = True)

result = GA.evolve_population()




dev = qml.device('default.qubit', wires = 2)
@qml.qnode(dev)
def circuit():
    qml.CNOT([0, 1])
    qml.Hadamard(1)
    return qml.expval(qml.PauliZ(0))

fig, ax = qml.draw_mpl(circuit)()
fig.show()



list_1 = list_to_gates(result)
dev = qml.device('default.qubit', wires = 2)
@qml.qnode(dev)
def circuit():
    for gates in list_1:
        qml.apply(gates)
    return qml.expval(qml.PauliZ(0))

fig, ax = qml.draw_mpl(circuit)()
fig.show()






