import numpy as np
import pennylane as qml
def list_to_gates(gates_tuples):
    '''Takes a list of gates and transforms them into qml.gate type'''
    gate_list = []
    for i in range(len(gates_tuples)):
        if gates_tuples[i][0] == 'Identity':
            gate_list.append(qml.Identity(gates_tuples[i][1]))
        elif gates_tuples[i][0] == 'PauliX':
            gate_list.append(qml.PauliX(gates_tuples[i][1]))
        elif gates_tuples[i][0] == 'PauliY':
            gate_list.append(qml.PauliY(gates_tuples[i][1]))
        elif gates_tuples[i][0] == 'PauliZ':
            gate_list.append(qml.PauliZ(gates_tuples[i][1]))
        elif gates_tuples[i][0] == 'Hadamard':
            gate_list.append(qml.Hadamard(gates_tuples[i][1]))
        elif gates_tuples[i][0] == 'Phase':
            gate_list.append(qml.S(gates_tuples[i][1]))
        elif gates_tuples[i][0] == 'CNOT':
            gate_list.append(qml.CNOT(gates_tuples[i][1]))

    return gate_list









