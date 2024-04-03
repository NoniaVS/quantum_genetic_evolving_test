import numpy as np
import pennylane as qml
def list_to_gates(gates_tuples):
    '''Takes a list of gates and transforms them into qml.gate type'''
    gate_list = []
    for i in range(len(gates_tuples)):
        if gates_tuples[i]['gate_name'] == 'Identity':
            gate_list.append(qml.Identity(gates_tuples[i]['target_qubit']))
        elif gates_tuples[i]['gate_name'] == 'PauliX':
            gate_list.append(qml.PauliX(gates_tuples[i]['target_qubit']))
        elif gates_tuples[i]['gate_name'] == 'PauliY':
            gate_list.append(qml.PauliY(gates_tuples[i]['target_qubit']))
        elif gates_tuples[i]['gate_name'] == 'PauliZ':
            gate_list.append(qml.PauliZ(gates_tuples[i]['target_qubit']))
        elif gates_tuples[i]['gate_name'] == 'Hadamard':
            gate_list.append(qml.Hadamard(gates_tuples[i]['target_qubit']))
        elif gates_tuples[i]['gate_name'] == 'Phase':
            gate_list.append(qml.S(gates_tuples[i]['target_qubit']))
        elif gates_tuples[i]['gate_name'] == 'CNOT':
            gate_list.append(qml.CNOT(wires=[gates_tuples[i]['control_qubit'],gates_tuples[i]['target_qubit']]))
        elif gates_tuples[i]['gate_name'] == 'RX':
            gate_list.append(qml.RX(gates_tuples[i]['rotation_param'], gates_tuples[i]['target_qubit']))
        elif gates_tuples[i]['gate_name'] == 'RY':
            gate_list.append(qml.RY(gates_tuples[i]['rotation_param'], gates_tuples[i]['target_qubit']))
        elif gates_tuples[i]['gate_name'] == 'RZ':
            gate_list.append(qml.RZ(gates_tuples[i]['rotation_param'], gates_tuples[i]['target_qubit']))

    return gate_list


def function_to_minimize(params, _peasant, target_state, num_wires):

    #Colocamos los parámetros
    dna = _peasant.dna
    count = 0
    for gene in dna:
        if gene['gate_name'] != 'CNOT':
            gene['rotation_param'] = params[count]
            count += 1
    gates = list_to_gates(dna)
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
    #drawer = qml.draw(circuit)
    #print(drawer())

    circuit_state = circuit()
    # print(circuit_state)
    error = (np.abs(np.dot(np.conj(target_state), circuit_state))) ** 2
    _peasant.fitness = error

    return 1 - _peasant.fitness



def function_to_minimize_matrix(params, _peasant, target_matrix, num_wires):

    #Colocamos los parámetros
    dna = _peasant.dna
    count = 0
    for gene in dna:
        if gene['gate_name'] != 'CNOT':
            gene['rotation_param'] = params[count]
            count += 1
    gates = list_to_gates(dna)
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
    #drawer = qml.draw(circuit)
    #print(drawer())
    circuit_matrix = qml.matrix(circuit)()  # circuit_matrix is a numpy.ndarray
    error = np.sum(np.abs(circuit_matrix - target_matrix))
    _peasant.fitness = 1 / (1 + error)
    return 1 - _peasant.fitness










