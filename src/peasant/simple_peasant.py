import numpy as np
from copy import deepcopy
import pennylane as qml
from src.tools import list_to_gates
from math import pi


class SimplePeasant():

    def __init__(self, dna_features, dna = None):
        self.__dna_features = deepcopy(dna_features)
        self.__dna = dna if dna is not None else self.__random_initilization()
        self.__fitness = None

    @property
    def dna(self):
        return deepcopy(self.__dna)

    @dna.setter
    def dna(self, value):
        self.__dna = value

    @property
    def fitness(self):
        return deepcopy(self.__fitness)

    @fitness.setter
    def fitness(self, value):
        self.__fitness = value

    @property
    def dna_features(self):
        return deepcopy(self.__dna_features)


    def __random_initilization(self):
        '''Function to initialize the SimplePeasant with random DNA

        Returns
        -------
        list
            List containing the tuples defining the DNA
        '''


        dna_inputs = self.__dna_features['input']
        number_gates_init = self.__dna_features['number_gates_init']
        number_wires = self.__dna_features['number_wires']
        random_dna = []
        for i in range(number_gates_init):
            gate = np.random.choice(dna_inputs, 1)[0]
            wire = np.random.choice(number_wires, 1)[0]
            if gate == 'CNOT':
                wire_2 = deepcopy(wire)
                while wire_2 == wire:
                    wire_2 = np.random.choice(number_wires, 1)[0]
                random_dna.append({'target_qubit': wire, 'gate_name': gate, 'control_qubit': wire_2})
            else:
                random_phase = np.random.rand() *2*pi
                random_dna.append({'target_qubit': wire, 'gate_name': gate, 'rotation_param': random_phase})
        '''
        gates = list_to_gates(random_dna)
        dev = qml.device("default.qubit", wires=6)

        @qml.qnode(dev)
        def circuit():
            for i in range(6):
                qml.Identity(wires=i)
            for gate in gates:
                qml.apply(gate)
            return qml.state()

        target = circuit()
        print(target)
        '''
        return random_dna



