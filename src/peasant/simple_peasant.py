import numpy as np
from copy import deepcopy


class SimplePeasant():

    def __init__(self, dna_features, dna = None):
        self.__dna_features = deepcopy(dna_features)
        self.__dna = dna if dna is not None else self.__random_initilization()
        self.__fitness = None

    @property
    def dna(self):
        return deepcopy(self.__dna)

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
        dna_inputs.remove('CNOT')
        random_dna = []


        number_wires = self.__dna_features['number_wires']
        for i in range(number_wires):
            gate = np.random.choice(dna_inputs, 1)[0]
            wire = np.random.choice(number_wires, 1)[0]
            random_dna.append((gate, wire))

        return random_dna



