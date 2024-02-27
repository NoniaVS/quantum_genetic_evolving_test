import numpy as np
from copy import deepcopy

class OneGenMutator():

    def __init__(self, mutation_probability, add_probability, del_probability, possible_gates):
        self.__mutation_probability = mutation_probability
        self.__add_probability = add_probability
        self.__del_probability = del_probability
        self.__possible_gates = deepcopy(possible_gates)


    def mutation_arb_change(self, _dna, num_wires):
        '''Function that can change any gate to arbitrary other gate.

        Parameters
        ----------
        _dna : list
            List of tuples which form the dna of the peasant
        num_wires : int
            Number of wires of the QC

        Returns
        -------
        list
            Mutated dna list
        '''

        r = np.random.rand()

        # There exists a probability of the mutation not happening
        if r > self.__mutation_probability:
            return _dna

        # Perform the mutation in the case it is happening
        if len(_dna) == 0:
            return _dna
        r = np.random.choice(len(_dna))

        # Fix the case where the replaced gate is a CNOT
        if _dna[r][0] == 'CNOT':
            modified_tuple = ('CNOT', _dna[r][1][0])
            _dna[r] = modified_tuple


        dna = _dna[:r] if r!= 0 else []
        new = str(np.random.choice(self.__possible_gates))     #Pick new gate from possible gates

        # Fix the case where the new gate is a CNOT by randomly choosing another target wire
        if new == 'CNOT':
            while True:
                x = np.random.choice(num_wires)
                if x != _dna[r][1]:
                    dna.append((new, [_dna[r][1],x]))
                    break
        else:
            dna.append((new, _dna[r][1]))

        dna += _dna[r+1:] if r+1 < len(_dna) else ''

        return dna



    def mutation_add_gate(self, _dna, num_wires):
        '''Function that adds a random gate with certain probability

        Parameters
        ----------
        _dna : list
            List of tuples which form the dna of the peasant
        num_wires : int
            Number of wires of the QC

        Returns
        -------
        list
            Mutated dna list
        '''

        r = np.random.rand()

        # There exists a probability of the mutation not happening
        if r > self.__add_probability:
            return _dna

        new_gate = str(np.random.choice(self.__possible_gates))
        new_wire = np.random.choice(num_wires)
        if new_gate == 'CNOT':
            while True:
                x = np.random.choice(num_wires)
                if x != new_wire:
                    _dna.append((new_gate, [new_wire, x]))
                    break
        else:
            _dna.append((new_gate, new_wire))

        return _dna


    def mutation_del_gate(self, _dna):
        '''Function that randomly deletes a gate

        Parameters
        ----------
        _dna : list
            List of tuples which form the dna of the peasant
        num_wires : int
            Number of wires of the QC

        Returns
        -------
        list
            Mutated dna list
        '''

        r = np.random.rand()

        # There exists a probability of the mutation not happening
        if r > self.__del_probability:
            return _dna
        if len(_dna) == 0:
            return _dna
        # In the case it happens remove a gate, random selection of the gate
        new_wire = np.random.choice(len(_dna))
        _dna.pop(new_wire)

        return _dna










