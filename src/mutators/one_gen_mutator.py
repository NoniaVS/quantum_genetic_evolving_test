import numpy as np
from copy import deepcopy

class OneGenMutator():

    def __init__(self, possible_gates):
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

        # Perform the mutation in the case it is happening
        if len(_dna) == 0:
            return _dna

        r = np.random.choice(num_wires)
        new = str(np.random.choice(self.__possible_gates))  # Pick new gate from possible gates
        dna = _dna[:r] if r != 0 else []

        if new == 'CNOT':
            wire_2 = deepcopy(r)
            while wire_2 == r:
                wire_2 = np.random.choice(num_wires, 1)[0]
            dna.append({'target_qubit': r, 'gate_name': new, 'control_qubit': wire_2})
        else:
            dna.append({'target_qubit': r, 'gate_name': new, 'rotation_param': 0})

        dna += _dna[r + 1:] if r + 1 < len(_dna) else ''




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

        new_gate = str(np.random.choice(self.__possible_gates))
        new_wire = np.random.choice(num_wires)
        if new_gate == 'CNOT':
            wire_2 = new_wire
            while wire_2 == new_wire:
                wire_2 = np.random.choice(num_wires, 1)[0]
            _dna.append({'target_qubit': new_wire, 'gate_name': new_gate, 'control_qubit': wire_2})
        else:
            _dna.append({'target_qubit': new_wire, 'gate_name': new_gate, 'rotation_param': 0})

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

        if len(_dna) == 0:
            return _dna
        # In the case it happens remove a gate, random selection of the gate
        new_wire = np.random.choice(len(_dna))
        _dna.pop(new_wire)

        return _dna










