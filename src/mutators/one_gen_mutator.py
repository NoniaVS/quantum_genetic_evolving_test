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

        r = np.random.choice(len(_dna))
        dna = _dna[:r] if r != 0 else []

        #r = np.random.choice(num_wires)
        if _dna[r]['gate_name'] == 'CNOT':
            possible_gates = deepcopy(self.__possible_gates)
            possible_gates.remove('CNOT')
            new_1 = str(np.random.choice(possible_gates))  # Pick new gate from possible gates
            new_2 = str(np.random.choice(possible_gates))  # Pick new gate from possible gates
            old_wire1 = _dna[r]['target_qubit']
            old_wire2 = _dna[r]['control_qubit']
            dna.append({'target_qubit': old_wire1, 'gate_name': new_1, 'rotation_param': 0})
            dna.append({'target_qubit': old_wire2, 'gate_name': new_2, 'rotation_param': 0})
        else:
            new = str(np.random.choice(self.__possible_gates))
            if new == 'CNOT':
                wire_2 = deepcopy(_dna[r]['target_qubit'])
                while wire_2 == _dna[r]['target_qubit']:
                    wire_2 = np.random.choice(num_wires, 1)[0]
                dna.append({'target_qubit': _dna[r]['target_qubit'], 'gate_name': new, 'control_qubit': wire_2})
            else:
                dna.append({'target_qubit': _dna[r]['target_qubit'], 'gate_name': new, 'rotation_param': 0})

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










