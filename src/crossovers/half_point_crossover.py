import numpy as np

class HalfPointCrossover():

    def __init__(self):
        pass

    def generate_dna(self, _peasant1, _peasant2):
        '''Crossover between two dnas cut in the middle of the first.
        dna_1 = [('PauliX', 0), ('Identity', 1), ('PauliY', 2)]
        dna_2 = [('PauliZ', 0), ('PauliY', 1), ('Hadamard', 2)]

        dna_3 = [('PauliX', 0),  ('PauliY', 1), ('Hadamard', 2)]

        Parameters
        ----------
        _peasant1 : Peasant type
            Peasant
        _peasant2 : Peasant type
            Peasant

        Returns
        -------
        list
            New DNA
        '''

        if len(_peasant1.dna) == 0:
            return _peasant1.dna
        r1 = (len(_peasant1.dna))//2
        r2 = (len(_peasant2.dna))//2
        dna = _peasant1.dna[:r1] + _peasant2.dna[r2:]

        return dna