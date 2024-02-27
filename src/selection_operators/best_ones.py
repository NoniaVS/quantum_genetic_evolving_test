import numpy as np

class BestOnes():

    def __init__(self):
        pass


    def selection(self,population, num_survivors):
        return population[:num_survivors]