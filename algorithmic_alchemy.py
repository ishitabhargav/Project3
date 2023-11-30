import numpy as np
import random
from wiring import Wiring


def sigmoid(z) -> int:
    return 1 / (1 + np.exp(z * -1))




class AlgorithmicAlchemy:
    def __init__(self):
        weights = np.zeros(400)
        for count in range(400):
            weights[count] = random.random()
