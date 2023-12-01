import numpy as np
import random
from wiring import Wiring


def sigmoid(z) -> int:
    return 1 / (1 + np.exp(z * -1))


def log_loss(dataset, w) -> int:
    loss = 0
    for x in dataset:
        y = x.isDangerous
        loss = loss + ((-1 * y)*np.log((sigmoid(np.dot(x, w)))) - (1 - y)*np.log(1 - sigmoid(np.dot(x, w))))
    return (1/len(dataset)) * loss


class AlgorithmicAlchemy:
    def __init__(self):
        weights = np.zeros(400)
        for count in range(400):
            weights[count] = random.random()
        dataset = []
        for count in range(500):
            dataset.append(Wiring())
