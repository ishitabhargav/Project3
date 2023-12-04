import numpy as np
import random
from wiring import Wiring
import matplotlib.pyplot as plt


def sigmoid(z) -> int:
    return 1 / (1 + np.exp(z * -1))


def sum_log_loss(dataset, w) -> int:
    loss = 0
    for x in dataset:
        y = dataset[x]
        loss = loss + ((-1 * y) * np.log((sigmoid(np.dot(x, w)))) - (1 - y) * np.log(1 - sigmoid(np.dot(x, w))))
    return (1 / len(dataset)) * loss


def log_loss(x, y, w) -> int:
    sigmoid_output = sigmoid(np.dot(x, w))
    return (-1 * y) * np.log(sigmoid_output) - (1 - y) * np.log(1 - sigmoid_output)


def calculate_gradient(x, y, w) -> int:
    return sigmoid(np.dot(x, w)) - y


def stochastic_gradient_descent(dataset, alpha):
    weights_length = len(dataset[0][0])  # get the length of an input vector
    weights = np.zeros(weights_length)
    for count in range(weights_length):
        weights[count] = random.random()
    time = 0
    termination = 1000
    loss_list = []
    while time < termination:
        # 1. pick a data point at random
        data_point = dataset[np.random.randint(0, len(dataset))]
        x = data_point[0]  # vector
        y = data_point[1]  # classification
        # 2. update weights vector
        loss = log_loss(x, y, weights)
        loss_list.append(loss)
        updated_weights = weights - (alpha * calculate_gradient(x, y, weights) * x)
        weights = updated_weights

    return [weights, loss_list]


class AlgorithmicAlchemy:
    def __init__(self, training_dataset_size):
        # training dataset
        dataset = []
        for count in range(training_dataset_size):
            data_point = Wiring()
            dataset.append((data_point.vector, data_point.is_dangerous))  # input, output pairing
        alpha = 1
        stochastic_gradient_output = stochastic_gradient_descent(dataset, alpha)
        self.weights = stochastic_gradient_output[0]
        self.loss_list = stochastic_gradient_output[1]


def main():
    # create model 1 of 500 examples for each of the training, validation, and testing sets
    model_1_size = 500
    algorithmic_alchemy_500 = AlgorithmicAlchemy(model_1_size)
    weights_500 = algorithmic_alchemy_500.weights

    # validation dataset
    validation_set = []
    for count in range(model_1_size):
        data_point = Wiring()
        validation_set.append((data_point.vector, data_point.is_dangerous))

    # give model input and get output for data points in validation dataset
    output_500 = []
    # classify input wirings in testing dataset
    for data_point in validation_set:
        x = data_point[0]
        y = data_point[1]
        sigmoid_output = sigmoid(np.dot(x, weights_500))
        if sigmoid_output < 0.5:
            output_500.append((x, 0))
        else:
            output_500.append((x, 1))

    # calculate performance
    num_correct = 0
    for count in range(len(output_500)):
        output_model = output_500[count][1]
        output_answer = validation_set[count][1]
        if output_answer == output_model:
            num_correct = num_correct + 1
    performance = num_correct / model_1_size
    print(performance)

    plt.plot(algorithmic_alchemy_500.loss_list)
    plt.show()


if __name__ == "__main__":
    main()
