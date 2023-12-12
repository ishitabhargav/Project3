import math

import numpy as np
import random
from wiring import Wiring
import matplotlib.pyplot as plt

def soft_max_regression(wr, wg, wb, wy, x, y):
    e = math.e
    denominator = e**(np.dot(wr, x)) + e**(np.dot(wg, x)) + e**(np.dot(wb, x)) + e**(np.dot(wy, x))
    frx = e ** (np.dot(wr, x)) / denominator
    fgx = e ** (np.dot(wg, x)) / denominator
    fbx = e ** (np.dot(wb, x)) / denominator
    fyx = e ** (np.dot(wy, x)) / denominator
    fx = (frx, fgx, fbx, fyx)
    loss_x_y = (frx + fgx + fbx + fyx) * -1 *


def sigmoid(z) -> float:
    return 1 / (1 + np.exp(z * -1))


def sum_log_loss(dataset, w) -> float:
    loss = 0
    for x, y in dataset:
        sigmoid_output = sigmoid(np.dot(x, w))
        '''if sigmoid_output <= 0:
            print('orange')
        elif 1 - sigmoid_output <= 0:
            print('papaya')
        else:
            print("successful")'''
        loss = loss + ((-1 * y) * np.log(sigmoid_output) - (1 - y) * np.log(1 - sigmoid_output))
    return (1 / len(dataset)) * loss


def log_loss(x, y, w, time) -> float:
    dot_product = np.dot(x, w)
    sigmoid_output = sigmoid(dot_product)

    if sigmoid_output <= 0:
        print('orange')
    elif 1 - sigmoid_output <= 0:
        print('papaya')
    else:
        print("successful")
    ans = (-1 * y) * np.log(sigmoid_output) - (1 - y) * np.log(1 - sigmoid_output)

    return ans


def calculate_gradient(x, y, w) -> float:
    return sigmoid(np.dot(x, w)) - y


def stochastic_gradient_descent(dataset, alpha, testing_set):
    weights_length = len(dataset[0][0])  # get the length of an input vector
    weights = np.zeros(weights_length)
    for count in range(weights_length):
        weights[count] = random.uniform(-0.025, 0.025)  # unit intervals b/w -9 to -3 work well
    time = 0
    termination = 100000
    loss_list = []
    test_loss_list = []
    # print("starting SGD")
    while time < termination:
        # 1. pick a data point at random
        data_point = dataset[np.random.randint(0, len(dataset))]
        x = data_point[0]  # vector
        y = data_point[1]  # classification
        # 2. update weights vector
        '''loss = log_loss(x, y, weights, time)
        loss_list.append(loss)'''
        loss_list.append(sum_log_loss(dataset, weights))
        test_loss_list.append(sum_log_loss(testing_set, weights))
        updated_weights = weights - (alpha * calculate_gradient(x, y, weights) * x)
        weights = updated_weights
        time = time + 1
        print(time)
    print("loss for training training_dataset: " + str(sum_log_loss(dataset, weights)))
    return [weights, loss_list, test_loss_list]


class AlgorithmicAlchemy2:
    def __init__(self, training_dataset_size, testing_set):
        # training training_dataset
        self.training_dataset = []
        for count in range(training_dataset_size):
            data_point = Wiring()
            vector = data_point.vector
            for i in range(len(vector)):
                noise = random.uniform(-0.05, 0.05)
                vector[i] = vector[i] + noise
            self.training_dataset.append((vector, data_point.is_dangerous))  # input, output pairing
        alpha = 0.1
        stochastic_gradient_output = stochastic_gradient_descent(self.training_dataset, alpha, testing_set)
        self.weights = stochastic_gradient_output[0]
        self.loss_list = stochastic_gradient_output[1]
        self.test_loss_list = stochastic_gradient_output[2]


def main():
    # create model 1 of 500 examples for each of the training, validation, and testing sets
    model_1_size = 2000
    validation_size = 500

    # validation training_dataset
    validation_set = []
    for count in range(validation_size):
        data_point = Wiring()
        validation_set.append((data_point.vector, data_point.is_dangerous))

    algorithmic_alchemy2_2000 = AlgorithmicAlchemy2(model_1_size, validation_set)
    weights_2000 = algorithmic_alchemy2_2000.weights

    # give model input and get output for data points in validation training_dataset to judge performance
    performance_hashtable = {}
    threshold_list = np.linspace(0, 1, 50)
    for threshold in threshold_list:
        performance_hashtable[threshold] = 0
    for threshold in threshold_list:
        output_2000 = []
        for data_point in validation_set:
            x = data_point[0]
            sigmoid_output = sigmoid(np.dot(x, weights_2000))
            if sigmoid_output < threshold:
                output_2000.append(0)
            else:
                output_2000.append(1)

        # calculate performance
        num_correct = 0
        for count in range(validation_size):
            output_model = output_2000[count]
            output_answer = validation_set[count][1]
            if output_answer == output_model:
                num_correct = num_correct + 1
        performance_hashtable[threshold] = num_correct / validation_size

    # performance on training training_dataset
    performance_training = {}
    for threshold in threshold_list:
        performance_training[threshold] = 0
    training_set = algorithmic_alchemy2_2000.training_dataset
    for threshold in threshold_list:
        output_2000 = []
        for data_point in training_set:
            x = data_point[0]
            sigmoid_output = sigmoid(np.dot(x, weights_2000))
            if sigmoid_output < threshold:
                output_2000.append(0)
            else:
                output_2000.append(1)

        # calculate performance
        num_correct = 0
        for count in range(model_1_size):
            output_model = output_2000[count]
            output_answer = training_set[count][1]
            if output_answer == output_model:
                num_correct = num_correct + 1
        performance_training[threshold] = num_correct / model_1_size

    iteration_values = np.arange(start=0, stop=100000, step=1)
    y_vals_validation = []
    for item in performance_hashtable:
        y_vals_validation.append(performance_hashtable[item])
    y_vals_training = []
    for item in performance_training:
        y_vals_training.append(performance_training[item])

    for w in weights_2000:
        print(w)

    print("loss for validation set: " + str(sum_log_loss(validation_set, weights_2000)))

    plt.subplot(1, 2, 1)  # First subplot of performance and threshold values VALIDATION
    plt.plot(threshold_list, y_vals_validation, marker='o', label='Validation Set')
    plt.plot(threshold_list, y_vals_training, marker='o', label='Training Set')
    plt.xlabel("Threshold")
    plt.ylabel("Performance")
    plt.legend(loc='upper right')
    plt.title('Performance as a Function of Threshold')

    plt.subplot(1, 2, 2)  # Second subplot of loss
    plt.plot(iteration_values, algorithmic_alchemy2_2000.loss_list, marker='o', label='Training Set') #marker='o'
    plt.plot(iteration_values, algorithmic_alchemy2_2000.test_loss_list, marker='o', label='Validation Set')
    plt.xlabel("Iteration Values")
    plt.ylabel("Loss Function Values")
    plt.title("Loss Function During Training")
    plt.legend(loc='upper right')

    '''plt.subplot(1, 3, 3)  # Third subplot of performance and threshold values TRAINING
    plt.plot(threshold_list, y_vals_training, marker='o')
    plt.xlabel("Threshold")
    plt.ylabel("Performance")
    plt.title('Performance as a Function of Threshold')'''

    plt.tight_layout()

    # Display the plot
    plt.show()

    '''plt.plot(iteration_values, algorithmic_alchemy2_2000.loss_list, label='Line Graph')
    plt.xlabel("Iteration Values")
    plt.ylabel("Loss Function Values")
    plt.title("Loss Function During Training")
    plt.legend(loc='upper right')
    plt.show()'''


if __name__ == "__main__":
    main()

'''
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


    iteration_values = np.arange(start=0, stop=100000, step=1)
    plt.plot(iteration_values, algorithmic_alchemy2_500.loss_list, label='Line Graph')
    plt.xlabel("Iteration Values")
    plt.ylabel("Loss Function Values")
    plt.title("Loss Function During Training")
    plt.legend(loc='upper right')
    plt.show()
'''
