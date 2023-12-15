import math

import numpy as np
import random
from dangerous_wiring import DangerousWiring
import matplotlib.pyplot as plt


def soft_max_regression(wr, wb, wy, wg, x):
    e = math.e
    denominator = e ** (np.dot(wr, x)) + e ** (np.dot(wg, x)) + e ** (np.dot(wb, x)) + e ** (np.dot(wy, x))
    frx = e ** (np.dot(wr, x)) / denominator
    fbx = e ** (np.dot(wb, x)) / denominator
    fyx = e ** (np.dot(wy, x)) / denominator
    fgx = e ** (np.dot(wg, x)) / denominator
    fx = (frx, fbx, fyx, fgx)

    return fx


def sum_log_loss(dataset, wr, wb, wy, wg) -> float:
    loss = 0
    for x, y in dataset:
        fx = soft_max_regression(wr, wb, wy, wg, x)
        frx = fx[0]
        fbx = fx[1]
        fyx = fx[2]
        fgx = fx[3]
        loss = (-1 * (y == [1, 0, 0, 0]) * math.log(frx)) + (-1 * (y == [0, 1, 0, 0]) * math.log(fbx)) + (
                -1 * (y == [0, 0, 1, 0]) * math.log(fyx)) + (-1 * (y == [0, 0, 0, 1]) * math.log(fgx))
    return (1 / len(dataset)) * loss


def stochastic_gradient_descent(dataset, alpha, testing_set):
    weights_length = len(dataset[0][0])  # get the length of an input vector
    wr = np.zeros(weights_length)
    wb = np.zeros(weights_length)
    wy = np.zeros(weights_length)
    wg = np.zeros(weights_length)
    for count in range(weights_length):
        wr[count] = random.uniform(-0.025, 0.025)
    for count in range(weights_length):
        wb[count] = random.uniform(-0.025, 0.025)
    for count in range(weights_length):
        wy[count] = random.uniform(-0.025, 0.025)
    for count in range(weights_length):
        wg[count] = random.uniform(-0.025, 0.025)
    time = 0
    termination = 80000
    loss_list = []
    test_loss_list = []
    best_weights = []
    best_test_loss = float('inf')
    time_best_test_loss = 0

    # print("starting SGD")
    while time < termination:
        # 1. pick a data point at random
        data_point = dataset[np.random.randint(0, len(dataset))]
        x = data_point[0]  # vector
        y = data_point[1]  # classification
        # 2. update weights vector
        fx = soft_max_regression(wr, wb, wy, wg, x)
        wr = wr - (alpha * (fx[0] - y[0]) * x)
        wb = wb - (alpha * (fx[1] - y[1]) * x)
        wy = wy - (alpha * (fx[2] - y[2]) * x)
        wg = wg - (alpha * (fx[3] - y[3]) * x)

        # 3. loss values
        loss_list.append(sum_log_loss(dataset, wr, wb, wy, wg))
        test_loss = sum_log_loss(testing_set, wr, wb, wy, wg)
        test_loss_list.append(test_loss)

        # 4. update best weights vectors
        if test_loss <= best_test_loss:
            best_test_loss = test_loss
            best_weights = [wr, wb, wy, wg]
            time_best_test_loss = time

        time = time + 1
        print(time)
    best_loss_list = [best_weights, best_test_loss, time_best_test_loss]
    print("loss for training set: " + str(sum_log_loss(dataset, wr, wb, wy, wg)))
    return [wr, wb, wy, wg, loss_list, test_loss_list, best_loss_list]


class AlgorithmicAlchemy2:
    def __init__(self, training_dataset_size, testing_set):
        # training training_dataset
        self.training_dataset = []
        for count in range(training_dataset_size):
            data_point = DangerousWiring()
            '''vector = data_point.vector
            for i in range(len(vector)):
                noise = random.uniform(-0.05, 0.05)
                vector[i] = vector[i] + noise'''
            self.training_dataset.append((data_point.vector, data_point.wire_to_cut))  # input, output pairing
        alpha = 0.05
        stochastic_gradient_output = stochastic_gradient_descent(self.training_dataset, alpha, testing_set)
        self.wr = stochastic_gradient_output[0]
        self.wb = stochastic_gradient_output[1]
        self.wy = stochastic_gradient_output[2]
        self.wg = stochastic_gradient_output[3]
        self.loss_list = stochastic_gradient_output[4]
        self.test_loss_list = stochastic_gradient_output[5]
        self.best_loss_list = stochastic_gradient_output[6]


def main():
    # create model 1 of 500 examples for each of the training, validation, and testing sets
    model_1_size = 5000
    validation_size = 500

    # validation training_dataset
    validation_set = []
    for count in range(validation_size):
        data_point = DangerousWiring()
        validation_set.append((data_point.vector, data_point.wire_to_cut))

    algorithmic_alchemy2_2000 = AlgorithmicAlchemy2(model_1_size, validation_set)
    wr_2000 = algorithmic_alchemy2_2000.wr
    wb_2000 = algorithmic_alchemy2_2000.wb
    wy_2000 = algorithmic_alchemy2_2000.wy
    wg_2000 = algorithmic_alchemy2_2000.wg

    # give model input and get output for data points in validation set to evaluate performance
    performance_hashtable = {}
    threshold_list = np.linspace(0, 1, 50)
    for threshold in threshold_list:
        performance_hashtable[threshold] = 0

    # evaluate performance on validation set
    num_correct_validation = 0
    for data_point in validation_set:
        x = data_point[0]
        validation_y = data_point[1]
        model_output_y = [0, 0, 0, 0]
        max_index = np.argmax(soft_max_regression(wr_2000, wb_2000, wy_2000, wg_2000, x))
        model_output_y[max_index] = 1
        if model_output_y == validation_y:
            num_correct_validation = num_correct_validation + 1
    performance_validation = num_correct_validation / validation_size

    # evaluate performance on training set
    num_correct_training = 0
    training_set = algorithmic_alchemy2_2000.training_dataset
    for data_point in training_set:
        x = data_point[0]
        validation_y = data_point[1]
        model_output_y = [0, 0, 0, 0]
        max_index = np.argmax(soft_max_regression(wr_2000, wb_2000, wy_2000, wg_2000, x))
        model_output_y[max_index] = 1
        if model_output_y == validation_y:
            num_correct_training = num_correct_training + 1
    performance_training = num_correct_training / model_1_size

    # evaluate performance of best weights on validation set
    best_list = algorithmic_alchemy2_2000.best_loss_list
    best_weights = best_list[0]
    num_correct_validation_best = 0
    for data_point in validation_set:
        x = data_point[0]
        validation_y = data_point[1]
        model_output_y = [0, 0, 0, 0]
        max_index = np.argmax(soft_max_regression(best_weights[0], best_weights[1], best_weights[2], best_weights[3], x))
        model_output_y[max_index] = 1
        if model_output_y == validation_y:
            num_correct_validation_best = num_correct_validation_best + 1
    performance_validation_best = num_correct_validation_best / validation_size

    # evaluate performance of best weights on training set
    num_correct_training_best = 0
    training_set = algorithmic_alchemy2_2000.training_dataset
    for data_point in training_set:
        x = data_point[0]
        validation_y = data_point[1]
        model_output_y = [0, 0, 0, 0]
        max_index = np.argmax(soft_max_regression(best_weights[0], best_weights[1], best_weights[2], best_weights[3], x))
        model_output_y[max_index] = 1
        if model_output_y == validation_y:
            num_correct_training_best = num_correct_training_best + 1
    performance_training_best = num_correct_training_best / model_1_size

    iteration_values = np.arange(start=0, stop=80000, step=1)
    print("loss for validation set: " + str(sum_log_loss(validation_set, wr_2000, wb_2000, wy_2000, wg_2000)))
    print("performance on validation set: " + str(performance_validation))
    print("performance on training set: " + str(performance_training))

    # print info about best weights
    print("lowest test loss: " + str(best_list[1]))
    print("training loss at time of best test loss: " + str(algorithmic_alchemy2_2000.loss_list[best_list[2]]))
    print("time lowest test loss: " + str(best_list[2]))
    print("performance on validation lowest test loss: " + str(performance_validation_best))
    print("performance on training lowest test loss: " + str(performance_training_best))

    # plot loss for training and validation sets
    plt.plot(iteration_values, algorithmic_alchemy2_2000.test_loss_list, marker='o', label='Testing Set', color='C1')
    plt.plot(iteration_values, algorithmic_alchemy2_2000.loss_list, marker='o', label='Training Set', color='C0')
    plt.xlabel("Iteration Values")
    plt.ylabel("Loss Function Values")
    plt.title("Loss Function During Training")
    plt.legend(loc='upper right')

    plt.show()



if __name__ == "__main__":
    main()


