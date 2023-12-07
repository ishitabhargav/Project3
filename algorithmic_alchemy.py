import numpy as np
import random
from wiring import Wiring
import matplotlib.pyplot as plt


def sigmoid(z) -> int:
    return 1 / (1 + np.exp(z * -1))


def sum_log_loss(dataset, w) -> float:
    loss = 0
    for x, y in dataset:
        sigmoid_output = sigmoid(np.dot(x, w))
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


def stochastic_gradient_descent(dataset, alpha):
    weights_length = len(dataset[0][0])  # get the length of an input vector
    weights = np.zeros(weights_length)
    for count in range(weights_length):
        weights[count] = random.uniform(0, 0.25)  # unit intervals b/w -9 to -3 work well
    time = 0
    termination = 7000
    loss_list = []
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
        updated_weights = weights - (alpha * calculate_gradient(x, y, weights) * x)
        weights = updated_weights
        time = time + 1
        print(time)
    print("loss for training dataset: " + str(sum_log_loss(dataset, weights)))
    return [weights, loss_list]


class AlgorithmicAlchemy:
    def __init__(self, training_dataset_size):
        # training dataset
        self.dataset = []
        for count in range(training_dataset_size):
            data_point = Wiring()
            vector = data_point.vector
            for i in range(len(vector)):
                noise = random.uniform(-0.1, 0.1)
                vector[i] = vector[i] + noise
            self.dataset.append((data_point.vector, data_point.is_dangerous))  # input, output pairing
        alpha = 0.2
        stochastic_gradient_output = stochastic_gradient_descent(self.dataset, alpha)
        self.weights = stochastic_gradient_output[0]
        self.loss_list = stochastic_gradient_output[1]


def main():
    # create model 1 of 500 examples for each of the training, validation, and testing sets
    model_1_size = 1000
    algorithmic_alchemy_500 = AlgorithmicAlchemy(model_1_size)
    weights_500 = algorithmic_alchemy_500.weights

    # validation dataset
    validation_set = []
    for count in range(model_1_size):
        data_point = Wiring()
        validation_set.append((data_point.vector, data_point.is_dangerous))

    # give model input and get output for data points in validation dataset

    performance_hashtable = {}
    # classify input wirings in testing dataset
    threshold_list = np.linspace(0, 1, 50)
    for threshold in threshold_list:
        performance_hashtable[threshold] = 0
    for threshold in threshold_list:
        output_500 = []
        for data_point in validation_set:
            x = data_point[0]
            sigmoid_output = sigmoid(np.dot(x, weights_500))
            if sigmoid_output < threshold:
                output_500.append(0)
            else:
                output_500.append(1)

        # calculate performance
        num_correct = 0
        for count in range(model_1_size):
            output_model = output_500[count]
            output_answer = validation_set[count][1]
            if output_answer == output_model:
                num_correct = num_correct + 1
        performance_hashtable[threshold] = num_correct / model_1_size

    # performance on training dataset
    performance_training = {}
    # classify input wirings in testing dataset
    for threshold in threshold_list:
        performance_training[threshold] = 0
    training_set = algorithmic_alchemy_500.dataset
    for threshold in threshold_list:
        output_500 = []
        for data_point in training_set:
            x = data_point[0]
            sigmoid_output = sigmoid(np.dot(x, weights_500))
            if sigmoid_output < threshold:
                output_500.append(0)
            else:
                output_500.append(1)

        # calculate performance
        num_correct = 0
        for count in range(model_1_size):
            output_model = output_500[count]
            output_answer = training_set[count][1]
            if output_answer == output_model:
                num_correct = num_correct + 1
        performance_training[threshold] = num_correct / model_1_size

    iteration_values = np.arange(start=0, stop=7000, step=1)
    y_vals_validation = []
    for item in performance_hashtable:
        y_vals_validation.append(performance_hashtable[item])
    y_vals_training = []
    for item in performance_training:
        y_vals_training.append(performance_training[item])

    for w in weights_500:
        print(w)

    print("loss for validation dataset: " + str(sum_log_loss(validation_set, weights_500)))

    plt.subplot(1, 2, 1)  # First subplot of performance and threshold values VALIDATION
    plt.plot(threshold_list, y_vals_validation, marker='o', label='Validation Set')
    plt.plot(threshold_list, y_vals_training, marker='o', label='Training Set')
    plt.xlabel("Threshold")
    plt.ylabel("Performance")
    plt.legend(loc='upper right')
    plt.title('Performance as a Function of Threshold')

    plt.subplot(1, 2, 2)  # Second subplot of loss VALIDATION
    plt.plot(iteration_values, algorithmic_alchemy_500.loss_list, marker='o')
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

    '''plt.plot(iteration_values, algorithmic_alchemy_500.loss_list, label='Line Graph')
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
    plt.plot(iteration_values, algorithmic_alchemy_500.loss_list, label='Line Graph')
    plt.xlabel("Iteration Values")
    plt.ylabel("Loss Function Values")
    plt.title("Loss Function During Training")
    plt.legend(loc='upper right')
    plt.show()
'''
