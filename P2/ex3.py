import csv
import numpy as np
import random
import matplotlib.pyplot as plt
import time

class Iris:
    def __init__(self, sepal_length, sepal_width, petal_length, petal_width, species):
        self.sepal_length = sepal_length
        self.sepal_width = sepal_width
        self.petal_length = petal_length
        self.petal_width = petal_width
        self.species = species

    def __str__(self):
        return f'sl: {self.sepal_length} sw: {self.sepal_width} pl: {self.petal_length} pw: {self.petal_width} species: {self.species}'

# list of Iris data points


def parse_csv():
    with open('irisdata.csv', newline='') as f:
        reader = csv.reader(f)
        irises = []
        for row in reader:
            if row[0] != 'sepal_length':
                irises.append(Iris(float(row[0]), float(row[1]), float(row[2]), float(row[3]), row[4]))
        return irises

# global vars
data = np.array(parse_csv())
expected_y = np.array([])
k = 2
w = np.array([])
b = 0
versicolor_w = []
versicolor_l = []
virginica_w = []
virginica_l = []
raw_data = []


# function to bind weights and inputs together and find sum
def summation_function():
    return np.dot(data, w)


# activation function -- in this case sigmoid function
def activation_function(z):
    return 1. / (1. + np.exp(-z))


# function to get the sigmoid nonlinearity
def get_nonlinearity():
    z = summation_function()
    a = activation_function(z)
    return a


# function for pt e to compare data and sigmoid outputs
def compare_data(dataset, outputs, index):
    plt.scatter(dataset[index][1], dataset[index][2], 150, color='maroon', label=f'output for index {index}: {outputs[index]}')


# function to plot approximate decision boundary based on weights
def get_decision_boundary(x1, w0, w1, w2):
    return -((w1 * x1 + w0) / w2)


# initialize parameters
def initialize_params():
    # initialize arrays to place data in
    global versicolor_w, versicolor_l, virginica_w, virginica_l, raw_data, data, expected_y, w
    versicolor_w = []
    versicolor_l = []
    virginica_w = []
    virginica_l = []
    raw_data = []
    exp_y = []
    for d in data:
        if d.species == "versicolor":
            versicolor_l.append(d.petal_length)
            versicolor_w.append(d.petal_width)
            raw_data.append(np.array([1, d.petal_length, d.petal_width]))
            exp_y.append(0)
        if d.species == "virginica":
            virginica_l.append(d.petal_length)
            virginica_w.append(d.petal_width)
            raw_data.append(np.array([1, d.petal_length, d.petal_width]))
            exp_y.append(1)
    data = np.array(raw_data)
    expected_y = np.array(exp_y)
    w = np.array([[-8.8], [1.1], [2]])


def plot_classes(title, w0, w1, w2, w0b, w1b, w2b, l1, l2):
    plt.scatter(np.array(versicolor_l), np.array(versicolor_w), label="versicolor")
    plt.scatter(np.array(virginica_l), np.array(virginica_w), label="virginica")
    plt.ylabel('Petal Width')
    plt.xlabel('Petal Length')
    plt.title(title)
    plt.xlim([2.90, 7.1])
    plt.ylim(0.9, 2.58)
    x_axis = np.arange(2.9, 7.1, 0.1)
    plt.plot(x_axis, get_decision_boundary(x_axis, w0, w1, w2), color='black', label=l1)
    plt.plot(x_axis, get_decision_boundary(x_axis, w0b, w1b, w2b), color='red', label=l2)
    plt.legend()
    plt.show()


# get the mean-squared error
def get_mean_squared(data_vectors, weights, pattern_classes):
    global data, w
    data = data_vectors
    w = weights
    summation = 0
    y_actual = get_nonlinearity()
    count = 0
    while count < 100:
        summation += (y_actual[count] - pattern_classes[count]) ** 2
        count += 1
    return summation / 2


def compute_for_weights():
    print("low error:")
    print("w0 = -8.8, w1 = 1.1, w2 = 2")
    w0 = -8.8
    w1 = 1.1
    w2 = 2
    print(get_mean_squared(data, np.array([[w0], [w1], [w2]]), expected_y))
    print("high error:")
    print("w0 = 2 w1 = 5 w2 = -18.5")
    w0b = 2
    w1b = 5
    w2b = -18.5
    print(get_mean_squared(data, np.array([[w0b], [w1b], [w2b]]), expected_y))
    plot_classes("Data with Error Graphs", w0, w1, w2, w0b, w1b, w2b, "Low Error", "High Error")


# function to get the summed gradient for an ensemble of patterns
def get_gradient(data_vectors, pattern_class):
    a = get_nonlinearity()
    y_exp = pattern_class
    ones = np.ones(100)
    unsummed_gradient = []
    count = 0
    while count < 100:
        unsummed_gradient.append(data_vectors[count] * a[count] * (ones[count] - a[count]) * (a[count] - y_exp[count]))
        count += 1
    gradient = np.sum(np.array(unsummed_gradient), axis=0)
    return gradient


# function to update the weights based on the gradient
def update_weights(data_vectors, weights, pattern_class, epsilon):
    dw = get_gradient(data_vectors, pattern_class)
    dw = np.array([[dw[0]], [dw[1]], [dw[2]]])
    return weights - dw * epsilon


# function to compare the updated weights and previous weights
def plot_gradient():
    w_old = np.array([[-8.8], [1.1], [2]])
    w_new = update_weights(data, w_old, expected_y, 0.005)
    plot_classes("Illustrate Gradient", w_old[0], w_old[1], w_old[2], w_new[0], w_new[1], w_new[2], "Old Weights", "New Weights")


initialize_params()
compute_for_weights()
plot_gradient()
