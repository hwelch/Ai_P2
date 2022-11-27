import csv
import numpy as np
import random
import matplotlib.pyplot as plt
import statistics
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


def normalize_data(d):
    mean = statistics.mean(d)
    sd = statistics.stdev(d)
    return (d - mean) / sd


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
    # plt.scatter(z, a)
    # plt.ylabel('Sigmoid Output')
    # plt.xlabel('z')
    # plt.title("Normalized data")
    # plt.show()
    return a


# function for pt e to compare data and sigmoid outputs
def compare_data(dataset, outputs, index):
    plt.scatter(dataset[index][1], dataset[index][2], 150, color='maroon', label=f'output for index {index}: {outputs[index]}')
    print(f'index {index}:   petal_length = {dataset[index][1]} petal_width = {dataset[index][2]} and sigmoid output is {outputs[index]}')


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
    # normalize the data since we are just using this to calculate a probability
    data[:, 1] = normalize_data(data[:, 1])
    data[:, 2] = normalize_data(data[:, 2])
    expected_y = np.array(exp_y)
    w = np.array([[0], [0.5], [0.5]])


# function to plot an overlay of clusters on the data
# accounted for weights by setting x0 = 1 and for now b = 0
def plot_classes(title, pt, index_for_e):
    sigmoid = get_nonlinearity()
    plt.scatter(np.array(versicolor_l), np.array(versicolor_w), label="versicolor")
    plt.scatter(np.array(virginica_l), np.array(virginica_w), label="virginica")
    plt.ylabel('Petal Width')
    plt.xlabel('Petal Length')
    plt.title(title)
    plt.xlim([2.90, 7.1])
    plt.ylim(0.9, 2.58)
    if pt == 'e':
        compare_data(np.array(raw_data), sigmoid, index_for_e)
    plt.legend()
    plt.show()


initialize_params()
indexing = np.arange(100)
for i in indexing:
    plot_classes("2nd and 3rd Classes", 'e', i)
    time.sleep(1)
