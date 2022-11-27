import csv
import numpy as np
import random
import matplotlib.pyplot as plt


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


def get_avg(dataset):
    arr = []
    total_sum = 0
    for d in dataset:
        total_sum += d
    arr.append(total_sum / len(dataset))
    return arr


# function to bind weights and inputs together and find sum
def summation_function():
    return np.dot(data, w)


# activation function -- in this case sigmoid function
def activation_function(z):
    return 1. / (1. + np.exp(-z))

#function to get the sigmoid nonlinearity
def get_nonlinearity():
    z = summation_function()
    print(z)
    a = activation_function(z)
    return a


# function to plot an overlay of clusters on the data
# accounted for weights by setting x0 = 1 and for now b = 0
def plot_classes(title):
    # initialize arrays to place data in
    versicolor_w = []
    versicolor_l = []
    virginica_w = []
    virginica_l = []
    updated_data = []
    exp_y = []
    global data
    global expected_y
    global w
    for d in data:
        if d.species == "versicolor":
            versicolor_l.append(d.petal_length)
            versicolor_w.append(d.petal_width)
            updated_data.append(np.array([1, d.petal_length, d.petal_width]))
            exp_y.append(0)
        if d.species == "virginica":
            virginica_l.append(d.petal_length)
            virginica_w.append(d.petal_width)
            updated_data.append(np.array([1, d.petal_length, d.petal_width]))
            exp_y.append(1)
    data = np.array(updated_data)
    expected_y = np.array(exp_y)
    w = np.array([[0], get_avg(data[:, 1]), get_avg(data[:, 2])])
    sigmoid = get_nonlinearity()
    plt.scatter(np.array(versicolor_l), np.array(versicolor_w), label="versicolor")
    plt.scatter(np.array(virginica_l), np.array(virginica_w), label="virginica")
    plt.ylabel('Petal Width')
    plt.xlabel('Petal Length')
    plt.title(title)
    plt.legend()
    plt.xlim([2.90, 7.1])
    plt.ylim(0.9, 2.58)
    plt.show()


plot_classes("2nd and 3rd Classes")