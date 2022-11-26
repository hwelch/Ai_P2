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


data = np.array(parse_csv())

# global vars
k = 1

# function to plot an overlay of clusters on the data
def plot_classes(title):
    # initialize arrays to place data in
    versicolor_w = []
    versicolor_l = []
    virginica_w = []
    virginica_l = []
    updated_data = []
    global data
    for d in data:
        if d.species == "versicolor":
            versicolor_l.append(d.petal_length)
            versicolor_w.append(d.petal_width)
            updated_data.append(d)
        if d.species == "virginica":
            virginica_l.append(d.petal_length)
            virginica_w.append(d.petal_width)
            updated_data.append(d)
    data = updated_data
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
