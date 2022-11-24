import csv
import numpy as np


class Iris:
    def __init__(self, sepal_length, sepal_width, petal_length, petal_width, species):
        self.sepal_length = sepal_length
        self.sepal_width = sepal_width
        self.petal_length = petal_length
        self.petal_width = petal_width
        self.species = species

# list of Iris data points


def parse_csv():
    with open('irisdata.csv', newline='') as f:
        reader = csv.reader(f)
        irises = []
        for row in reader:
            if row[0] != 'sepal_length':
                irises.append(Iris(row[0], row[1], row[2], row[3], row[4]))
        return irises


data = np.array(parse_csv())
print(data)

# global vars
k = 1
# mus will be an array of k Irises with their species set to mean 0-k-1
mus = np.array([])
# each cluster has its own array of elements it contains
data_k0 = np.array([])
data_k1 = np.array([])
data_k2 = np.array([])


# helper function to calculate distortion
def distortion(mu, x):
    d = 0
    d += (mu.sepal_length - x.sepal_length) ** 2
    d += (mu.sepal_width - x.sepal_width) ** 2
    d += (mu.petal_length - x.petal_length) ** 2
    d += (mu.petal_width - x.petal_width) ** 2
    return d


def place_data(x, cluster):
    if cluster == 0:
        global data_k0
        data_k0 = np.append(data_k0, x)
    if cluster == 1:
        global data_k1
        data_k1 = np.append(data_k1, x)
    if cluster == 2:
        global data_k2
        data_k2 = np.append(data_k2, x)


# the objective function both sums the total distortion and classifies the data
def classify_data():
    total = 0
    for datapt in data:
        count = 0
        lowest = 100000000
        k_class = -1
        while count < k:
            d = distortion(mus[count], datapt)
            if d < lowest:
                lowest = d
                k_class = count
            count += 1
        place_data(datapt, k_class)
        total += lowest
    return total


