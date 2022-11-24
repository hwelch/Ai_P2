import csv
import numpy as np
import random


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


def calculate_mean(dataset, mean_num, count):
    total_sepal_length = 0
    total_sepal_width = 0
    total_petal_length = 0
    total_petal_width = 0
    for datapt in dataset:
        total_sepal_width += datapt.sepal_width
        total_sepal_length += datapt.sepal_length
        total_petal_width += datapt.petal_width
        total_petal_length += datapt.petal_length

    total_size = float(dataset.size)
    if total_size != 0:
        avg_sepal_length = total_sepal_length / total_size
        avg_sepal_width = total_sepal_width / total_size
        avg_petal_length = total_petal_length / total_size
        avg_petal_width = total_petal_width / total_size
    else:
        avg_sepal_length = mus[count].sepal_length
        avg_sepal_width = mus[count].sepal_width
        avg_petal_length = mus[count].petal_length
        avg_petal_width = mus[count].petal_width

    return Iris(avg_sepal_length, avg_sepal_width, avg_petal_length, avg_petal_width, mean_num)


# the learning rule for the data
def update_means():
    count = 0
    updated_means = []
    while count < k:
        if count == 0:
            mu0 = calculate_mean(data_k0, "mu", count)
            updated_means.append(mu0)
        if count == 1:
            mu1 = calculate_mean(data_k1, "mu", count)
            updated_means.append(mu1)
        if count == 2:
            mu2 = calculate_mean(data_k2, "mu", count)
            updated_means.append(mu2)
        count += 1
    return updated_means


def calculate_kmeans(kclusters):
    # initiate k and mus
    global k
    k = kclusters
    count = 0
    global mus
    indexes = []
    initial_mus = []
    while count < k:
        index = random.randint(0, data.size - 1)
        if not indexes.__contains__(index):
            datapt = data[index]
            datapt.species = "mu"
            initial_mus.append(datapt)
        count += 1
    mus = initial_mus
    # initialize a np array and append a new d every time you iterate
    distortions = []
    # breakpoint is when it converges or when difference in classify data = 0
    prev_distortion = -1
    curr_distortion = classify_data()
    # recursively classify data and update means
    for m in mus:
        print(m)
    while prev_distortion != curr_distortion:
        distortions.append(curr_distortion)
        print(prev_distortion, " / ", curr_distortion)
        prev_distortion = curr_distortion
        mus = np.array(update_means())
        global data_k0
        global data_k1
        global data_k2
        data_k0 = np.array([])
        data_k1 = np.array([])
        data_k2 = np.array([])
        curr_distortion = classify_data()
    print(prev_distortion, " / ", curr_distortion)


calculate_kmeans(3)
