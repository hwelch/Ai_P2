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
def objective_function():
    # reset the cluster before each new objective function call
    global data_k0
    global data_k1
    global data_k2
    data_k0 = np.array([])
    data_k1 = np.array([])
    data_k2 = np.array([])

    # add the total displacement for the data and place into clusters
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


# plotting function to plot the objective function as a function of iterations
def plot_displacement(displacement):
    d = np.array(displacement)
    len = d.size
    # evenly sampled time for each iteration
    iterations = np.arange(0., len, 1.0)
    plt.scatter(iterations, d)
    plt.ylabel('Iterations')
    plt.xlabel('Objective Function')
    plt.title('Function of Iterations')
    plt.show()


# function to get y for the decision boundaries i.e. perpendicular line formula at midpoint at a given point x
def calculate_decision_bounds(pt1_x, pt1_y, pt2_x, pt2_y, x):
    midpt_x = (pt2_x + pt1_x) / 2
    midpt_y = (pt1_y + pt2_y) / 2
    slope = (pt2_y - pt1_y) / (pt2_x - pt1_x)
    perp_slope = -1 / slope
    b = midpt_y - (perp_slope * midpt_x)
    return (perp_slope * x) + b


# function to plot an overlay of clusters on the data
def plot_overlay(title, plot_decision_boundaries):
    # initialize arrays to place data in
    setosa_w = []
    setosa_l = []
    versicolor_w = []
    versicolor_l = []
    virginica_w = []
    virginica_l = []
    means_w = []
    means_l = []
    means_size = []
    for d in data:
        if d.species == "setosa":
            setosa_l.append(d.petal_length)
            setosa_w.append(d.petal_width)
        if d.species == "versicolor":
            versicolor_l.append(d.petal_length)
            versicolor_w.append(d.petal_width)
        if d.species == "virginica":
            virginica_l.append(d.petal_length)
            virginica_w.append(d.petal_width)
    for mu in mus:
        means_l.append(mu.petal_length)
        means_w.append(mu.petal_width)
        means_size.append(200)
    plt.scatter(np.array(setosa_l), np.array(setosa_w), label="setosa")
    plt.scatter(np.array(versicolor_l), np.array(versicolor_w), label="versicolor")
    plt.scatter(np.array(virginica_l), np.array(virginica_w), label="virginica")
    plt.scatter(np.array(means_l), np.array(means_w), np.array(means_size), label="mu", marker=(5, 1))
    if plot_decision_boundaries:
        # sort the means
        ordered_means = sorted(mus, key=lambda x: x.petal_length, reverse=False)
        for e in ordered_means:
            print(e)
        # keep calculating lines between means while there are means there
        i = 0
        x = np.arange(0, 7, 0.01)
        while i < mus.size - 1:
            y = calculate_decision_bounds(ordered_means[i].petal_length, ordered_means[i].petal_width,
                                          ordered_means[i + 1].petal_length, ordered_means[i + 1].petal_width, x)
            plt.plot(x, y, color='black', label='Decision Bound' if i == 0 else "")
            i += 1

    plt.ylabel('Petal Width')
    plt.xlabel('Petal Length')
    plt.title(title)
    plt.legend()
    plt.xlim([0, 7.1])
    plt.ylim(0, 2.6)
    plt.show()


def calculate_kmeans(kclusters, part_to_plot):
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
    # initialize an array and append a new d every time you iterate (FOR PT B)
    distortions = []
    # breakpoint is when it converges or when difference in classify data = 0
    prev_distortion = -1
    curr_distortion = objective_function()
    distortions.append(curr_distortion)
    iteration = 1  # keep track of internal iterations for part c
    if part_to_plot == 'c':
        plot_overlay(f'k={k} Clusters: Initial', False)
    # recursively classify data and update means
    while prev_distortion != curr_distortion:
        prev_distortion = curr_distortion  # update the previous distortion
        mus = np.array(update_means())  # update the means
        curr_distortion = objective_function()  # update the current distortion
        distortions.append(curr_distortion)  # add it to the list of distortions
        if part_to_plot == 'c':
            plot_overlay(f'k={k} Clusters: Intermediate (Iteration {iteration})', False)
        iteration += 1
    if part_to_plot == 'b':
        plot_displacement(distortions)
    if part_to_plot == 'c':
        plot_overlay(f'k={k} Clusters: Converged', False)
        print(curr_distortion)
    if part_to_plot == 'd':
        plot_overlay(f'k={k} Clusters: Plot with Decision Boundaries', True)
        print(curr_distortion)


calculate_kmeans(3, 'b')
calculate_kmeans(3, 'c')
calculate_kmeans(2, 'c')
calculate_kmeans(3, 'd')
calculate_kmeans(2, 'd')

