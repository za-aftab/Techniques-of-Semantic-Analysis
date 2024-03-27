# UZH: Department of Computational Linguistics
# Module: Techniques of Semantic Analysis
# PA 2

# Author: Zainab Aftab

# INSTRUCTIONS
# This python script needs to run from the command line where the input textfile has to be passed as an argument.
# The command to run the code looks like this: $ python3 Sub_PA_2.py pa2_input.txt

# Please note that this script contains commented print-functions. If necessary they can be uncommented to understand
# the calculations done in the background.

import numpy as np
import re
import sys
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


####### Create Co-occurrence Matrix + List of Labels #######
filename = sys.argv[1]

# Read input file
with open(filename, 'r', encoding='utf-8') as file:
    pa2_input = file.read()

# Set pattern
pattern = r'\d+'

# Extract all counts from pa2_input
flat_counts = re.findall(pattern, pa2_input)
flat_counts = [int(count) for count in flat_counts]  # convert str to int
# print(flat_counts)  #len(counts) == 3735

nested_counts = [flat_counts[i:i + 83] for i in range(0, len(flat_counts), 83)]
# print(nested_counts)

# Convert nested_counts into matrix
matrix = np.array(nested_counts)
# print(matrix)

# Detect and classify the labels: 0 for peace and 1 for war
list_of_labels = []
match = re.findall(r'\|\t(WAR|PEACE)', pa2_input)  # match is a list!
for m in match:
    if m == 'WAR':
        list_of_labels.append(1)
    elif m == 'PEACE':
        list_of_labels.append(0)

list_of_labels = np.array(list_of_labels)
# print(list_of_labels)
############################################################


# Initialise the weights and add the bias
X = matrix.shape[1]  # better to extract them from the co-occ matrix: X.shape[1]
weights = np.zeros(X + 1)  # To add the bias of 0 to the weights we do X + 1


# Define the sigmoid function
def sigmoid(a, x):
    return 1 / (1 + np.exp(-a * x))


# Define the unit step
def unit_step(x):
    return 1.0 * (x >= 0)


# Tuple-> input: nested counts, output: list_of_labels
training_set_and = [(nested_counts[i], list_of_labels[i]) for i in range(0, len(nested_counts))]

# Compute the output
results = []
for input, desired_out in training_set_and:
    input = np.append(input, 1)  # Append 1 to account for the bias term
    result = unit_step(np.dot(input, weights) - 0.5)
    results.append((input, result, desired_out))
# print(results)

# Compute output and error
for i in range(100):
    error_count = 0

    for input, desired_out in training_set_and:
        # Compute output
        input = np.append(input, 1)  # Append 1 to account for the bias term
        result = unit_step(np.dot(input, weights) - (0.5))
        # print("input:",input, "output:",result, "true result:",desired_out)

        # Compute error
        error = desired_out - result
        # print(error)

        if abs(error) > 0.0:
            error_count += 1
            for i, val in enumerate(input):
                ## 0.2 is the learning rate which we can choose freely between 0 and 1
                # Updating the weights including the bias
                weights[i] += val * error * 0.2
    # Stopping criterion
    if error_count == 0:
        break

    # print("#"*40)
    # print(weights)
    # print("#"*40)

# Save weights in a text file
with open('weights_pa2.txt', 'w', encoding='utf-8') as output_file:
    for w in weights:
        out = str(w) + '\n'
        output_file.write(out)


#### Plotting ####
# To be able to run the function below, we first need to extract all the t-words from the input file:
t_pattern = r'\w+'
t_words = re.findall(r'^\w+', pa2_input, re.MULTILINE)  # re.MULTILINE indicates the start of line rather than a string
t_words = t_words[1:]
# print(t_words)


def pca_and_plotting(targets: list, matrix: np.ndarray) -> plt:
    """
    Given a matrix and target-words, this function will print a plot showing all target words in a 2D-space.
    The target words in the space are also labeled in the process.
    """

    dataset = matrix

    # PCA
    pca = PCA()
    pca.fit(dataset)
    plt.figure(figsize=(10, 8))
    plt.plot(pca.explained_variance_ratio_.cumsum())  # explore the number of components that contains the variance

    # Applying PCA 2 components:
    pca = PCA(n_components=2)
    pca.fit(dataset)
    pca_data = pca.transform(dataset)

    # Plotting#
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)

    xs = pca_data[:, 0]  # first component
    ys = pca_data[:, 1]  # second component

    ax.scatter(xs, ys, s=50, alpha=0.6, edgecolors='w')

    for x, y, label in zip(xs, ys, targets):
        ax.text(x, y, label)

    return plt.show()


print(pca_and_plotting(t_words, matrix))
