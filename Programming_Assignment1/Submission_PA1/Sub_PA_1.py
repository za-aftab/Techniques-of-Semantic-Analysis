# UZH: Techniques of Semantic Analysis
# Programming Assignment 1 (PA1)
# By: Zainab Aftab (OLAT-name: za.aftab)


# INSTRUCTIONS
# This python script needs to run from the command line where the input textfile has to be passed as an argument.
# The command to run the code looks like this: $ python3 Sub_PA_1.py <RAW TEXT> T.txt B.txt


import sys
import numpy as np
from nltk.tokenize import RegexpTokenizer
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


def preproccessing(file) -> list:
    """
    This function takes a text file as an input, separates the words by white spaces, lower-cases
    and saves them into a list called 'tokens'.
    """
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(file)
    tokens = [token.lower() for token in tokens]
    return tokens


def weighted_cooccurrence_matrix(token_list: list, targets: list, vec_basis: list) -> np.ndarray:
    """
    Here, the weighted co-occurrence matrix TxB is calculated with the lists of the preprocessed text-files.
    First, the matrix is initialised, then you iterate over the pre-proccesed text (=token_list) and count the
    co-occurrences of the context-words (=vec_basis) around the target-words (=targets) in the given vector size.
    The counts are saved into the matrix which is lastly returned.
    """
    cooccurrence_matrix = np.zeros((len(targets), len(vec_basis)))  # construct the matrix space
    target_indices = {target: index for index, target in enumerate(targets)}
    vec_indices = {vec: index for index, vec in enumerate(vec_basis)}

    for idx, token in enumerate(token_list):
        if token in targets:
            target_index = target_indices[token]
            for i in range(-2, 2):
                window_size = idx + i
                if 0 <= window_size < len(token_list):
                    context_word = token_list[window_size]
                    if context_word in vec_basis:
                        vec_index = vec_indices[context_word]
                        cooccurrence_matrix[target_index, vec_index] += 1

    return cooccurrence_matrix
   

def PPMI(cooccurrence_matrix: np.ndarray) -> np.ndarray:
    """
    The PPMI-function takes a co-occurrence matrix as a parameter and returns a matrix with PPMI-scores as weights.
    For easier interpretation of the calculation an inner-function was set up.
    """

    def compute_ppmi_value(matrix: np.ndarray, x: int, y: int) -> float:
        """
        This inner-fct is responsible for the calculation of the PPMI-values with the formulas which were derived
        from Jurasky&Martin chapter 6.
        This function is later used in a for-loop and therefore only returns one value.
        """
        f_wc = matrix[x,y]
        matrix_sum = np.sum(matrix)

        if matrix_sum == 0: # division by zero will cause errors!
            return 0
        
        p_wc = f_wc / matrix_sum
        p_w = sum(matrix[x,:]) / matrix_sum
        p_c = sum(matrix[:,y]) / matrix_sum

        formula = p_wc / (p_w * p_c) if p_w * p_c != 0 else 0  # if-statement needed to avoid ZeroDivision errors

        if formula == 0:
            # meaning of "np.finfo(float).eps": infinitely small float is replaced by a small (representable) number
            ppmi_value = max(0, np.log2(np.finfo(float).eps))
        else:
            ppmi_value = max(0, np.log2(p_wc / (p_w * p_c)))

        return ppmi_value

    # Initialise the PPMI matrix:
    PPMI_matrix = np.zeros_like(cooccurrence_matrix)  # np.zeros() gives TypeError

    # Calculate the PPMI weights:
    for i in range(cooccurrence_matrix.shape[0]):
        for j in range(cooccurrence_matrix.shape[1]):
            PPMI_matrix[i,j] = compute_ppmi_value(cooccurrence_matrix, i, j)

    return PPMI_matrix


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

    #Plotting
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)

    xs = pca_data[:, 0]  # first component
    ys = pca_data[:, 1]  # second component

    ax.scatter(xs, ys, s=50, alpha=0.6, edgecolors='w')

    for x, y, label in zip(xs, ys, targets):
        ax.text(x, y, label)

    return plt.show()


def cosine_similarity_matrix(token_list: list, targets: list, matrix: np.ndarray):
    """
    The goal of this function is to calculate the cosine matrix TxT of a given PPMI-matrix, target-words and a
    pre-proccessed text.
    The output are tuples with the following construction: '(target word, most similar word, similarity score)'
    Firstly the PPMI-matrix is changed into a sparse-matrix which is then normalised. Secondly the most similar word
    for each target word is found by finding the highest value of each row. Finally, the tuples are printed.
    """
    ppmi_matrix = PPMI(matrix)

    # Vector normalisation of the PPMI values
    denominator = np.sqrt(np.sum(np.square(ppmi_matrix), axis=1))
    normalised_ppmi = ppmi_matrix / denominator.reshape(-1, 1)

    # Create the cosine similarity matrix
    cosine_sim_matrix = cosine_similarity(normalised_ppmi)

    print('\n Target Word \t Most Similar Word \t Cosine Similarity')
    print('-------------------------------------------------------------')

    # Find the most similar words!
    most_similar_words = []
    for idx, target_word in enumerate(targets):
        # Find the target word in the row of the cosine similarity matrix
        target_row = cosine_sim_matrix[idx, :]

        # Find the column index with the highest similarity value. Target word not included!
        most_similar_index = np.argmax(target_row)
        while most_similar_index == idx:
            # If the highest similarity is with the target word itself, find the next highest similarity
            target_row[most_similar_index] = -1
            most_similar_index = np.argmax(target_row)

        # The most similar word is appended to the list
        most_similar_word = targets[most_similar_index] 
        similarity_score = cosine_sim_matrix[idx][most_similar_index]
        most_similar_words.append(most_similar_word)

        # output_tuple = (target_word, most_similar_word, similarity_score)

        print(f"{target_word: <15}\t{most_similar_word: <20}\t{similarity_score}")


def main():

    textfile = sys.argv[1]
    t_text = sys.argv[2]
    b_text = sys.argv[3]

    with open(textfile, "r", encoding="utf-8") as file:
        raw_text = file.read()  # Type of 'text' is str
        proccessed_text = preproccessing(raw_text)
        # print(proccessed_text)

    with open(t_text, "r", encoding="utf-8") as file_t, open(b_text, "r", encoding="utf-8") as file_b:
        t_words = preproccessing(file_t.read())  # output is a list
        b_words = preproccessing(file_b.read())

        x = weighted_cooccurrence_matrix(proccessed_text, t_words, b_words)
        # print(PPMI(x)) # output: class 'numpy.ndarray'
        print(cosine_similarity_matrix(proccessed_text, t_words, PPMI(x)))
        print(pca_and_plotting(t_words, x))


if __name__ == '__main__':
    main()