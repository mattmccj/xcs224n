#!/usr/bin/env python

import sys
import os
import numpy as np
import scipy as sp
import matplotlib
import matplotlib.pyplot as plt
from utils import *
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import PCA

sys.path.append(os.path.abspath(os.path.join('..')))

def distinct_words(corpus):
    """ Determine a list of distinct words for the corpus.
        Params:
            corpus (list of list of strings): corpus of documents
        Return:
            corpus_words (list of strings): list of distinct words across the corpus, sorted (using python 'sorted' function)
            num_corpus_words (integer): number of distinct words across the corpus
    """
    corpus_words = []
    num_corpus_words = 0

    # ### START CODE HERE ###
    corpus2 = []
    for passage in range(len(corpus)):
        corpus2 = np.append(corpus2,corpus[passage])
    #corpus= np.array(corpus)
    #corpus2 = corpus.flatten()
    corpus_words = list(np.unique(corpus2))
    num_corpus_words= len(corpus_words)
    """for passage in range(len(corpus)):
        for word in range(len(corpus[passage])):
            in_corpus=False
            for i in range(len(corpus_words)):
                if corpus[passage][word]==corpus_words[i]:
                    in_corpus=True
                    break
            if in_corpus != True:
                corpus_words.append(corpus[passage][word])
                num_corpus_words = num_corpus_words+1
    
    corpus_words = corpus_words.sort()"""
    # ### END CODE HERE ###

    return corpus_words, num_corpus_words

def compute_co_occurrence_matrix(corpus, window_size=4):
    """ Compute co-occurrence matrix for the given corpus and window_size (default of 4).

        Note: Each word in a document should be at the center of a window. Words near edges will have a smaller
              number of co-occurring words.

              For example, if we take the document "START All that glitters is not gold END" with window size of 4,
              "All" will co-occur with "START", "that", "glitters", "is", and "not".

        Params:
            corpus (list of list of strings): corpus of documents
            window_size (int): size of context window
        Return:
            M (numpy matrix of shape (number of unique words in the corpus , number of unique words in the corpus)):
                Co-occurrence matrix of word counts.
                The ordering of the words in the rows/columns should be the same as the ordering of the words given by the distinct_words function.
            word2Ind (dict): dictionary that maps word to index (i.e. row/column number) for matrix M.
    """
    words, num_words = distinct_words(corpus)
    M = None
    word2Ind = {}

    # ### START CODE HERE ###
    #build index dictionary
    #word2Ind.keys = words
    
    index = []
    for i in range(len(words)):
       index.append(i)
    word2Ind = {words[i]:index[i] for i in range(num_words)}
        #word2Ind[words[i]].append(i)
        #word2Ind.update({words[i]:i})
    #word2Ind = dict(zip(words,index))

    #build the co-occurence matrix
    M = np.zeros([num_words,num_words])
    # passage iterator
    for passage in range(len(corpus)):
        #center word iterator
        for cw in range(len(corpus[passage])):
            #outer word iterator
            for ow in range(-(window_size),window_size+1):
                if (cw+ow < 0) or (ow == 0): continue
                if cw+ow >= len(corpus[passage]): break
                ##add 1 to value at M[cw,ow]
                cw_idx = word2Ind[corpus[passage][cw]]
                ow_idx = word2Ind[corpus[passage][cw+ow]]
                M[cw_idx,ow_idx] = M[cw_idx,ow_idx] + 1
    # ### END CODE HERE ###

    return M, word2Ind

def reduce_to_k_dim(M, k=2):
    """ Reduce a co-occurrence count matrix of dimensionality (num_corpus_words, num_corpus_words)
        to a matrix of dimensionality (num_corpus_words, k) using the following SVD function from Scikit-Learn:
            - http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.TruncatedSVD.html

        Params:
            M (numpy matrix of shape (number of unique words in the corpus , number of corpus words)): co-occurrence matrix of word counts
            k (int): embedding size of each word after dimension reduction
        Return:
            M_reduced (numpy matrix of shape (number of corpus words, k)): matrix of k-dimensioal word embeddings.
                    In terms of the SVD from math class, this actually returns U * S
    """
    np.random.seed(4355)
    n_iter = 10     # Use this parameter in your call to `TruncatedSVD`
    M_reduced = None
    print("Running Truncated SVD over %i words..." % (M.shape[0]))

    # ### START CODE HERE ###
    svd=TruncatedSVD(n_components=k,n_iter=n_iter,random_state=4355)
    M_reduced = svd.fit_transform(M)
    # ### END CODE HERE ###

    print("Done.")
    return M_reduced

def main():
    matplotlib.use('agg')
    plt.rcParams['figure.figsize'] = [10, 5]

    assert sys.version_info[0] == 3
    assert sys.version_info[1] >= 5

    def plot_embeddings(M_reduced, word2Ind, words, title):

        for word in words:
            idx = word2Ind[word]
            x = M_reduced[idx, 0]
            y = M_reduced[idx, 1]
            plt.scatter(x, y, marker='x', color='red')
            plt.text(x, y, word, fontsize=9)
        plt.savefig(title)

    #Read in the corpus
    reuters_corpus = read_corpus()

    M_co_occurrence, word2Ind_co_occurrence = compute_co_occurrence_matrix(reuters_corpus)
    M_reduced_co_occurrence = reduce_to_k_dim(M_co_occurrence, k=2)
    # Rescale (normalize) the rows to make them each of unit-length
    M_lengths = np.linalg.norm(M_reduced_co_occurrence, axis=1)
    M_normalized = M_reduced_co_occurrence / M_lengths[:, np.newaxis] # broadcasting

    words = ['barrels', 'bpd', 'ecuador', 'energy', 'industry', 'kuwait', 'oil', 'output', 'petroleum', 'venezuela']
    plot_embeddings(M_normalized, word2Ind_co_occurrence, words, 'co_occurrence_embeddings_(soln).png')

if __name__ == "__main__":
    main()
