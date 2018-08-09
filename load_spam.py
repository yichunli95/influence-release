import os

from influence.dataset import DataSet
from tensorflow.contrib.learn.python.learn.datasets import base
import numpy as np

from influence.nlprocessor import NLProcessor

from scipy.io import savemat

import IPython

def init_lists(folder):
    a_list = []
    file_list = os.listdir(folder)
    for a_file in file_list:
        f = open(folder + a_file, 'r', encoding="ISO-8859-1")
        a_list.append(f.read())
    f.close()
    return a_list


def process_spam(ex_to_leave_out=None, num_examples=None):
    """
    Process the spam/ham data
    """
    np.random.seed(0)

    nlprocessor = NLProcessor()
    spam = init_lists('data/spam/enron1/spam/')
    ham = init_lists('data/spam/enron1/ham/')
    if num_examples:
        # take care of it if num_examples is too big
        num_examples = min(num_examples, len(spam), len(ham))
        spam = spam[:num_examples]
        ham = ham[:num_examples]

    docs, Y = nlprocessor.process_spam(spam, ham)
    num_examples = len(Y)

    # The number of documents used for training, validation and tests
    train_fraction = 0.8
    valid_fraction = 0.0
    num_train_examples = int(train_fraction * num_examples)
    num_valid_examples = int(valid_fraction * num_examples)
    num_test_examples = num_examples - num_train_examples - num_valid_examples

    # Apply the numbers to the location in the list of documents
    docs_train = docs[:num_train_examples]
    Y_train = Y[:num_train_examples]

    docs_valid = docs[num_train_examples : num_train_examples+num_valid_examples]
    Y_valid = Y[num_train_examples : num_train_examples+num_valid_examples]

    docs_test = docs[-num_test_examples:]
    Y_test = Y[-num_test_examples:]

    if ex_to_leave_out is not None:
        Y_train = np.delete(Y_train, ex_to_leave_out)
        docs_train = np.delete(np.array(docs_train), ex_to_leave_out)
        number_of_elements_excluded = 1
    else:
        number_of_elements_excluded = 0


    print('Based on provided data and CLI args, there are {} train examples and {} test examples'.format(
        len(docs_train), len(docs_test)
    ))

    assert(len(docs_train) == len(Y_train))
    assert(len(docs_valid) == len(Y_valid))
    assert(len(docs_test) == len(Y_test))
    assert(len(Y_train) + len(Y_valid) + len(Y_test) == num_examples - number_of_elements_excluded)

    # Learn vocab (transform the documents into a dictionary of words appeared in the docs)
    nlprocessor.learn_vocab(docs_train)
    # Bag of Words matrices for each division of the docs, freqs of each word
    X_train = nlprocessor.get_bag_of_words(docs_train)
    if docs_valid:
        X_valid = nlprocessor.get_bag_of_words(docs_valid)
    else: 
        X_valid = None
    X_test = nlprocessor.get_bag_of_words(docs_test)

    return X_train, Y_train, X_valid, Y_valid, X_test, Y_test


def load_spam(ex_to_leave_out=None, num_examples=None):

    X_train, Y_train, X_valid, Y_valid, X_test, Y_test = process_spam(ex_to_leave_out, num_examples)

    # Convert them to dense matrices
    X_train = X_train.toarray()
    if X_valid is not None:
        X_valid = X_valid.toarray()
    X_test = X_test.toarray()

    train = DataSet(X_train, Y_train)
    if X_valid is not None:
        validation = DataSet(X_valid, Y_valid)
    else:
        validation = None
    test = DataSet(X_test, Y_test)
    #print(X_train[1])
    return base.Datasets(train=train, validation=validation, test=test)
