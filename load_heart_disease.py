from influence.dataset import DataSet
import numpy as np
import io
import csv
from tensorflow.contrib.learn.python.learn.datasets import base


def process_heart_disease(ex_to_leave_out=None, num_examples=None):
    """
    Process heart_disease data
    """
    np.random.seed(0)
    train_fraction = 0.8
    valid_fraction = 0.0
    with io.open('data/heart_disease/cleveland.csv', 'r',encoding='utf-8-sig') as f:
        reader = csv.reader(f)
        example_list = list(reader)
        for counter,example in enumerate(example_list):
            for number in example:
                number = np.float64(number)


    if num_examples is not None:
        num_examples = num_examples * 2
        example_list = example_list[:num_examples]
    else:
        num_examples = len(example_list)

    attributes =[]
    Y = []
    counter = 0
    for example in example_list:
        counter += 1
        example_vector = []
        for counter,value in enumerate(example):
            if counter == len(example) - 1 :
                Y.append(int(value))
            else:
                example_vector.append(value)
        attributes.append(example_vector)

    # print(attributes[0:10])
    # print(Y)

    num_train_examples = int(train_fraction * num_examples)
    num_valid_examples = int(valid_fraction * num_examples)
    num_test_examples = num_examples - num_train_examples - num_valid_examples

    # Apply the numbers to the location in the list of documents
    attributes_train = attributes[:num_train_examples]
    Y_train = Y[:num_train_examples]

    attributes_valid = attributes[num_train_examples : num_train_examples+num_valid_examples]
    Y_valid = Y[num_train_examples : num_train_examples+num_valid_examples]

    attributes_test = attributes[-num_test_examples:]
    Y_test = Y[-num_test_examples:]

    if ex_to_leave_out is not None:
        Y_train = np.delete(Y_train, ex_to_leave_out)
        attributes_train = np.delete(attributes_train, ex_to_leave_out, axis = 0)
        number_of_elements_excluded = 1
    else:
        number_of_elements_excluded = 0
    
    #print(len(attributes_train))
    #print(len(Y_train))
    assert(len(attributes_train) == len(Y_train))
    assert(len(attributes_valid) == len(Y_valid))
    assert(len(attributes_test) == len(Y_test))
    assert(len(Y_train) + len(Y_valid) + len(Y_test) == num_examples - number_of_elements_excluded)

    return attributes_train, Y_train, attributes_valid, Y_valid, attributes_test, Y_test


    

def load_heart_disease(ex_to_leave_out=None,num_examples=None):
    X_train, Y_train, X_valid, Y_valid, X_test, Y_test = process_heart_disease(ex_to_leave_out, num_examples)
    # Convert them to dense matrices
    Y_train = np.array(Y_train)
    Y_valid = np.array(Y_valid)
    Y_test = np.array(Y_test)
    X_train = np.array(X_train)
    if X_valid is not None:
        X_valid = np.array(X_valid)
    X_test = np.array(X_test)

    train = DataSet(X_train, Y_train)
    if X_valid is not None:
        validation = DataSet(X_valid, Y_valid)
    else:
        validation = None
    test = DataSet(X_test, Y_test)
    #print(X_train[1])
    return base.Datasets(train=train, validation=validation, test=test)


    