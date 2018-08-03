"""
This module can run a variety of experiments that compare different Data Credit Assignment Functions (DCAF)
"""
import os
import math
import numpy as np
import pandas as pd
import sklearn.linear_model as linear_model
from sklearn.metrics import roc_auc_score, accuracy_score
import time
import argparse
import random

import yagmail
import scipy
import sklearn

import influence.experiments as experiments
from influence.nlprocessor import NLProcessor
from influence.binaryLogisticRegressionWithLBFGS import BinaryLogisticRegressionWithLBFGS
from load_spam import load_spam
from load_mnist import load_small_mnist, load_mnist


import tensorflow as tf
import csv

np.random.seed(0)

class Scenario():
    """
    One Scenario object corresponds to a single counterfactual scenario
    e.g. we are running spam classification on the enron dataset and training_example #100 does not exist
    """

    def __init__(self, task, ex_to_leave_out, num_examples=None):
        """
        init a Scenario objects.
        """
        self.task = task
        self.ex_to_leave_out = ex_to_leave_out
        self.num_examples = num_examples
        self.datasets = self.load_data_sets()
        self.init_model()


    def load_data_sets(self):
        if self.task == 'spam_enron':
            self.data_sets = load_spam(ex_to_leave_out=self.ex_to_leave_out, num_examples=self.num_examples)
        elif self.task == 'mnist':
            pass


    def init_model(self):
        """
        Initialize a tf model based on task
        """
        if self.task == 'spam_enron':
            num_classes = 2
            input_dim = self.data_sets.train.x.shape[1]
            weight_decay = 0.0001
            # weight_decay = 1000 / len(lr_data_sets.train.labels)
            batch_size = 100
            initial_learning_rate = 0.001
            keep_probs = None
            decay_epochs = [1000, 10000]
            max_lbfgs_iter = 1000

            self.model = BinaryLogisticRegressionWithLBFGS(
                input_dim=input_dim,
                weight_decay=weight_decay,
                max_lbfgs_iter=max_lbfgs_iter,
                num_classes=num_classes,
                batch_size=batch_size,
                data_sets=self.data_sets,
                initial_learning_rate=initial_learning_rate,
                keep_probs=keep_probs,
                decay_epochs=decay_epochs,
                mini_batch=False,
                train_dir='output',
                log_dir='log',
                model_name='spam_logreg'
            )
        elif self.task == 'mnist':
            num_classes = 10
            input_side = 28
            input_channels = 1
            input_dim = input_side * input_side * input_channels 
            weight_decay = 0.001
            batch_size = 500

            initial_learning_rate = 0.0001 
            decay_epochs = [10000, 20000]
            hidden1_units = 8
            hidden2_units = 8
            hidden3_units = 8
            conv_patch_size = 3
            keep_probs = [1.0, 1.0]

            model = All_CNN_C(
                input_side=input_side, 
                input_channels=input_channels,
                conv_patch_size=conv_patch_size,
                hidden1_units=hidden1_units, 
                hidden2_units=hidden2_units,
                hidden3_units=hidden3_units,
                weight_decay=weight_decay,
                num_classes=num_classes, 
                batch_size=batch_size,
                data_sets=data_sets,
                initial_learning_rate=initial_learning_rate,
                damping=1e-2,
                decay_epochs=decay_epochs,
                mini_batch=True,
                train_dir='output', 
                log_dir='log',
                model_name='mnist_small_all_cnn_c'
            )



def run_one_scenario(task, ex_to_leave_out=None, num_examples=None):
    """
    args:
        ex_to_leave_out - integer
            If ex_to_leave_out is None, don't leave any out. Otherwise, leave out the example at the specified index.
            If num_examples is None, use all the examples
        num_examples - integer
            number of examples to use
    """
    tf.reset_default_graph()

    # regardless of the choice of tasks, we must do all of the following
    # 1. load the data set into a tensorflow data objects
    # 2. choose hyperparameters like learning_rate, iterations, etc
    # 3. initalize some tensorflow model with these hyperparams

    scenario = Scenario(task, ex_to_leave_out, num_examples)

    # if task == 'spam_enron':
        
    #     data_sets = load_spam(ex_to_leave_out=ex_to_leave_out, num_examples=num_examples)
    #     num_classes = 2
    #     input_dim = data_sets.train.x.shape[1]
    #     weight_decay = 0.0001
    #     batch_size = 100
    #     initial_learning_rate = 0.001
    #     keep_probs = None
    #     decay_epochs = [1000, 10000]
    #     max_lbfgs_iter = 1000

    #     tf_model = BinaryLogisticRegressionWithLBFGS(
    #         input_dim=input_dim,
    #         weight_decay=weight_decay,
    #         max_lbfgs_iter=max_lbfgs_iter,
    #         num_classes=num_classes,
    #         batch_size=batch_size,
    #         data_sets=data_sets,
    #         initial_learning_rate=initial_learning_rate,
    #         keep_probs=keep_probs,
    #         decay_epochs=decay_epochs,
    #         mini_batch=False,
    #         train_dir='output',
    #         log_dir='log',
    #         model_name='spam_logreg')
    tf_model = scenario.model
    tf_model.train()

    # X_train = np.copy(tf_model.data_sets.train.x)
    # Y_train = np.copy(tf_model.data_sets.train.labels)
    # X_test = np.copy(tf_model.data_sets.test.x)
    Y_test = np.copy(tf_model.data_sets.test.labels)
    
    orig_results = tf_model.sess.run(
        fetches=[tf_model.loss_no_reg, tf_model.accuracy_op, tf_model.preds, tf_model.logits],
        feed_dict=tf_model.all_test_feed_dict
    )
    print('orig_results', orig_results)
    preds = orig_results[2]
    logits = orig_results[3]

    print('Y_test', Y_test[:5])
    sk_auc = roc_auc_score(y_true=Y_test, y_score=np.array(preds[:,1]))

    print('preds', preds[:5])
    sk_acc = accuracy_score(y_true=Y_test, y_pred=[1 if x[1] >= 0.5 else 0 for x in preds])

    print('orig_results: (loss and tf accuracy)\n', orig_results[0], orig_results[1])
    print('sk_acc', sk_acc)
    print('sk_auc', sk_auc)
    assert sk_acc == orig_results[1]
    result = [tf_model, orig_results]
    return result


def dcaf(model, task, test_indices, orig_loss, method='influence', num_examples=None,num_to_sample_from_train_data=None):
    """
    args:
        model - a tensorflow model
        test_indices - a list of indices corresponding to test dataset. each index in test_indices will analyzed
        orig_loss - the loss value
        method - which method to use
            method can be 'influence' (the influence function approach[DEFAULT]), 'leave-one-out'(leave-one-out approach)
            'equal' (equal-assignment approach) or 'random' (equal-assignment approach)
        num_examples - how many examples of each class to load. Set a small number to test quickly.
        num_to_sample_from_train_data - how many of the train examples to test.

    returns:
        the filepath where output data was written as CSV
    """
    model.reset_datasets()
    train_size = model.data_sets.train.num_examples
    valid_size = model.data_sets.validation.num_examples
    test_size = model.data_sets.test.num_examples
    print('============================')
    print('The training dataset has %s examples' % train_size)
    print('The validation dataset has %s examples' % valid_size)
    print('The test dataset has %s examples' % test_size)
    print("The %s method is chosen." % method)
    print('============================')

    if num_to_sample_from_train_data is not -1:
        random.seed(1)
        train_sample_index_set = random.sample(range(train_size),num_to_sample_from_train_data)

    if not os.path.isdir('csv_output'):
        os.mkdir('csv_output')

    if method == 'influence':
        # List of tuple: (index of training example, predicted loss of training example, average accuracy of training example)
        predicted_loss_diffs_per_training_point = [None] * len(train_sample_index_set)
        # Sum up the predicted loss for every training example on every test example
        # for idx in test_indices:
        #     curr_predicted_loss_diff = model.get_influence_on_test_loss(
        #         [idx], train_sample_index_set,force_refresh=True
        #     )
        #     for i in range(len(train_sample_index_set)):
        #         if predicted_loss_diffs_per_training_point[i] is None:
        #             predicted_loss_diffs_per_training_point[i] = (train_sample_index_set[i], curr_predicted_loss_diff[i])
        #         else:
        #             predicted_loss_diffs_per_training_point[i] = (train_sample_index_set[i], predicted_loss_diffs_per_training_point[i][1] + curr_predicted_loss_diff[i])
        curr_predicted_loss_diff = model.get_influence_on_test_loss(
            test_indices, train_sample_index_set,force_refresh=True
        )
        for i in range(len(train_sample_index_set)):
            predicted_loss_diffs_per_training_point[i] = (train_sample_index_set[i], curr_predicted_loss_diff[i])

        # for predicted_loss_sum_tuple in predicted_loss_diffs_per_training_point:
        #     predicted_loss_sum_tuple = (predicted_loss_sum_tuple[0],predicted_loss_sum_tuple[1]/len(test_indices))
        helpful_points = sorted(predicted_loss_diffs_per_training_point,key=lambda x: x[1], reverse=True)
        top_k = train_size
        print("If the predicted difference in loss is very positive,that means that the point helped it to be correct.")
        #print("Top %s training points making the loss on the test point better:" % top_k)
        csvdata = [["index","class","predicted_loss_diff"]]
        for i in helpful_points:        
            csvdata.append([i[0],model.data_sets.train.labels[i[0]],i[1]])
            # FLAG: error with this string print
            print("#%s, class=%s, predicted_loss_diff=%.8f" % (
                i[0],
                model.data_sets.train.labels[i[0]],
                i[1]))
        csv_filename = 'influence.csv'

    elif method == 'leave-one-out':
        print("The credit of each training example is ranked in the form of original loss - current loss.")
        print("The higher up on the ranking, the example which the leave-one-out approach tests on has a more positive influence on the model.")
        result = [None] * len(train_sample_index_set)
        for i in range(len(train_sample_index_set)):
            start1 = time.time()
            curr_results = run_one_scenario(task=task, ex_to_leave_out=train_sample_index_set[i], num_examples=num_examples)[1]
            duration1 = time.time() - start1
            print('The original LOSS is %s' % orig_loss)
            print('The current LOSS is %s' % curr_results[0])
            print('The experiment #%s took %s seconds' %(train_sample_index_set[i],duration1))
            print('======================')
            result[i] = (train_sample_index_set[i], orig_loss - curr_results[0],curr_results[1])
        result = sorted(result,key=lambda x: x[1], reverse = True)
        csvdata = [["index","class","loss_diff","accuracy"]]
        for j in result:
            csvdata.append([j[0],model.data_sets.train.labels[j[0]],j[1],j[2]])
            print("#%s,class=%s,loss_diff = %.8f, accuracy = %.8f" %(j[0], model.data_sets.train.labels[j[0]],j[1],j[2]))

        csv_filename = 'leave_one_out.csv'
        

    elif method == 'equal':
        csvdata = [["index","class","credit"]]
        for i in range(train_sample_index_set):
            csvdata.append([i,model.data_sets.train.labels[i],1/train_size])
            print("#%s,class=%s,credit = %.8f%%" %(i, model.data_sets.train.labels[i],100/train_size))

        csv_filename = 'equal.csv'

    elif method == 'random':
        result = [None] * len(train_sample_index_set)
        a = np.random.rand(train_size)
        a /= np.sum(a)
        for counter, value in enumerate(result):
            result[counter] = (counter, a[counter])
        result = sorted(result,key=lambda x: x[1], reverse = True)
        csvdata = [["index","class","credit"]]
        for i in result:
            csvdata.append([i[0],model.data_sets.train.labels[i[0]],i[1]])
            print("#%s,class=%s,credit = %.8f%%" %(i[0], model.data_sets.train.labels[i[0]],i[1]*100.00))

        csv_filename = 'random.csv'

    filepath = 'csv_output/{}'.format(csv_filename)
    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(csvdata)
    return filepath
    

def main(args):
    """
    runs the experiments based on CLI args
    """
    start_contents = """
        About to run experiments, with the following arguments: {}. 
        """.format(
            args
        )
    if not args.test:
        yag = yagmail.SMTP(os.environ['MAILBOT_ADDRESS'], os.environ['MAILBOT_PASSWORD'])
        yag.send(args.email_recipient, 'dcaf.py starting', start_contents)
    start_time = time.time()

    filepaths = []
    for task in args.tasks:
        result = run_one_scenario(task, num_examples=args.num_examples)
        model = result[0]
        orig_results = result[1]

        print('Orig loss: %.5f. Accuracy: %.3f' % (orig_results[0], orig_results[1]))
        filepath = dcaf(
            model, task, range(model.data_sets.test.num_examples), orig_loss=orig_results[0],
            method=args.method, num_to_sample_from_train_data=args.num_to_sample_from_train_data,
            num_examples=args.num_examples
        )
        filepaths.append(filepath)
    duration = round((time.time() - start_time) / 3600.0, 3)
    msg = 'Done running experiments for method {}. The DCAF function took {} hours'.format(args.method, duration)
    print(msg)
    contents = [msg] + filepaths
    if not args.test:
        yag.send(args.email_recipient, 'done with dcaf.py', contents)


def parse():
    """
    Parse CLI Args

    Here's an example
    python dcaf.py --method random --num_examples 50 --test
    """
    parser = argparse.ArgumentParser(description='see docstring')
    parser.add_argument(
        '--method', help='What method to use for computing loss. defaults to random', default='random'
    )
    parser.add_argument(
        '--test', action='store_true', help='When testing, pass this argument to suppress emails.'
    )
    parser.add_argument(
        '--num_examples', type=int, help='How many examples to use? Set this smaller to run faster.'
    )
    parser.add_argument(
        '--email_recipient', help='Who gets testing emails?',
    )
    parser.add_argument(
        '--data_dir', default='./data/',
        help='Where is the data? Default is ./data/, i.e. in the directory called "data" which lives in the same directory as this script.'
    )
    parser.add_argument(
        '--tasks', default='spam_enron', help="""
        Which task/dataset to test. Currently supports:\n
        spam_enron - WIP
        cifar - WIP
        income - WIP

        Can be a comma-separated list, e.g. \n
        "--tasks spam,cifar,income"
        """
    )
    parser.add_argument(
        '--num_to_sample_from_train_data', type=int, default=-1,
        help='Running data credit for every test example could take a LONG Time. Just do it on a sample'
    )

    args = parser.parse_args()
    if ',' in args.tasks:
        args.tasks = args.tasks.split(',')
    else:
        args.tasks = [args.tasks]
    main(args)

if __name__ == '__main__':
    parse()