from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals

import os
import math
import numpy as np
import pandas as pd
import sklearn.linear_model as linear_model
import time
import argparse

import yagmail
import scipy
import sklearn

import influence.experiments as experiments
from influence.nlprocessor import NLProcessor
from influence.binaryLogisticRegressionWithLBFGS import BinaryLogisticRegressionWithLBFGS
from load_spam import load_spam

import tensorflow as tf

import csv

np.random.seed(0)

def run_spam(ex_to_leave_out=None, num_examples=None):
    """
    If ex_to_leave_out is None, don't leave any out. Otherwise, leave out the example at the specified index.
    If num_examples is None, use all the examples
    """
    data_sets = load_spam(ex_to_leave_out=ex_to_leave_out, num_examples=num_examples)
    # "Spam" and "Ham"
    num_classes = 2

    input_dim = data_sets.train.x.shape[1]
    weight_decay = 0.0001
    # weight_decay = 1000 / len(lr_data_sets.train.labels)
    batch_size = 100
    initial_learning_rate = 0.001
    keep_probs = None
    decay_epochs = [1000, 10000]
    max_lbfgs_iter = 1000

    tf.reset_default_graph()

    tf_model = BinaryLogisticRegressionWithLBFGS(
        input_dim=input_dim,
        weight_decay=weight_decay,
        max_lbfgs_iter=max_lbfgs_iter,
        num_classes=num_classes,
        batch_size=batch_size,
        data_sets=data_sets,
        initial_learning_rate=initial_learning_rate,
        keep_probs=keep_probs,
        decay_epochs=decay_epochs,
        mini_batch=False,
        train_dir='output',
        log_dir='log',
        model_name='spam_logreg')

    tf_model.train()

    # NMV 7/26: appears to be unused right now.
    # X_train = np.copy(tf_model.data_sets.train.x)
    # Y_train = np.copy(tf_model.data_sets.train.labels)
    # X_test = np.copy(tf_model.data_sets.test.x)
    # Y_test = np.copy(tf_model.data_sets.test.labels)


    # num_train_examples = Y_train.shape[0]
    # num_flip_vals = 6
    # num_check_vals = 6
    # num_random_seeds = 40

    # dims = (num_flip_vals, num_check_vals, num_random_seeds, 3)
    # fixed_influence_loo_results = np.zeros(dims)
    # fixed_loss_results = np.zeros(dims)
    # fixed_random_results = np.zeros(dims)

    #flipped_results = np.zeros((num_flip_vals, num_random_seeds, 3))

    orig_results = tf_model.sess.run(
        [tf_model.loss_no_reg, tf_model.accuracy_op],
        feed_dict=tf_model.all_test_feed_dict)
    #print('Orig loss: %.5f. Accuracy: %.3f' % (orig_results[0], orig_results[1]))
    result = [tf_model,orig_results]
    return result


def dcaf(model, test_indices, orig_loss, method='influence', num_examples=None):
    # method can be 'influence' (the influence function approach[DEFAULT]), 'leave-one-out'(leave-one-out approach)
    # 'equal' (equal-assignment approach) or 'random' (equal-assignment approach)
    model.reset_datasets()
    train_size = model.data_sets.train.num_examples
    valid_size = model.data_sets.validation.num_examples
    test_size = model.data_sets.test.num_examples
    # Implemented by Tensorflow
    # Datasets = collections.namedtuple('Datasets', ['train', 'validation', 'test'])
    print('============================')
    print('The training dataset has %s examples' % train_size)
    print('The validation dataset has %s examples' % valid_size)
    print('The test dataset has %s examples' % test_size)
    print("The %s method is chosen." %method)
    print('============================')

    if not os.path.isdir('csv_output'):
        os.mkdir('csv_output')

    if method == 'influence':
        indices_to_remove = np.arange(1)
        # List of tuple: (index of training example, predicted loss of training example, average accuracy of training example)
        predicted_loss_diffs_per_training_point = [None] * train_size
        # Sum up the predicted loss for every training example on every test example
        for idx in test_indices:
            curr_predicted_loss_diff = model.get_influence_on_test_loss(
                [idx], indices_to_remove,force_refresh=True
            )
            for train_idx in range(train_size):
                if predicted_loss_diffs_per_training_point[train_idx] is None:
                    predicted_loss_diffs_per_training_point[train_idx] = (train_idx, curr_predicted_loss_diff[train_idx])
                else:
                    predicted_loss_diffs_per_training_point[train_idx] = (train_idx, predicted_loss_diffs_per_training_point[train_idx][1] + curr_predicted_loss_diff[train_idx])

        for predicted_loss_sum_tuple in predicted_loss_diffs_per_training_point:
            predicted_loss_sum_tuple = (predicted_loss_sum_tuple[0],predicted_loss_sum_tuple[1]/len(test_indices))

        helpful_points = sorted(predicted_loss_diffs_per_training_point,key=lambda x: x[1], reverse=True)
        top_k = train_size
        print("If the predicted difference in loss is very positive,that means that the point helped it to be correct.")
        print("Top %s training points making the loss on the test point better:" % top_k)
        csvdata = [["index","class","predicted_loss_diff"]]
        for i in helpful_points:
            csvdata.append([i[0],model.data_sets.train.labels[i[0]],i[1]])
            # FLAG: error with this string print
            print("#%s, class=%s, predicted_loss_diff=%.8f, accuracy=%.8f" % (
                i[0],
                model.data_sets.train.labels[i[0]],
                i[1]))

        csv_filename = 'influence.csv'

    elif method == 'leave-one-out':
        print("The credit of each training example is ranked in the form of original loss - current loss.")
        print("The higher up on the ranking, the example which the leave-one-out approach tests on has a more positive influence on the model.")
        result = [None] * train_size
        for i in range(train_size):
            start1 = time.time()
            curr_results = run_spam(i, num_examples=num_examples)[1]
            duration1 = time.time() - start1
            print('The original LOSS is %s' % orig_loss)
            print('The current LOSS is %s' % curr_results[0])
            print('The experiment #%s took %s seconds' %(i,duration1))
            print('======================')
            result[i] = (i, orig_loss - curr_results[0],curr_results[1])
        result = sorted(result,key=lambda x: x[1], reverse = True)
        csvdata = [["index","class","loss_diff","accuracy"]]
        for j in result:
            csvdata.append([j[0],model.data_sets.train.labels[j[0]],j[1],j[2]])
            print("#%s,class=%s,loss_diff = %.8f, accuracy = %.8f" %(j[0], model.data_sets.train.labels[j[0]],j[1],j[2]))

        csv_filename = 'leave_one_out.csv'
        

    elif method == 'equal':
        csvdata = [["index","class","credit"]]
        print("\\\\\\\\\\ The credits sum up to 1. //////////")
        for i in range(train_size):
            csvdata.append([i,model.data_sets.train.labels[i],1/train_size])
            print("#%s,class=%s,credit = %.8f%%" %(i, model.data_sets.train.labels[i],100/train_size))

        csv_filename = 'equal.csv'

    elif method == 'random':
        print("\\\\\\\\\\ The credits sum up to 1. //////////")
        result = [None] * train_size
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
    if not args.test:
        yag = yagmail.SMTP(os.environ['MAILBOT_ADDRESS'], os.environ['MAILBOT_PASSWORD'])
        start_contents = """
            About to run run_spam experiments, with the following arguments: {}. 
            Another email will sent upon a successful run.
            """.format(
                args
            )
        yag.send(args.email_recipient, 'run_spam_experiment.py', start_contents)
    start_time = time.time()
    result = run_spam(num_examples=args.num_examples)
    model=result[0]
    orig_results = result[1]
    print('Orig loss: %.5f. Accuracy: %.3f' % (orig_results[0], orig_results[1]))
    filepath = dcaf(
        model, range(model.data_sets.test.num_examples), orig_loss=orig_results[0], method=args.method, num_examples=args.num_examples
    )
    duration = (time.time() - start_time)/3600.0
    msg = 'Done running experiments for method {}. The DCAF function took {} hours'.format(args.method, duration)
    print(msg)
    contents = [msg, filepath]
    if not args.test:
        yag.send(args.email_recipient, 'done with run_spam_experiment.py', contents)

    # NMV 7/25: as far as I can see, this output/spam_results is not currently being used by us
    # np.savez(
    #     'output/spam_results',
    #     orig_results=orig_results,
    #     #flipped_results=flipped_results,
    #     #fixed_influence_loo_results=fixed_influence_loo_results,
    #     #fixed_loss_results=fixed_loss_results,
    #     #fixed_random_results=fixed_random_results
    # )

def parse():
    """
    Parse CLI Args
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
        '--email_recipient', help='who gets testing emails',
    )
    args = parser.parse_args()
    main(args)

if __name__ == '__main__':
    parse()