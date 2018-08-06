"""
This module can run a variety of experiments that compare different Data Credit Assignment Functions (DCAF)
"""
import os
import math
from collections import defaultdict


import numpy as np
import pandas as pd
import sklearn.linear_model as linear_model
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import pearsonr
import time
import argparse
import random

import yagmail
from joblib import Parallel, delayed

import influence.experiments as experiments
from influence.nlprocessor import NLProcessor
from influence.binaryLogisticRegressionWithLBFGS import BinaryLogisticRegressionWithLBFGS
from load_spam import load_spam
from load_mnist import load_small_mnist, load_mnist
from influence.all_CNN_c import All_CNN_C



import tensorflow as tf
import csv

SEED = 0
np.random.seed(SEED)

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
        self.name = "{}__missing_{}__trunc_to_{}__seed_{}".format(
            task, ex_to_leave_out, num_examples, SEED
        )


    def load_data_sets(self):
        if self.task == 'spam_enron':
            self.data_sets = load_spam(ex_to_leave_out=self.ex_to_leave_out, num_examples=self.num_examples)
        elif self.task == 'mnist':
            self.data_sets = load_small_mnist('data')


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

            self.model = All_CNN_C(
                input_side=input_side,
                input_channels=input_channels,
                conv_patch_size=conv_patch_size,
                hidden1_units=hidden1_units,
                hidden2_units=hidden2_units,
                hidden3_units=hidden3_units,
                weight_decay=weight_decay,
                num_classes=num_classes,
                batch_size=batch_size,
                data_sets=self.data_sets,
                initial_learning_rate=initial_learning_rate,
                damping=1e-2,
                decay_epochs=decay_epochs,
                mini_batch=True,
                train_dir='output',
                log_dir='log',
                model_name='mnist_small_all_cnn_c'
            )



def run_one_scenario(task, test_indices, ex_to_leave_out=None, num_examples=None, return_model=False):
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

    tf_model = scenario.model
    tf_model.train()

    if test_indices is None:
        test_indices = range(tf_model.data_sets.test.num_examples)

    # X_train = np.copy(tf_model.data_sets.train.x)
    # Y_train = np.copy(tf_model.data_sets.train.labels)
    # X_test = np.copy(tf_model.data_sets.test.x)
    Y_test = np.copy(tf_model.data_sets.test.labels)

    test_to_metrics = {}

    # confusing to readers
    all_one_preds = []
    for test_idx in test_indices:
        test_feed_dict = tf_model.fill_feed_dict_with_one_ex(
            tf_model.data_sets.test,
            test_idx
        )
        loss, accuracy, preds = tf_model.sess.run(
            fetches=[tf_model.loss_no_reg, tf_model.accuracy_op, tf_model.preds],
            feed_dict=test_feed_dict
        )
        test_to_metrics[test_idx] = {
            'loss': loss, 'accuracy': accuracy, 'preds': preds
        }
        all_one_preds.append(preds[:,1])

    loss, accuracy, preds = tf_model.sess.run(
        fetches=[tf_model.loss_no_reg, tf_model.accuracy_op, tf_model.preds],
        feed_dict=tf_model.all_test_feed_dict
    )

    sk_auc = roc_auc_score(y_true=Y_test, y_score=np.array(preds[:,1]))
    sk_acc = accuracy_score(y_true=Y_test, y_pred=[1 if x[1] >= 0.5 else 0 for x in preds])
    print('results: (loss and tf accuracy)\n', loss, accuracy)
    print('sk_auc', sk_auc)
    assert sk_acc == accuracy
    assert roc_auc_score(y_true=Y_test, y_score=all_one_preds) == sk_auc

    mean_loss = np.mean([test_to_metrics[x]['loss'] for x in test_to_metrics.keys()])
    assert np.isclose(mean_loss, loss)
    ret = {
        #'tf_model': tf_model,
        'loss_no_reg': loss,
        'accuracy': sk_acc,
        'auc': sk_auc,
        #'scenario_obj': scenario,
        'test_to_metrics': test_to_metrics,
        'ex_to_leave_out': ex_to_leave_out
    }
    # can't pickle tf model
    if return_model:
        ret['tf_model'] = tf_model
    return ret


def dcaf(
        model, task, test_indices, orig_loss, methods, num_to_sample_from_train_data=None,
        num_examples=None, per_test=True,
    ):
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
        per_test - if true, pass in one individual test example to predict at a time

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
    print("The %s methods are chosen." % methods)
    print('============================')

    if num_to_sample_from_train_data is not None:
        random.seed(1)
        train_sample_indices = random.sample(range(train_size), num_to_sample_from_train_data)

    if not os.path.isdir('csv_output'):
        os.mkdir('csv_output')

    train_to_test_to_method_to_loss = defaultdict(lambda: defaultdict(dict))
    train_to_method_to_avgloss = defaultdict(dict)

    if 'influence' in methods or 'all' in methods:
        # List of tuple: (index of training example, predicted loss of training example, average accuracy of training example)
        start_time = time.time()
        predicted_loss_diffs_per_training_point = [None] * len(train_sample_indices)

        curr_predicted_loss_diff = model.get_influence_on_test_loss(
            test_indices=test_indices, train_indices=train_sample_indices, force_refresh=True
        )
        for i, train_idx in enumerate(train_sample_indices):
            predicted_loss_diffs_per_training_point[i] = (train_idx, curr_predicted_loss_diff[i])
            train_to_test_to_method_to_loss[train_idx]['all_at_once']['influence'] = curr_predicted_loss_diff[i]

        if per_test:
            # could parallelize here?
            # doesn't work: _pickle.PicklingError: Could not pickle the task to send it to the workers.
            # def helper(test_idx, train_sample_indices_indices):
            #     return model.get_influence_on_test_loss(test_indices=[test_idx], train_indices=train_sample_indices, force_refresh=True), test_idx

            # for one_test_loss, test_idx in Parallel(n_jobs=-1, verbose=5)(
            #     delayed(helper)(
            #         test_idx, train_sample_indices
            #     ) for test_idx in test_indices
            # ):
            #     for i, train_idx in enumerate(train_sample_indices):
            #         train_to_test_to_method_to_loss[train_idx][test_idx]['influence'] = one_test_loss[i]

            for test_idx in test_indices:
                one_test_loss = model.get_influence_on_test_loss(
                    test_indices=[test_idx], train_indices=train_sample_indices, force_refresh=True
                )
                for i, train_idx in enumerate(train_sample_indices):
                    train_to_test_to_method_to_loss[train_idx][test_idx]['influence'] = one_test_loss[i]

        for train_idx, test_to_method_to_loss in train_to_test_to_method_to_loss.items():
            losses = [x['influence'] for x in test_to_method_to_loss.values()]
            train_to_method_to_avgloss[train_idx]['influence'] = np.mean(losses)
        
        # TODO: quantify error

        all_at_once_errors = []
        for train_idx, loss in predicted_loss_diffs_per_training_point:
            per_test_loss = train_to_method_to_avgloss[train_idx]['influence']
            all_at_once_error = loss - per_test_loss
            all_at_once_errors.append(all_at_once_error)
            print('loss, per_test_loss', loss, per_test_loss)
            print('train_idx, all_at_once_error:', train_idx, all_at_once_error)
        print('rmse of all_at_once_errors')
        print(np.sqrt(np.mean([err ** 2 for err in all_at_once_errors])))

        influence_duration = time.time() - start_time


        predicted_loss_diffs_per_training_point = sorted(predicted_loss_diffs_per_training_point, key=lambda x: x[0], reverse=True)
        #print("If the predicted difference in loss is very positive,that means that the point helped it to be correct.")
        csvdata = [["index","class","predicted_loss_diff"]]
        for train_idx, loss in predicted_loss_diffs_per_training_point:
            csvdata.append([train_idx, model.data_sets.train.labels[train_idx], loss])
            print("#{}, label={}, predicted_loss_diff={}".format(
                train_idx,
                model.data_sets.train.labels[train_idx],
                loss
            ))
        csv_filename = 'influence_' + str(num_to_sample_from_train_data) + '.csv'
        filepath = 'csv_output/{}'.format(csv_filename)
        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(csvdata)

        if num_to_sample_from_train_data is not None:
            estimated_total_time = influence_duration/num_to_sample_from_train_data * train_size
            print("The estimated total time to run the entire dataset using influence method is {} seconds, which is {} hours.".format(estimated_total_time,estimated_total_time/3600))

    if 'leave-one-out' in methods or 'all' in methods:
        print("The credit of each training example is ranked in the form of original loss - current loss.")
        print("The higher up on the ranking, the example which the leave-one-out approach tests on has a more positive influence on the model.")
        start_time = time.time()
        out = Parallel(n_jobs=-1)(
            delayed(run_one_scenario)(
                task=task, test_indices=test_indices, ex_to_leave_out=train_idx, num_examples=num_examples
            ) for train_idx in train_sample_indices
        )
        result = []

        for curr_results in out:
            # curr_scenario = curr_results['scenario_obj']
            curr_loss = curr_results['loss_no_reg']
            train_index_to_leave_out = curr_results['ex_to_leave_out']
            print('The original LOSS is %s' % orig_loss)
            print('The current LOSS is %s' % curr_loss)
            print('======================')
            result.append(
                (train_index_to_leave_out, curr_loss - orig_loss, curr_results['accuracy'])
            )
            for test_idx, metrics in curr_results['test_to_metrics'].items():
                train_to_test_to_method_to_loss[train_index_to_leave_out][test_idx]['leave-one-out'] =  metrics['loss'] - orig_loss
        loo_duration = time.time() - start_time
        print('All experiments took {}'.format(loo_duration))


        # for i, train_idx in enumerate(train_sample_indices):
        #     start1 = time.time()
        #     curr_results = run_one_scenario(task=task, test_indices=test_indices, ex_to_leave_out=i, num_examples=num_examples)
        #     curr_scenario = curr_results['scenario_obj']
        #     curr_loss = curr_results['loss_no_reg']
        #     duration1 = time.time() - start1
        #     print('The original LOSS is %s' % orig_loss)
        #     print('The current LOSS is %s' % curr_loss)
        #     print('The experiment #%s took %s seconds' % (i, duration1))
        #     print('======================')
        #     result[i] = (train_idx, orig_loss - curr_loss, curr_results['accuracy'])
        #     for test_idx, metrics in curr_results['test_to_metrics'].items():
        #         train_to_test_to_method_to_loss[train_idx][test_idx]['leave-one-out'] =  metrics['loss'] - orig_loss

        # sorts by index
        result = sorted(result, key=lambda x: x[0], reverse=True)
        csvdata = [["index","class","loss_diff","accuracy"]]
        for j in result:
            csvdata.append([j[0], model.data_sets.train.labels[j[0]], j[1], j[2]])
            print("#%s,class=%s,loss_diff = %.8f, accuracy = %.8f" %(j[0], model.data_sets.train.labels[j[0]],j[1],j[2]))

        csv_filename = 'leave_one_out_' + str(num_to_sample_from_train_data) + '.csv'

        filepath = 'csv_output/{}'.format(csv_filename)
        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(csvdata)

        for train_idx, test_to_method_to_loss in train_to_test_to_method_to_loss.items():
            #print('train_idx', train_idx)
            #print(test_to_method_to_loss)
            losses = [x['leave-one-out'] for x in test_to_method_to_loss.values() if 'leave-one-out' in x]
            train_to_method_to_avgloss[train_idx]['leave-one-out'] = np.mean(losses)

        # TODO: save train_to_method_to_avgloss to json or csv

        errs = []
        for train_idx, method_to_avgloss in train_to_method_to_avgloss.items():
            print(train_idx)
            err = method_to_avgloss['influence'] - method_to_avgloss['leave-one-out']
            print('Infl: {}. LOO: {}. Error: {}.'.format(
                method_to_avgloss['influence'], method_to_avgloss['leave-one-out'],
                err
            ))
            errs.append(err)

        print([x[0] for x in predicted_loss_diffs_per_training_point])
        print([x[0] for x in result])
        print('Pearson R')
        print(pearsonr([x[1] for x in result], [x[1] for x in predicted_loss_diffs_per_training_point]))

        rmse_val = np.sqrt(np.mean([err ** 2 for err in errs]))
        print('RMSE')
        print(rmse_val)

        print('Average all_at_once_error for influence function')
        print(np.mean(all_at_once_errors))
        
        if num_to_sample_from_train_data is not None:
            estimated_total_time = loo_duration/num_to_sample_from_train_data * train_size
            print("The estimated total time to run the entire dataset using leave-one-out method is {} seconds, which is {} hours.".format(estimated_total_time,estimated_total_time/3600))

    if 'cosine_similarity' in methods or 'all' in methods:
        start_time = time.time()
        train_sample_array = []
        for train_idx in train_sample_indices:
            # if model.data_sets.train.labels[train_idx] == 1:
            train_sample_array.append(model.data_sets.train.x[train_idx])

        # test_sample_array = []
        # for counter,example in enumerate(model.data_sets.test.x):
        #     if model.data_sets.test.labels[counter] == 1:
        #         test_sample_array.append(example)
        #
        # similarities = cosine_similarity(train_sample_array, test_sample_array)
        similarities = cosine_similarity(train_sample_array, model.data_sets.test.x)
        mean_similarities = np.mean(similarities, axis=1)

        cos_duration = time.time() - start_time

        csvdata = [["index","class","cosine_similarity"]]
        for i in range(len(train_sample_indices)):
            csvdata.append([train_sample_indices[i],model.data_sets.train.labels[train_sample_indices[i]],mean_similarities[i]])
            print("#{},class={},avg_cosine_similarity={}".format(train_sample_indices[i],model.data_sets.train.labels[train_sample_indices[i]],mean_similarities[i]))

        csv_filename = 'cosine_similarity_' + str(num_to_sample_from_train_data) + '.csv'

        filepath = 'csv_output/{}'.format(csv_filename)
        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(csvdata)

        if num_to_sample_from_train_data is not None:
            estimated_total_time = cos_duration/num_to_sample_from_train_data * train_size
            print("The estimated total time to run the entire dataset using cosine similarity method is {} seconds, which is {} hours.".format(estimated_total_time,estimated_total_time/3600))

    if 'equal' in methods or 'all' in methods:
        start_time = time.time()
        csvdata = [["index","class","credit"]]
        for i in range(len(train_sample_indices)):
            csvdata.append([train_sample_indices[i],model.data_sets.train.labels[train_sample_indices[i]],1/train_size])
            print("#%s,class=%s,credit = %.8f%%" %(i, model.data_sets.train.labels[train_sample_indices[i]],100/train_size))
        eq_duration = time.time() - start_time
        csv_filename = 'equal_' + str(num_to_sample_from_train_data) + '.csv'
        filepath = 'csv_output/{}'.format(csv_filename)
        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(csvdata)
        if num_to_sample_from_train_data is not None:
            estimated_total_time = eq_duration/num_to_sample_from_train_data * train_size
            print("The estimated total time to run the entire dataset using equal method is {} seconds, which is {} hours.".format(estimated_total_time,estimated_total_time/3600))

    if 'random' in methods or 'all' in methods:
        start_time = time.time()
        result = [None] * len(train_sample_indices)
        a = np.random.rand(train_size)
        a /= np.sum(a)
        for counter, _ in enumerate(result):
            result[counter] = (counter, a[counter])
        result = sorted(result,key=lambda x: x[0], reverse = True)
        csvdata = [["index","class","credit"]]
        for i in result:
            csvdata.append([i[0],model.data_sets.train.labels[i[0]],i[1]])
            print("#%s,class=%s,credit = %.8f%%" %(i[0], model.data_sets.train.labels[i[0]],i[1]*100.00))
        rand_duration = time.time() - start_time

        csv_filename = 'random_' + str(num_to_sample_from_train_data) + '.csv'
        filepath = 'csv_output/{}'.format(csv_filename)
        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(csvdata)

        if num_to_sample_from_train_data is not None:
            estimated_total_time = rand_duration/num_to_sample_from_train_data * train_size
            print("The estimated total time to run the entire dataset using random method is {} seconds, which is {} hours.".format(estimated_total_time, estimated_total_time/3600))


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
        result = run_one_scenario(task, test_indices=None, num_examples=args.num_examples, return_model=True)
        model = result['tf_model']
        orig_loss = result['loss_no_reg']
        orig_accuracy = result['accuracy']

        test_indices = range(model.data_sets.test.num_examples)
        print('Orig loss: %.5f. Accuracy: %.3f' % (orig_loss, orig_accuracy))
        filepath = dcaf(
            model, task, test_indices=test_indices, orig_loss=orig_loss,
            methods=args.methods, num_to_sample_from_train_data=args.num_to_sample_from_train_data,
            num_examples=args.num_examples
        )
        filepaths.append(filepath)
    duration = round((time.time() - start_time) / 3600.0, 3)
    msg = 'Done running experiments for methods {}. The DCAF function took {} hours'.format(args.methods, duration)
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
        '--methods', help='What methods to use for computing loss. defaults to all', default='influence,leave-one-out'
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
        '--num_to_sample_from_train_data', type=int,
        help='Running data credit for every test example could take a LONG Time. Just do it on a sample'
    )

    args = parser.parse_args()
    if ',' in args.tasks:
        args.tasks = args.tasks.split(',')
    else:
        args.tasks = [args.tasks]

    if ',' in args.methods:
        args.methods = args.methods.split(',')
    else:
        args.methods = [args.methods]
    main(args)

if __name__ == '__main__':
    parse()
