"""
This module can run a variety of experiments that compare different Data Credit Assignment Functions (DCAF)
"""
import os
import math
from collections import defaultdict
from itertools import zip_longest
import json

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
from influence.logisticRegressionWithLBFGS import LogisticRegressionWithLBFGS
from influence.smooth_hinge import SmoothHinge

from load_spam import load_spam
from load_mnist import load_small_mnist, load_mnist
from load_heart_disease import load_heart_disease
from load_income import load_income
from influence.all_CNN_c import All_CNN_C

import tensorflow as tf
import csv

SEED = 0
np.random.seed(SEED)

class Scenario():
    """
    One Scenario object corresponds to a single counterfactual scenario
    e.g. we are running spam classification using Logistic Regression on the enron dataset and training example #100 does not exist
    or we are doing MNIST classification using CNN and training exampke #57 does not exist
    """

    def __init__(self, task, model_name, ex_to_leave_out, num_examples=None, data_dir='data'):
        """
        init a Scenario objects.
        """
        self.task = task
        self.model_name = model_name
        self.ex_to_leave_out = ex_to_leave_out
        self.num_examples = num_examples
        self.data_dir = data_dir
        self.datasets = self.load_data_sets()
        self.init_model()
        self.name = "{}__missing_{}__trunc_to_{}__seed_{}".format(
            task, ex_to_leave_out, num_examples, SEED
        )


    def load_data_sets(self):
        if self.task == 'spam_enron':
            self.data_sets = load_spam(ex_to_leave_out=self.ex_to_leave_out, num_examples=self.num_examples)
        elif self.task == 'small_mnist':
            self.data_sets = load_small_mnist(self.data_dir)
        elif self.task == 'mnist':
            self.data_sets = load_small_mnist('data')
        elif self.task == 'heart_disease':
            self.data_sets = load_heart_disease(ex_to_leave_out=self.ex_to_leave_out, num_examples=self.num_examples)
        elif self.task == 'income':
            self.data_sets = load_income(ex_to_leave_out=self.ex_to_leave_out, num_examples=self.num_examples)

        if 'mnist' in self.task:
            self.input_side = 28
            self.input_channels = 1
            self.input_dim = self.input_side * self.input_side * self.input_channels 
        else:
            self.input_dim = self.data_sets.train.x.shape[1]

        

    def init_model(self):
        """
        Initialize a tf model based on model_name and datasets
        """

        # TODO: make it easier to use non-default hyperparams?

        # we can always infer # classes of from the training data
        num_classes = len(set(self.data_sets.train.labels))
        model_name = self.task + '_' + self.model_name
        print('Num classes', num_classes)
        if self.model_name == 'binary_logistic':
            #num_classes = 2
            assert num_classes == 2
            weight_decay = 0.0001
            batch_size = 100
            initial_learning_rate = 0.001
            keep_probs = None
            decay_epochs = [1000, 10000]
            max_lbfgs_iter = 1000

            self.model = BinaryLogisticRegressionWithLBFGS(
                input_dim=self.input_dim,
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
                model_name=model_name
            )
        elif self.model_name == 'multi_logistic':
            #num_classes = 10
            weight_decay = 0.01
            batch_size = 1400
            initial_learning_rate = 0.001 
            keep_probs = None
            max_lbfgs_iter = 1000
            decay_epochs = [1000, 10000]

            self.model = LogisticRegressionWithLBFGS(
                input_dim=self.input_dim,
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
                model_name=model_name)

        elif self.model_name == 'cnn':
            assert num_classes == 10
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
                input_side=self.input_side,
                input_channels=self.input_channels,
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
                model_name=model_name
            )
        elif self.task == 'income':
            num_classes = 2
            input_dim = self.data_sets.train.x.shape[1]
            weight_decay = 0.0001
            # weight_decay = 1000 / len(lr_data_sets.train.labels)
            batch_size = 10
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
                model_name='income_logreg'
            )
        elif self.model_name == 'hinge_svm':
            #num_classes = 2
            weight_decay = 0.01
            use_bias = False
            batch_size = 100
            initial_learning_rate = 0.001 
            keep_probs = None
            decay_epochs = [1000, 10000]

            temps = [0, 0.001, 0.1]
            num_temps = len(temps)

            num_params = 784

            temp = 0
            self.model = SmoothHinge(
                use_bias=use_bias,
                temp=temp,
                input_dim=self.input_dim,
                weight_decay=weight_decay,
                num_classes=num_classes,
                batch_size=batch_size,
                data_sets=self.data_sets,
                initial_learning_rate=initial_learning_rate,
                keep_probs=keep_probs,
                decay_epochs=decay_epochs,
                mini_batch=False,
                train_dir='output',
                log_dir='log',
                model_name='smooth_hinge_17_t-%s' % temp)



def run_one_scenario(
         task, model_name, test_indices, ex_to_leave_out=None, num_examples=None, return_model=False, data_dir='data'
    ):
    """
    args:
        ex_to_leave_out - integer
            If ex_to_leave_out is None, don't leave any out. Otherwise, leave out the example at the specified index.
            If num_examples is None, use all the examples
        num_examples - integer
            number of examples to use
    """
    tf.reset_default_graph()
    scenario = Scenario(task, model_name, ex_to_leave_out, num_examples, data_dir)

    tf_model = scenario.model
    tf_model.train(verbose=False)

    if test_indices is None:
        test_indices = range(tf_model.data_sets.test.num_examples)

    Y_test = np.copy(tf_model.data_sets.test.labels)

    test_to_metrics = {}

    # we might want predictions broken down by test_idx
    one_at_a_time_preds = []
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
            
        if model_name == 'hinge_svm':
            one_at_a_time_preds.append(preds)
        else:
            one_at_a_time_preds.append(preds[:,1])

    loss, accuracy, preds = tf_model.sess.run(
        fetches=[tf_model.loss_no_reg, tf_model.accuracy_op, tf_model.preds],
        feed_dict=tf_model.all_test_feed_dict
    )
    # right now, this is hard coded for a 2-class problem where preds has a probablity for each class
    # so it doesn't work for SVM (the current implementation has 1 label in preds)
    if model_name == 'hinge_svm':        
        y_pred = [1 if x >= 0.5 else 0 for x in preds]
        # currently, we're not computing AUROC for SVM
        sk_auc = 0
        one_at_a_time_roc = 0
    else:
        y_pred = [1 if x[1] >= 0.5 else 0 for x in preds]
        sk_auc = roc_auc_score(y_true=Y_test, y_score=np.array(preds[:,1]))
        one_at_a_time_roc = roc_auc_score(y_true=Y_test, y_score=one_at_a_time_preds)
    
    sk_acc = accuracy_score(y_true=Y_test, y_pred=y_pred)
    assert np.isclose(sk_acc, accuracy),'{} != {}'.format(sk_acc, accuracy)
    assert np.isclose(one_at_a_time_roc, sk_auc), 'one_at_a_time_auroc ({}) != all_at_once_auroc ({})'.format(one_at_a_time_roc, sk_auc)

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
        model, model_name, task, test_indices, orig_loss, methods, num_to_sample_from_train_data=None,
        num_examples=None, per_test=None, data_dir='data'
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

    summary_dict = {}
    model.reset_datasets()
    train_size = model.data_sets.train.num_examples
    test_size = model.data_sets.test.num_examples

    summary_dict['model_name'] = model_name
    summary_dict['task'] = task
    summary_dict['num_train'] = train_size
    summary_dict['num_test'] = test_size
    summary_dict['methods'] = methods
    summary_dict['per_test'] = per_test

    prefix = 'task={}_modelname={}_'.format(task, model_name)
    if num_to_sample_from_train_data is not None:
        random.seed(1)
        train_sample_indices = random.sample(range(train_size), num_to_sample_from_train_data)
        prefix += '_trainingsamples={}'.format(num_to_sample_from_train_data)
    else:
        train_sample_indices = list(range(train_size))
        prefix += '_trainingsamples=all'
    if num_examples:
        prefix += '_numexamples={}'.format(num_examples)
    prefix += '_'

    summary_dict['prefix'] = prefix

    if not os.path.isdir('csv_output'):
        os.mkdir('csv_output')

    filepaths = []

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
            train_to_test_to_method_to_loss[train_idx]['all_at_once']['influence'] = float(curr_predicted_loss_diff[i]) # so we can print to json...

        if per_test is not None:
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
                    train_to_test_to_method_to_loss[train_idx][test_idx]['influence'] = float(one_test_loss[i]) # so we can print to json

            for train_idx, test_to_method_to_loss in train_to_test_to_method_to_loss.items():
                losses = [test_idx['influence'] for test_idx, dic in test_to_method_to_loss.items() if test_idx != 'all_at_once']
                train_to_method_to_avgloss[train_idx]['influence'] = np.mean(losses)
        

            all_at_once_errors = []
            for train_idx, loss in predicted_loss_diffs_per_training_point:
                per_test_loss = train_to_method_to_avgloss[train_idx]['influence']
                all_at_once_error = loss - per_test_loss
                all_at_once_errors.append(all_at_once_error)
                #print('loss, per_test_loss', loss, per_test_loss)
                #print('train_idx, all_at_once_error:', train_idx, all_at_once_error)
            rmse_allatonce = np.sqrt(np.mean([err ** 2 for err in all_at_once_errors]))
            print('rmse of all_at_once_errors:', rmse_allatonce)
            summary_dict['rmse_allatonce'] = rmse_allatonce
        else:
            for train_idx, test_to_method_to_loss in train_to_test_to_method_to_loss.items():
                losses = [x['influence'] for x in test_to_method_to_loss.values()]
                train_to_method_to_avgloss[train_idx]['influence'] = test_to_method_to_loss['all_at_once']['influence']

        influence_duration = time.time() - start_time
        summary_dict['influence_duration'] = influence_duration

        # sort by index to be consistent
        predicted_loss_diffs_per_training_point = sorted(predicted_loss_diffs_per_training_point, key=lambda x: x[0], reverse=True)
        # If the predicted difference in loss is very positive,that means that the point helped it to be correct.
        csvdata = [["index","class","predicted_loss_diff"]]
        for train_idx, loss in predicted_loss_diffs_per_training_point:
            csvdata.append([train_idx, model.data_sets.train.labels[train_idx], loss])

        csv_filename = prefix + 'influence.csv'
        filepath = 'csv_output/{}'.format(csv_filename)
        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(csvdata)
        filepaths.append(filepath)

        if num_to_sample_from_train_data is not None:
            estimated_total_time = influence_duration / num_to_sample_from_train_data * train_size
            summary_dict['estimated_total_influence_duration'] = estimated_total_time
            print("The estimated total time to run the entire dataset using influence method is {} seconds, which is {} hours.".format(estimated_total_time,estimated_total_time/3600))

    if 'leave-one-out' in methods or 'all' in methods:
        # The credit of each training example is ranked in the form of current loss - original loss
        start_time = time.time()

        def grouper(iterable, n, fillvalue=None):
            "Collect data into fixed-length chunks or blocks"
            # grouper('ABCDEFG', 3, 'x') --> ABC DEF Gxx"
            args = [iter(iterable)] * n
            return zip_longest(*args, fillvalue=fillvalue)

        batchsize = 32 # testing on butter suggest going from 34 -> 35 causes on BrokenProcessPool
        # use 32 to be safe
        out = []
        with Parallel(n_jobs=batchsize) as parallel:
            for batch in grouper(train_sample_indices, batchsize):
                batch = [x for x in batch if x is not None]
                print(batch)
                out += parallel(
                    delayed(run_one_scenario)(
                        task=task, model_name=model_name, test_indices=test_indices,
                        ex_to_leave_out=train_idx, num_examples=num_examples, data_dir=data_dir
                    ) for train_idx in batch
                )

        result = []
        for curr_results in out:
            curr_loss = curr_results['loss_no_reg']
            train_index_to_leave_out = curr_results['ex_to_leave_out']
            result.append(
                (train_index_to_leave_out, curr_loss - orig_loss, curr_results['accuracy'], curr_results['auc'])
            )
            for test_idx, metrics in curr_results['test_to_metrics'].items():
                train_to_test_to_method_to_loss[train_index_to_leave_out][test_idx]['leave-one-out'] =  float(metrics['loss'] - orig_loss)
        loo_duration = time.time() - start_time
        print('loo took {} seconds'.format(loo_duration))
        summary_dict['loo_duration'] = loo_duration

        # keep this just in case
        # for i, train_idx in enumerate(train_sample_indices):
        #     start1 = time.time()
        #     curr_results = run_one_scenario(task=task, test_indices=test_indices, ex_to_leave_out=i, num_examples=num_examples)
        #     curr_scenario = curr_results['scenario_obj']
        #     curr_loss = curr_results['loss_no_reg']
        #     duration1 = time.time() - start1
        #     result[i] = (train_idx, orig_loss - curr_loss, curr_results['accuracy'])
        #     for test_idx, metrics in curr_results['test_to_metrics'].items():
        #         train_to_test_to_method_to_loss[train_idx][test_idx]['leave-one-out'] =  metrics['loss'] - orig_loss

        # sort by index to be consistent
        result = sorted(result, key=lambda x: x[0], reverse=True)
        csvdata = [["index","class","loss_diff","accuracy"]]
        for j in result:
            csvdata.append([j[0], model.data_sets.train.labels[j[0]], j[1], j[2]])

        csv_filename = prefix + 'leave_one_out.csv'

        filepath = 'csv_output/{}'.format(csv_filename)
        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(csvdata)
        filepaths.append(filepath)

        for train_idx, test_to_method_to_loss in train_to_test_to_method_to_loss.items():
            losses = [x['leave-one-out'] for x in test_to_method_to_loss.values() if 'leave-one-out' in x]
            train_to_method_to_avgloss[train_idx]['leave-one-out'] = np.mean(losses)

        with open('csv_output/' + prefix + 'train_to_method_to_loss.json', 'w') as f:
            json.dump(train_to_test_to_method_to_loss, f)

        errs = []
        for train_idx, method_to_avgloss in train_to_method_to_avgloss.items():
            err = method_to_avgloss['influence'] - method_to_avgloss['leave-one-out']
            errs.append(err)

        # checks that the influence predictions and LOO results are sorted in the same manner
        assert [x[0] for x in predicted_loss_diffs_per_training_point] == [x[0] for x in result]
        pearson_corr = pearsonr([x[1] for x in result], [x[1] for x in predicted_loss_diffs_per_training_point])
        summary_dict['pearson_corr'] = pearson_corr
        print('Pearson R:', pearson_corr)

        rmse_val = np.sqrt(np.mean([err ** 2 for err in errs]))
        summary_dict['rmse_influence'] = rmse_val
        print('RMSE:', rmse_val)


        if num_to_sample_from_train_data is not None:
            estimated_total_time = loo_duration / num_to_sample_from_train_data * train_size
            summary_dict['estimated_total_loo_duration'] = estimated_total_time
            print("The estimated total time to run the entire dataset using leave-one-out method is {} seconds, which is {} hours.".format(
                estimated_total_time, estimated_total_time / 3600))

    if 'cosine_similarity' in methods or 'all' in methods:
        start_time = time.time()
        train_sample_array = []
        #train_sample_same_class_indices =[]
        for train_idx in train_sample_indices:
        #    if model.data_sets.train.labels[train_idx] == 0:
            train_sample_array.append(model.data_sets.train.x[train_idx])
        #        train_sample_same_class_indices.append(train_idx)

        #test_sample_array = []
        #for counter,example in enumerate(model.data_sets.test.x):
        #    if model.data_sets.test.labels[counter] == 0:
        #        test_sample_array.append(example)
        
        #similarities = cosine_similarity(train_sample_array, test_sample_array)
        similarities = cosine_similarity(train_sample_array, model.data_sets.test.x)
        mean_similarities = np.mean(similarities, axis=1)

        cos_duration = time.time() - start_time
        summary_dict['cosine_duration'] = cos_duration

        csvdata = [["index","class","cosine_similarity"]]
        #for counter,idx in enumerate(train_sample_same_class_indices):
            #csvdata.append([idx,model.data_sets.train.labels[idx],mean_similarities[counter]])
        
        for i in range(len(train_sample_indices)):
            csvdata.append([train_sample_indices[i],model.data_sets.train.labels[train_sample_indices[i]],mean_similarities[i]])

        csv_filename = prefix + 'cosine_similarity.csv'

        filepath = 'csv_output/{}'.format(csv_filename)
        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(csvdata)
        filepaths.append(filepath)

        if num_to_sample_from_train_data is not None:
            estimated_total_time = cos_duration/num_to_sample_from_train_data * train_size
            summary_dict['estimated_total_cosine_duration'] = estimated_total_time
            print("The estimated total time to run the entire dataset using cosine similarity method is {} seconds, which is {} hours.".format(estimated_total_time,estimated_total_time / 3600))

    if 'equal' in methods or 'all' in methods:
        start_time = time.time()
        csvdata = [["index","class","credit"]]
        for i in range(len(train_sample_indices)):
            csvdata.append([train_sample_indices[i],model.data_sets.train.labels[train_sample_indices[i]],1/train_size])
        eq_duration = time.time() - start_time
        summary_dict['equal_duration'] = eq_duration
        csv_filename = prefix + 'equal.csv'
        filepath = 'csv_output/{}'.format(csv_filename)
        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(csvdata)
        filepaths.append(filepath)
        if num_to_sample_from_train_data is not None:
            estimated_total_time = eq_duration/num_to_sample_from_train_data * train_size
            summary_dict['estimated_total_equal_duration'] = estimated_total_time
            print("The estimated total time to run the entire dataset using equal method is {} seconds, which is {} hours.".format(estimated_total_time,estimated_total_time / 3600))

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
        rand_duration = time.time() - start_time
        summary_dict['random_duration'] = rand_duration

        csv_filename = prefix + 'random.csv'
        filepath = 'csv_output/{}'.format(csv_filename)
        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(csvdata)
        filepaths.append(filepath)

        if num_to_sample_from_train_data is not None:
            estimated_total_time = rand_duration/num_to_sample_from_train_data * train_size
            summary_dict['estimated_total_random_duration'] = estimated_total_time
            print("The estimated total time to run the entire dataset using random method is {} seconds, which is {} hours.".format(
                estimated_total_time, estimated_total_time / 3600))

    with open('csv_output/' + prefix + 'summary.json', 'w') as f:
        json.dump(summary_dict, f)


    return filepaths


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
        result = run_one_scenario(task, args.model_name, test_indices=None, num_examples=args.num_examples, return_model=True, data_dir=args.data_dir)
        model = result['tf_model']
        orig_loss = result['loss_no_reg']
        orig_accuracy = result['accuracy']

        test_indices = range(model.data_sets.test.num_examples)
        print('Orig loss: %.5f. Accuracy: %.3f' % (orig_loss, orig_accuracy))
        filepaths += dcaf(
            model, args.model_name, task, test_indices=test_indices, orig_loss=orig_loss,
            methods=args.methods, num_to_sample_from_train_data=args.num_to_sample_from_train_data,
            num_examples=args.num_examples, per_test=args.calc_infl_one_test_at_a_time, data_dir=args.data_dir
        )
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
        '--calc_infl_one_test_at_a_time', help='Should we calculate influence one test at time or in bulk? Will differ slightly.',
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
        '--model_name', default='binary_logistic', help="""
        Which model to test. Currently supports:\n
        binary_logistic (working)
        multi_logistic (needs some work)
        CNN (needs some work)
        Hinge SVM

        See code for model details, like how many layers in the net, learning rate, etc
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
