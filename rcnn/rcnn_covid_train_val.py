"""
- load our data (tabels of MERS sars-cov-1 sars-cov-2 high confidence)
- transfer learning - train on sars-cov-1 and mers with freezing weights
- evaluate on sars-cov-2 (look at inference.py and analysis.py)
- load and save trained models
- get some results on val and test for both tasks
- write up results
"""

from seq2tensor import s2t

import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Dropout, Embedding, LSTM, Bidirectional, BatchNormalization, merge, add
from keras.layers.core import Flatten, Reshape
from keras.layers.merge import Concatenate, concatenate, subtract, multiply
from keras.layers.convolutional import Conv1D
from keras.layers.pooling import MaxPooling1D, AveragePooling1D, GlobalAveragePooling1D
from keras.layers import Input, CuDNNGRU
from keras.optimizers import Adam, RMSprop
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, average_precision_score
import sys
from tqdm import tqdm
from numpy import linalg as LA
import scipy
import numpy as np
import os
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF
from utils import *
from models import *
import argparse
import matplotlib.pyplot as plt
from utils import plot_train_history, plot_learning_results

swissprot_file = '../data/swissprot-proteome.tab'
cov_prot_file = '../data/new/cov_protein_sequences.csv'


def get_trainable_indexes(indexes_str):
    indexes_couples = [index_couple.split('-') for index_couple in indexes_str.split(',')]
    indexes = [(int(index_couple[0]), int(index_couple[1])) for index_couple in indexes_couples]
    return indexes


def is_in_range(ranges, num):
    for r in ranges:
        a, b = r
        if a <= num <= b:
            return True
    return False


def get_h_and_v_proteins(interactions_file, vp_set, hp_set):
    positives = set()
    vp2numPos = {}
    with open(interactions_file, 'r') as f:
        next(f)
        for line in f:
            items = line.strip().split(',')
            vp_key = items[0] + items[1].upper()
            hp = items[2]
            if vp_key not in vp_set or hp not in hp_set:
                continue
            positives.add((hp, vp_key))
            if vp_key not in vp2numPos:
                vp2numPos[vp_key] = 0
            vp2numPos[vp_key] += 1
    return list(positives), vp2numPos


def get_non_frozen_indexes(indexes_str):
    indexes = [int(index_str) for index_str in indexes_str.split(',')]
    return indexes


def fit_model(model, model_num, epochs, batch_size, steps, score_threshold,
              train_gen, val_gen, test_gen, triple_test):
    history = model.fit_generator(generator=train_gen,
                                  epochs=epochs,
                                  steps_per_epoch=steps,
                                  verbose=2,
                                  max_queue_size=50,
                                  use_multiprocessing=False,
                                  workers=1,
                                  validation_data=val_gen)
    plot_train_history(history.history['accuracy'],
                   history.history['val_accuracy'],
                   history.history['loss'],
                   history.history['val_loss'], model_num)

    # test
    y_score = model.predict_generator(generator=test_gen, verbose=2,
                                          steps=int(np.ceil(len(triple_test) / batch_size)),
                                          max_queue_size=50, workers=1)
    y_true = np.array([int(example[-1]) for example in triple_test])

    test_acc = accuracy_score(y_true, (y_score > score_threshold).astype(int))
    test_auc = roc_auc_score(y_true, y_score)
    print('Test ROCAUC: %.3f, acc: %.3f' % (test_auc, test_acc))


def fine_tune_rcnn(model_file, model_num, epochs, batch_size, steps, score_threshold, option,
                   trainable_layers, seq_size,
                   hp_set, prot2embed,
                   train_positives, test_positives, train_vps, test_vps, log_each_epoch):
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.3
    sess = tf.Session(config=config)
    K.set_session(sess)
    counter = 0
    K.clear_session()
    triple_train, triple_val, triple_test = get_triples_without_family(train_positives, test_positives, hp_set,
                                                                       train_vps,
                                                                       test_vps)
    print("Number of triples:")
    print("\ttrain:", len(triple_train))
    print("\tvalidation:", len(triple_val))
    print("\ttest:", len(triple_test))

    model = load_model(model_file)

    trainable_layers_names = []
    # Set non trainable layers.
    for i in range(len(model.layers)):
        if is_in_range(trainable_layers, i):
            trainable_layers_names.append(model.layers[i].name)
            continue
        l = model.layers[i]
        l.trainable = False
    print('\ttrainable layers names:', trainable_layers_names)

    adam = Adam(lr=0.001, amsgrad=True, epsilon=1e-6)
    model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])
    train_gen, val_gen, test_gen = get_generators(triple_train, triple_val, triple_test, batch_size,
                                                  prot2embed, option, MAXLEN=seq_size)

    if not log_each_epoch:
        return fit_model(model, model_num, epochs, batch_size, steps, score_threshold,
              train_gen, val_gen, test_gen, triple_test)

    train_acc, train_loss = [], []
    validation_acc, validation_loss, val_acc, val_auc = [], [], [], []
    test_acc, test_auc = [], []
    for i in range(epochs):
        # We want to print the test accuracy on each epoch.
        print('taxon ', counter, ' epoch ', i)
        history = model.fit_generator(generator=train_gen,
                                      epochs=1,
                                      steps_per_epoch=steps,
                                      verbose=2,
                                      max_queue_size=50,
                                      use_multiprocessing=False,
                                      workers=1,
                                      validation_data=val_gen)

        # compute metrics on validation data
        y_score = model.predict_generator(generator=val_gen, verbose=2,
                                          steps=int(np.ceil(len(triple_val) / batch_size)),
                                          max_queue_size=50, workers=1)
        y_true = np.array([int(example[-1]) for example in triple_val])

        val_acc.append(accuracy_score(y_true, (y_score > score_threshold).astype(int)))
        val_auc.append(roc_auc_score(y_true, y_score))
        print('Validation ROCAUC: %.3f, acc: %.3f' % (val_auc[-1], val_acc[-1]))

        # test
        y_score = model.predict_generator(generator=test_gen, verbose=2,
                                          steps=int(np.ceil(len(triple_test) / batch_size)),
                                          max_queue_size=50, workers=1)
        y_true = np.array([int(example[-1]) for example in triple_test])

        test_acc.append(accuracy_score(y_true, (y_score > score_threshold).astype(int)))
        test_auc.append(roc_auc_score(y_true, y_score))
        print('Test ROCAUC: %.3f, acc: %.3f' % (test_auc[-1], test_acc[-1]))

        train_acc += history.history["accuracy"]
        train_loss += history.history["loss"]
        validation_acc += history.history["val_accuracy"]
        validation_loss += history.history["val_loss"]

    plot_train_history(train_acc, validation_acc, train_loss, validation_loss, model_num)
    plot_learning_results(val_acc, val_auc, test_acc, test_auc, model_num)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model_num', metavar='model_num', type=str,
                        help='the number of the model file to use')
    parser.add_argument('trainable_layers', metavar='model_num', type=str,
                        help='the layers that should be trainable: should be list of\
                     ranges (a-b) splited by commas. for example, 21-22,39-39')
    parser.add_argument('train', metavar='train', type=str,
                        help='the name of the training file')
    parser.add_argument('test', metavar='test', type=str,
                        help='the name of the testing file')
    parser.add_argument('--seq2tensor', metavar='seq2tensor', type=str, nargs='?',
                        default='vec5_CTC.txt')
    parser.add_argument('--seq_size', metavar='seq_size', type=int, default=1000, nargs='?',
                        help='the length of the sequence to use for embedding')
    parser.add_argument('--steps', metavar='steps', type=int, default=1000, nargs='?',
                        help='the length of the sequence to use for embedding')
    parser.add_argument('--epochs', metavar='epochs', type=int, default=5, nargs='?')
    parser.add_argument('--thresh', metavar='thresh', type=float, default=0.5, nargs='?',
                        help='the threshold for comparing accuracy')
    parser.add_argument('--log_each_epoch', metavar='log_each_epoch', type=bool, default=False)

    args = parser.parse_args()
    seq_size = args.seq_size
    max_sequence_length = 1000
    score_threshold = args.thresh
    seq2t = s2t(args.seq2tensor)
    epochs = args.epochs
    steps = args.steps
    num_gpus = 1
    batch_size = 200 * num_gpus
    option = 'seq'
    model_num = args.model_num
    model_file = f'model_rcnn_all_{model_num}.h5'
    preds_file = f'preds_rcnn.txt'
    open_preds = open(preds_file, "w")
    open_preds.close()
    train_file = f'../data/new/train/{args.train}.csv'
    test_file = f'../data/new/test/{args.test}.csv'
    trainable_layers = get_trainable_indexes(args.trainable_layers)
    print('parameters for training are:')
    print('\ttraining file:', train_file)
    print('\ttesting file:', test_file)
    print('\tmodel file:', model_file)
    print('\ttrainable layers:', trainable_layers)

    hp_set = set()
    prot2embed = {}
    with open(swissprot_file, 'r') as f:
        next(f)
        for line in f:
            items = line.strip().split('\t')
            if len(items[3]) > max_sequence_length:
                continue
            hp_set.add(items[0])
            prot2embed[items[0]] = np.array(seq2t.embed_normalized(items[3], seq_size))
    print('Number of host proteins: ', len(hp_set))
    vp_set = {}
    with open(cov_prot_file, 'r') as f:
        next(f)
        for line in f:
            items = line.strip().split(',')
            vp_set[items[0] + items[1]] = items[2]
    train_positives, vp2numPos = get_h_and_v_proteins(train_file, vp_set, hp_set)
    for i in range(len(train_positives)):
        hp, vp = train_positives[i]
        prot2embed[vp] = np.array(seq2t.embed_normalized(vp_set[vp], seq_size))
    train_vps = set(vp2numPos.keys())
    print('Number of positives: ', len(train_positives))
    print('Number of train viral proteins: ', len(train_vps))
    test_positives, test_vp2numPos = get_h_and_v_proteins(test_file, vp_set, hp_set)
    for i in range(len(test_positives)):
        hp, vp = test_positives[i]
        prot2embed[vp] = np.array(seq2t.embed_normalized(vp_set[vp], seq_size))
    test_vps = set(test_vp2numPos.keys())
    print('Number of test positives: ', len(test_positives))
    print('Number of test viral proteins: ', len(test_vps))

    fine_tune_rcnn(model_file, model_num, epochs, batch_size, steps, score_threshold, option,
                   trainable_layers, seq_size,
                   hp_set, prot2embed,
                   train_positives, test_positives, train_vps, test_vps, args.log_each_epoch)


if __name__ == "__main__":
    main()
