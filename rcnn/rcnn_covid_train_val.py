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
from keras.optimizers import Adam,  RMSprop
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

seq_size = 1000
MAXLEN = 1000
THRESH = 0.5
seq2t = s2t('vec5_CTC.txt')
hidden_dim = 50
dim = seq2t.dim

epochs = 5
num_gpus = 1
batch_size = 200*num_gpus
steps = 1000
# batch_size = 20*num_gpus
# steps = 1000

thres = '0'
option = 'seq'
model_num = sys.argv[2]
embedding_file = sys.argv[1]
print("option: ", option, "threshold: ", thres)

# model_file = f'model_rcnn_all_01.h5'
model_file = f'model_rcnn_all_{model_num}.h5'
preds_file = f'preds_rcnn.txt'
open_preds = open(preds_file, "w")
open_preds.close()

swissprot_file = '../data/swissprot-proteome.tab'
cov_prot_file = '../data/new/cov_protein_sequences.csv'
# hpi_file = '../data/train_1000.txt'

train = sys.argv[4]
test = sys.argv[5]
train_file = f'../data/new/train/{train}.csv'  # '../data/new/train/unthresholded.csv'
test_file = f'../data/new/test/{test}.csv'     # '../data/new/test/2b.csv'

embed_dict = read_embedding(embedding_file)

hp_set = set()
prot2embed = {}
with open(swissprot_file, 'r') as f:
    next(f)
    for line in f:
        items = line.strip().split('\t')
        if items[0] not in embed_dict:
            continue
        if len(items[3]) > MAXLEN:
            continue
        hp_set.add(items[0])
        prot2embed[items[0]] = np.array(seq2t.embed_normalized(items[3], seq_size))
print('Number of host proteins: ', len(hp_set))

vp_dict = {}
with open(cov_prot_file, 'r') as f:
    next(f)
    for line in f:
        items = line.strip().split(',')
        vp_dict[items[0] + items[1]] = items[2]

train_positives = set()
vp2numPos = {}
with open(train_file, 'r') as f:
    next(f)
    for line in f:
        items = line.strip().split(',')
        vp_key = items[0] + items[1].upper()
        hp = items[2]
        if vp_key not in vp_dict or hp not in hp_set:
            continue
        prot2embed[vp_key] = np.array(seq2t.embed_normalized(vp_dict[vp_key], seq_size))
        train_positives.add((hp, vp_key))
        if vp_key not in vp2numPos:
            vp2numPos[vp_key] = 0
        vp2numPos[vp_key] += 1
train_vps = set(vp2numPos.keys())
print('Number of positives: ', len(train_positives))
print('Number of train viral proteins: ', len(train_vps))

test_positives = set()
test_vp2numPos = {}
with open(test_file, 'r') as f:
    next(f)
    for line in f:
        items = line.strip().split(',')
        vp_key = items[0] + items[1].upper()
        hp = items[2]
        if vp_key not in vp_dict or hp not in hp_set:
            continue
        prot2embed[vp_key] = np.array(seq2t.embed_normalized(vp_dict[vp_key], seq_size))
        test_positives.add((hp, vp_key))
        if vp_key not in test_vp2numPos:
            test_vp2numPos[vp_key] = 0
        test_vp2numPos[vp_key] += 1
test_vps = set(test_vp2numPos.keys())
print('Number of test positives: ', len(test_positives))
print('Number of test viral proteins: ', len(test_vps))


config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.3
sess = tf.Session(config=config)
K.set_session(sess)

counter = 0
K.clear_session()

# indices = np.random.choice(len(positives), size = int(len(positives)/5), replace=False)
# val_positives = {tuple(v) for v in np.take(np.array(list(positives)), indices, axis=0)}
# train_positives = positives - val_positives
#
# val_vps = {p[1] for p in val_positives}
# train_vps = {p[1] for p in train_positives}
# print("Number of viral proteins in train, val, test: ", len(train_vps), len(val_vps), len(test_vps))
#
# triple_train = get_triple_without_family(train_positives, hp_set, train_vps, 'train')
# triple_val, numPos_val = get_triple_without_family(val_positives, hp_set, val_vps,  'val')
# triple_test, numPos_test = get_triple_without_family(test_positives, hp_set, test_vps, 'test')

triple_train, triple_val, triple_test = get_triples_without_family(train_positives, test_positives, hp_set, train_vps, test_vps)
print("Number of triples in train, val, test", len(triple_train), len(triple_val), len(triple_test))

# todo: restore
model = load_model(model_file)

# Frozen the weights of the first CNN layers
# frozen_layer_index = 20 # -5, -8, -12
frozen_layer_index = int(sys.argv[3])
for l in model.layers[0:frozen_layer_index]:
    l.trainable=False

adam = Adam(lr=0.001, amsgrad=True, epsilon=1e-6)
rms = RMSprop(lr=0.001)
model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])

# TODO: add triple_test
train_gen, val_gen, test_gen = get_generators(triple_train, triple_val, triple_test, batch_size, prot2embed, option, embed_dict, MAXLEN=seq_size)

val_maxauc = 0
for i in range(epochs):
    print('taxon ', counter, ' epoch ', i)
    history = model.fit_generator(generator=train_gen,
                        epochs=1,
                        steps_per_epoch = steps,
                        verbose=2,
                        max_queue_size = 50,
                        use_multiprocessing=False,
                        workers = 1)


    # compute metrics on validation data
    y_score = model.predict_generator(generator=val_gen, verbose=2,
                                       steps=int(np.ceil(len(triple_val)/batch_size)),
                                        max_queue_size = 50, workers = 1)
    y_true = np.array([int(example[-1]) for example in triple_val])

    val_acc = accuracy_score(y_true, (y_score>THRESH).astype(np.int))
    val_auc = roc_auc_score(y_true, y_score)
    print('Validation ROCAUC: %.3f, acc: %.3f' % (val_auc, val_acc))

    #test
    y_score = model.predict_generator(generator=test_gen, verbose=2,
                                      steps=int(np.ceil(len(triple_test)/batch_size)),
                                      max_queue_size = 50, workers = 1)
    y_true = np.array([int(example[-1]) for example in triple_test])

    test_acc = accuracy_score(y_true, (y_score>THRESH).astype(np.int))
    test_auc = roc_auc_score(y_true, y_score)
    print('Test ROCAUC: %.3f, acc: %.3f' % (val_auc, val_acc))
