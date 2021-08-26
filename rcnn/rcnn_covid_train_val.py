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
tid = sys.argv[2]
embedding_file = sys.argv[1]
print("option: ", option, "threshold: ", thres)

# model_file = f'model_rcnn_all_01.h5'
model_file = f'model_rcnn_all_{tid}.h5'
preds_file = f'preds_rcnn.txt'
open_preds = open(preds_file, "w")
open_preds.close()

swissprot_file = '../data/swissprot-proteome.tab'
hpi_file = '../data/train_1000.txt'

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

positives = set()
family_dict = {}
pathogens = set()
family2vp = {}
vp2patho = {}
vp2numPos = {}

with open(hpi_file, 'r') as f:
    next(f)
    for line in f:
        items = line.strip().split('\t')
        if items[0] not in hp_set:
            continue
        if float(items[6]) >= float(thres):
            hp = items[0]
            vp = items[1]
            patho = '<http://purl.obolibrary.org/obo/NCBITaxon_' + items[2] + '>'
            if hp not in embed_dict or patho not in embed_dict:
                continue
            if len(items[5]) > MAXLEN:
                continue
            family = '<http://purl.obolibrary.org/obo/NCBITaxon_' + items[3] + '>'
            prot2embed[vp] = np.array(seq2t.embed_normalized(items[5], seq_size))
            family_dict[patho] = family
            positives.add((hp, vp, patho, family))
            pathogens.add(patho)
            if family not in family2vp:
                family2vp[family] = set()
            family2vp[family].add(vp)
            vp2patho[vp] = patho
            if vp not in vp2numPos:
                vp2numPos[vp] = 0
            vp2numPos[vp] += 1
vp_set = set(vp2patho.keys())
families = set(family2vp.keys())
print('Number of positives: ', len(positives))
print('Number of pathogens: ', len(pathogens))
print('Number of families: ', len(families))
print('Number of viral proteins: ', len(vp_set))

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.3
sess = tf.Session(config=config)
K.set_session(sess)

counter = 0
family_aucs = []


K.clear_session()
# tv_families = list(families - set([test_family]))
# val_families = set(np.random.choice(tv_families, size = int(len(tv_families)/5), replace=False))
# train_families = set(tv_families) - val_families
# print('Train families: ', len(train_families), 'validation families', len(val_families))

train_vps = set()
train_families = families
val_families = families
for family in train_families:
    train_vps = train_vps | family2vp[family]
# val_vps = vp_set - family2vp[test_family] - train_vps
val_vps = train_vps

indices = np.random.choice(len(positives), size = int(len(positives)/5), replace=False)
val_positives = {tuple(v) for v in np.take(np.array(list(positives)), indices, axis=0)}
train_positives = positives - val_positives
print(1)
triple_train = get_triple(train_positives, train_families, hp_set, train_vps, vp2patho, 'train')
print(2)
triple_val, numPos_val = get_triple(val_positives, val_families, hp_set, val_vps,  vp2patho, 'val')
print(3)
print("Number of triples in train, val", len(triple_train), len(triple_val))

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
train_gen, val_gen, test_gen = get_generators(triple_train, triple_val, triple_train, batch_size, prot2embed, option, embed_dict, MAXLEN=seq_size)

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

    y_score = model.predict_generator(generator=train_gen, verbose=2,
                                       steps=int(np.ceil(len(triple_train)/batch_size)),
                                        max_queue_size = 50, workers = 1)

    y_true = np.concatenate((np.ones(numPos_val), np.zeros(len(triple_train) - numPos_val)))
    y_true = np.array([int(example[-1]) for example in triple_train])

    val_auc = roc_auc_score(y_true, y_score)
    print('The ROCAUC for the val families in this epoch is ', val_auc)
