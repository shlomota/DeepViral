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

thres = '0'
option = 'seq'
tid = sys.argv[2]
embedding_file = sys.argv[1]
print("option: ", option, "threshold: ", thres)

model_file = f'model_rcnn_{tid}.h5'
preds_file = f'preds_rcnn_{tid}.txt'
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

def build_model():
    seq_input1 = Input(shape=(seq_size, dim), name='seq1')
    seq_input2 = Input(shape=(seq_size, dim), name='seq2')
    l1=Conv1D(hidden_dim, 3)
    r1=Bidirectional(CuDNNGRU(hidden_dim, return_sequences=True))
    l2=Conv1D(hidden_dim, 3)
    r2=Bidirectional(CuDNNGRU(hidden_dim, return_sequences=True))
    l3=Conv1D(hidden_dim, 3)
    r3=Bidirectional(CuDNNGRU(hidden_dim, return_sequences=True))
    l4=Conv1D(hidden_dim, 3)
    r4=Bidirectional(CuDNNGRU(hidden_dim, return_sequences=True))
    l5=Conv1D(hidden_dim, 3)
    r5=Bidirectional(CuDNNGRU(hidden_dim, return_sequences=True))
    l6=Conv1D(hidden_dim, 3)
    s1=MaxPooling1D(3)(l1(seq_input1))
    s1=concatenate([r1(s1), s1])
    s1=MaxPooling1D(3)(l2(s1))
    s1=concatenate([r2(s1), s1])
    s1=MaxPooling1D(3)(l3(s1))
    s1=concatenate([r3(s1), s1])
    s1=MaxPooling1D(3)(l4(s1))
    s1=concatenate([r4(s1), s1])
    s1=MaxPooling1D(3)(l5(s1))
    s1=concatenate([r5(s1), s1])
    s1=l6(s1)
    s1=GlobalAveragePooling1D()(s1)
    s2=MaxPooling1D(3)(l1(seq_input2))
    s2=concatenate([r1(s2), s2])
    s2=MaxPooling1D(3)(l2(s2))
    s2=concatenate([r2(s2), s2])
    s2=MaxPooling1D(3)(l3(s2))
    s2=concatenate([r3(s2), s2])
    s2=MaxPooling1D(3)(l4(s2))
    s2=concatenate([r4(s2), s2])
    s2=MaxPooling1D(3)(l5(s2))
    s2=concatenate([r5(s2), s2])
    s2=l6(s2)
    s2=GlobalAveragePooling1D()(s2)
    merge_text = multiply([s1, s2])
    x = Dense(100, activation='linear')(merge_text)
    x = keras.layers.LeakyReLU(alpha=0.3)(x)
    x = Dense(int((hidden_dim+7)/2), activation='linear')(x)
    x = keras.layers.LeakyReLU(alpha=0.3)(x)
    main_output = Dense(1, activation='sigmoid')(x)
    merge_model = Model(inputs=[seq_input1, seq_input2], outputs=[main_output])
    return merge_model

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.3
sess = tf.Session(config=config)
K.set_session(sess)

counter = 0
family_aucs = []


K.clear_session()
tv_families = list(families - set([test_family]))
val_families = set(np.random.choice(tv_families, size = int(len(tv_families)/5), replace=False))
train_families = set(tv_families) - val_families
print('Train families: ', len(train_families), 'validation families', len(val_families))

train_vps = set()
train_families = families
val_families = families
for family in train_families:
    train_vps = train_vps | family2vp[family]
val_vps = vp_set - family2vp[test_family] - train_vps

triple_train = get_triple(positives, train_families, hp_set, train_vps, vp2patho, 'train')
triple_val, numPos_val = get_triple(positives, val_families, hp_set, val_vps,  vp2patho, 'val')

print("Number of triples in train", len(triple_train))

model = None
model = build_model()
adam = Adam(lr=0.001, amsgrad=True, epsilon=1e-6)
rms = RMSprop(lr=0.001)
model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])

train_gen, val_gen, test_gen = get_generators(triple_train, set(), set(), batch_size, prot2embed, option, embed_dict, MAXLEN=seq_size)

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

    val_auc = roc_auc_score(y_true, y_score)
    print('The ROCAUC for the val families in this epoch is ', val_auc)
    if val_auc > val_maxauc:
        print('Saving current model...')
        model.save("/content/drive/My Drive/" + model_file)
        val_maxauc = val_auc


