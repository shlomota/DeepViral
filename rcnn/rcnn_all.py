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
# from utils import plot_train_history
from utils import *
from models import *

THRESH = 0.5
seq_size = 1000
MAXLEN = 1000
seq2t = s2t('vec5_CTC.txt')
hidden_dim = 50
dim = seq2t.dim

# epochs = 10
epochs = 1
num_gpus = 1
batch_size = 200*num_gpus
steps = 1000
# batch_size = 20*num_gpus
# steps = 1000

thres = '0'
option = 'seq'
embedding_file = sys.argv[1]
print("option: ", option, "threshold: ", thres)

model_file = f'model_rcnn.h5'
preds_file = f'preds_rcnn.txt'
open_preds = open(preds_file, "w")
open_preds.close()

swissprot_file = '../data/swissprot-proteome.tab'
# hpi_file = '../data/train_1000.txt'
hpi_file = '../data/train.txt'

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
        if items[0] not in hp_set or len(items) < 7:
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

# print(1)
# triple_train = get_triple(positives, train_families, hp_set, train_vps, vp2patho, 'train')
# print(2)
# triple_val, numPos_val = get_triple(positives, val_families, hp_set, val_vps,  vp2patho, 'val')

# get_triples_without_family()
triple_train, triple_val, triple_test = get_triples_without_family(positives, positives, hp_set, vp_set, vp_set, do_test=False)
print(3)
print("Number of triples in train", len(triple_train))
print("Number of triples in test", len(triple_test))

# todo: restore
model = None
model = build_model()
adam = Adam(lr=0.001, amsgrad=True, epsilon=1e-6)
rms = RMSprop(lr=0.001)
model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])

train_gen, val_gen, test_gen = get_generators(triple_train, triple_val, triple_test, batch_size, prot2embed, option, embed_dict, MAXLEN=seq_size)

test_maxauc = 0
train_acc, test_acc, train_loss, test_loss = [], [], [], []
for i in range(epochs):
    print('taxon ', counter, ' epoch ', i)
    history = model.fit_generator(generator=train_gen,
                        epochs=1,
                        steps_per_epoch = steps,
                        verbose=2,
                        max_queue_size = 50,
                        use_multiprocessing=False,
                        workers = 1,
                        validation_data=test_gen)
    # plot_train_history(history)

    y_score = model.predict_generator(generator=train_gen, verbose=2,
                                       steps=int(np.ceil(len(triple_train)/batch_size)),
                                        max_queue_size = 50, workers = 1)

    # y_true = np.concatenate((np.ones(numPos_val), np.zeros(len(triple_train) - numPos_val)))
    y_true = np.array([int(example[-1]) for example in triple_train])
    train_acc = accuracy_score(y_true, (y_score>THRESH).astype(int))
    train_auc = roc_auc_score(y_true, y_score)
    print('Train ROCAUC: %.3f, acc: %.3f' % (train_auc, train_acc))


    y_score = model.predict_generator(generator=test_gen, verbose=2,
                                      steps=int(np.ceil(len(triple_test)/batch_size)),
                                      max_queue_size = 50, workers = 1)
    y_true = np.array([int(example[-1]) for example in triple_test])
    test_acc = accuracy_score(y_true, (y_score>THRESH).astype(int))
    test_auc = roc_auc_score(y_true, y_score)
    print('Test ROCAUC: %.3f, acc: %.3f' % (test_auc, test_acc))

    train_acc += history.history["accuracy"]
    train_loss += history.history["loss"]
    test_acc += history.history["val_accuracy"]
    test_loss += history.history["val_loss"]

    print(train_acc[-1], train_loss[-1], test_acc[-1], test_loss[-1])

    if test_auc > test_maxauc:
        print('Saving current model...')
        model.save("/content/drive/My Drive/" + model_file)
        test_maxauc = test_auc


history.history["accuracy"] = train_acc
history.history["val_accuracy"] = test_acc
history.history["loss"] = train_loss
history.history["val_loss"] = test_acc
plot_train_history(history)