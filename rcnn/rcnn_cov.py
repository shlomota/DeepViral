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
# option = 'seq'
# tid = sys.argv[2]
# embedding_file = sys.argv[1]

# option = sys.argv[1]
embedding_file = sys.argv[1]
taxon1 = sys.argv[2]
taxon2 = sys.argv[3]
taxon3 = sys.argv[4]
cv = sys.argv[5]
tid = sys.argv[6]

# print("option: ", option, "threshold: ", thres)
sars2_interactions = '../data/media-6.xlsx'
sars2_sequences = '../data/2020-04-krogan-sarscov2-sequences-uniprot-mapping.xlsx'

# model_file = f'model_{option}_t{taxon1}_v{taxon2}_{cv}_{tid}.h5'
# preds_file = f'preds_{option}_t{taxon1}_v{taxon2}_{cv}_{tid}.txt'


model_file = f'model_rcnn_{tid}.h5'
preds_file = f'preds_rcnn_{tid}.txt'
open_preds = open(preds_file, "w")
open_preds.close()

swissprot_file = '../data/swissprot-proteome.tab'
hpi_file = '../data/train_1000.txt'

params = get_params()

haaindex, vaaindex = get_aaindex(swissprot_file, hpi_file)
# print("The amino acids for human are ", list(haaindex))
# print("The amino acids for viruses are ", list(vaaindex))


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

test_taxon = "<http://purl.obolibrary.org/obo/NCBITaxon_" + taxon1 + ">"
val_taxon = "<http://purl.obolibrary.org/obo/NCBITaxon_" + taxon2 + ">"
original_family = "<http://purl.obolibrary.org/obo/NCBITaxon_" + taxon3 + ">"
taxon2count = {}
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
            if patho == test_taxon:
                family = "<http://purl.obolibrary.org/obo/NCBITaxon_test>"
            if cv == "species" and patho == val_taxon:
                family = "<http://purl.obolibrary.org/obo/NCBITaxon_val>"
            prot2embed[vp] = np.array(seq2t.embed_normalized(items[5], seq_size))
            # prot2embed[vp] = to_onehot(items[5], vaaindex)
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


#added
# family 11118 sars1 694009 sars2 2697049
sars2_interactions = 'data/media-6.xlsx'
sars2_sequences = 'data/2020-04-krogan-sarscov2-sequences-uniprot-mapping.xlsx'
dfs_interaction = pd.read_excel(sars2_interactions, sheet_name = None, skiprows=1)
for index, row in dfs_interaction["Sheet1"].iterrows():
    hp = row['Preys']
    vp = row['Bait']
    patho = '<http://purl.obolibrary.org/obo/NCBITaxon_2697049>'
    family = '<http://purl.obolibrary.org/obo/NCBITaxon_test>'
    if hp not in embed_dict or hp not in prot2embed:
            continue
    family_dict[patho] = family
    positives.add((hp, vp, patho, family))
    pathogens.add(patho)
    if family not in family2vp:
        family2vp[family] = set()
    family2vp[family].add(vp)
    vp2patho[vp] = patho

hcov2id = {}
with open('data/coronaviridae_seqs.txt', 'r') as f:
    next(f)
    for line in f:
        items = line.strip().split('\t')
        seq = items[2]
        if len(seq) > MAXLEN:
            seq = seq[:MAXLEN]
        if len(seq) > MAXLEN or len(seq) == 0:
            continue
        vp = items[0] + items[1]
        prot2embed[vp] = to_onehot(seq, vaaindex)
        hcov2id[items[0]] = items[3]

gname2unip = {"BTF3":"P20290", "BAP1":"Q92560", "ALB":"P02768", "NMB":"P08949", "BRF1":"Q92994",
              "CCNL1":"Q9UK58", "PCM1": "Q15154", "TRAF2": "Q12933", "DDX21": "Q9NR30", "PRKACA": "P17612",
             "CEP68": "Q76N32", "GLA":"P06280", "TRMT1":"Q9NXH9", "SCAP":"Q12770", "RPS4":"P62701",
             "USP10": "Q14694"}
with open('data/genename2uniprot.tab', 'r') as f:
    next(f)
    for line in f:
        items = line.strip().split('\t')
        gname = items[0]
        unip = items[1]
        full = items[2]
        if gname in gname2unip:
            continue
        gname2unip[gname] = unip
print("len(gname2unip)", len(gname2unip))

ppi_file = "data/12967_2020_2480_MOESM1_ESM.xlsx"
dfs = pd.read_excel(ppi_file, sheet_name = "Table S1")

hcovs = ["SARS-CoV", "MERS-CoV", "HCoV-NL63", "HCoV-229E", "HCoV-OC43"]
for index, row in dfs.iterrows():
    if row["VIRUS"] not in hcovs or row["SPECIES"] != "Human":
        continue
    vp = row["VIRUS"] + row["VIRAL PROTEIN"]
    hp = gname2unip[row["CELLULAR PROTEIN (Uniprot)"]]
    if vp not in prot2embed or hp not in prot2embed or hp not in embed_dict:
        continue
    family = '<http://purl.obolibrary.org/obo/NCBITaxon_11118>'
    patho = '<http://purl.obolibrary.org/obo/NCBITaxon_' + hcov2id[row["VIRUS"]] + ">"
    if cv == "species" and patho == val_taxon:
        family = "<http://purl.obolibrary.org/obo/NCBITaxon_val>"
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

dfs_seq = pd.read_excel(sars2_sequences)
for index, row in dfs_seq.iterrows():
    vp = row['Krogan name']
    seq = row['Sequence']
    if '*' == seq[len(seq)-1]:
        seq = seq[:len(seq)-1]
    prot2embed[vp] = to_onehot(seq, vaaindex)
#done




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

#for test_family in families:
for test_family in ["<http://purl.obolibrary.org/obo/NCBITaxon_test>"]:
    counter+=1
    K.clear_session()
    print('Test family %d: %s' % (counter, test_family))

    #before
    """
    tv_families = list(families - set([test_family]))
    val_families = set(np.random.choice(tv_families, size = int(len(tv_families)/5), replace=False))
    train_families = set(tv_families) - val_families
    """

    #after
    if cv == "family":
        tv_families = list(families - set([test_family, "<http://purl.obolibrary.org/obo/NCBITaxon_val>", original_family]))
        val_families = set(np.random.choice(tv_families, size = int(len(tv_families)/5), replace=False))
        train_families = set(tv_families) - val_families
    elif cv == "species":
        tv_families = list(families - set([test_family]))
        val_families = set(["<http://purl.obolibrary.org/obo/NCBITaxon_val>"])
        train_families = set(tv_families) - val_families
    #done

    print('Train families: ', len(train_families), 'validation families', len(val_families))

    train_vps = set()
    for family in train_families:
        train_vps = train_vps | family2vp[family]

    #start
    val_vps = set()
    for family in val_families:
        val_vps = val_vps | family2vp[family]

    #val_vps = vp_set - family2vp[test_family] - train_vps
    print("Number of viral proteins in train, val and test: ", len(train_vps), len(val_vps), len(family2vp[test_family]))
    
    triple_train = get_triple(positives, train_families, hp_set, train_vps, vp2patho, 'train')
    triple_val, numPos_val = get_triple(positives, val_families, hp_set, val_vps,  vp2patho, 'val')
    triple_test, numPos_test = get_triple(positives, [test_family], hp_set, family2vp[test_family], vp2patho, 'test')
    print("Number of triples in train, val, test", len(triple_train), len(triple_val), len(triple_test))
    
    model = None
    model = build_model()
    adam = Adam(lr=0.001, amsgrad=True, epsilon=1e-6)
    rms = RMSprop(lr=0.001)
    model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])

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

        y_score = model.predict_generator(generator=val_gen, verbose=2,
                                           steps=int(np.ceil(len(triple_val)/batch_size)), 
                                            max_queue_size = 50, workers = 1)
            
        y_true = np.concatenate((np.ones(numPos_val), np.zeros(len(triple_val) - numPos_val)))
        
        val_auc = roc_auc_score(y_true, y_score)
        print('The ROCAUC for the val families in this epoch is ', val_auc)
        if val_auc > val_maxauc:
            print('Saving current model...')
            model.save(model_file)
            val_maxauc = val_auc

        # y_score = model.predict_generator(generator=test_gen,
        #                             verbose=2,steps=np.ceil(len(triple_test)/batch_size),
        #                             max_queue_size = 30, use_multiprocessing=False, workers = 1)
        #
        # y_true = np.concatenate((np.ones(numPos_test), np.zeros(len(triple_test) - numPos_test)))
        # auprc = average_precision_score(y_true, y_score)
        # test_auc = roc_auc_score(y_true, y_score)
        # print("ROCAUC: %.4f, AUPRC: %.4f" % (test_auc, auprc))

    del model
    K.clear_session()
    model = load_model(model_file)
    y_score = model.predict_generator(generator=test_gen,
                                    verbose=2,steps=np.ceil(len(triple_test)/batch_size), 
                                    max_queue_size = 30, use_multiprocessing=False, workers = 1)

    y_true = np.concatenate((np.ones(numPos_test), np.zeros(len(triple_test) - numPos_test)))
    auprc = average_precision_score(y_true, y_score)
    test_auc = roc_auc_score(y_true, y_score)
    family_aucs.append((test_auc))
    print("ROCAUC: %.4f, AUPRC: %.4f" % (test_auc, auprc))
    
    with open(preds_file, 'a+') as f:
        for i in range(triple_test.shape[0]):
            f.write("%s\t%s\t%s\t%s\t%f\t%s\n" % (triple_test[i,1], triple_test[i,0], triple_test[i,2], test_family, y_score[i], i<numPos_test))
    
print("Mean ROCAUC of test families: %.4f" % (np.mean(family_aucs)))
