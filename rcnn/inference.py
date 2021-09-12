"""
code for generating predictions using the latest model
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
from tqdm import tqdm

import pandas as pd

b2 = pd.read_csv(r"../data/new/2b.csv")
unthresh = pd.read_csv(r"../data/new/unthresholded.csv")


# hp = pd.read_csv(r"../data/new/all_human_protein_sequences.csv")
hp = pd.read_csv(r"../data/new/human_protein_sequences.csv")
vp = pd.read_csv(r"../data/new/cov2_proteins.csv")

# missing_proteins = ['KIAA0907', 'HIST1H1C', 'ATP5O', 'DEFA1']
# hp_list = b2['Host protein'].unique()

df = pd.DataFrame(columns=["hp", "hg", "vp", "score", "score1", "mist", "score2"])


tid = 0
model_file = f'model_rcnn.h5'
model = load_model(model_file)



import numpy as np
from seq2tensor import s2t
# embedding_file = "../data/julia_embed_cleaned.txt"
# embed_dict = read_embedding(embedding_file)
# a = embed_dict["P14907"] #sars

seq2t = s2t('vec5_CTC.txt')
hp["Entry"] = hp["Entry"].str.upper()
hp["Gene names  (primary )"] = hp["Gene names  (primary )"].str.upper()
vp["name"] = vp["name"].str.upper()
unthresh["Virus Protein"] = unthresh["Virus Protein"].str.upper()
unthresh["Human Protein"] = unthresh["Human Protein"].str.upper()
b2["Viral protein"] = b2["Viral protein"].str.upper()
b2["Host protein"] = b2["Host protein"].str.upper()

for vname in tqdm(vp.name.unique()):
    for hname in hp["Entry"].unique():

        hgname = hp[hp["Entry"]==hname]["Gene names  (primary )"].values[0]
        score1 = 0
        mist = 0
        score2 = 0
        hseq = hp[hp["Entry"] == hname]["Sequence"].values[0]
        vseq = vp[vp["name"] == vname]["sequence"].values[0]
        a = seq2t.embed_normalized(hseq, 1000)
        b = seq2t.embed_normalized(vseq, 1000)
        score = model.predict([[a], [b]])[0][0]

        row = unthresh[(unthresh["Human Protein"]==hname)&(unthresh["Virus Protein"]==vname)]
        if not row.empty:
            score1 = float(row["SaintScore"])
            mist = float(row["MIST"])

        if not b2[(b2["Host protein"]==hgname)&(b2["Viral protein"]==vname)].empty:
            score2 = 1

        if score1 or score2:
            print([hname, hgname, vname, score, score1, mist, score2])
        df.loc[len(df.index)] = [hname, hgname, vname, score, score1, mist, score2]

    #encoding="cp1252"
df.to_csv("/content/drive/My Drive/results.csv")
