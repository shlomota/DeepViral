"""
code used for initial analysis of model predictions w.r.t different labels
"""

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, precision_recall_curve, roc_auc_score, roc_curve, confusion_matrix, plot_confusion_matrix

thresh = 1e-4
thresh_clf = 0.2
df = pd.read_csv("results.csv")
df["score1"] = (df.score1>thresh).astype(np.int)
df["mist"] = (df.mist>thresh).astype(np.int)
df["score3"] = (df.score1) & (df.score2)
df["score4"] = (df.score1) & (df.mist)
df["score5"] = (df.score2) & (df.mist)

df["bin_score"] = (df.score>thresh_clf).astype(np.int)


df1 = df[df["score1"]==0].sample(sum(df["score1"])).append(df[df["score1"]==1])
df_mist = df[df["mist"]==0].sample(sum(df["mist"])).append(df[df["mist"]==1])
df2 = df[df["score2"]==0].sample(sum(df["score2"])).append(df[df["score2"]==1])
df3= df[df["score3"]==0].sample(sum(df["score3"])).append(df[df["score3"]==1])
df4= df[df["score4"]==0].sample(sum(df["score4"])).append(df[df["score4"]==1])
df5= df[df["score5"]==0].sample(sum(df["score5"])).append(df[df["score5"]==1])

print("saint results:")
print("Accuracy: %.3f" % accuracy_score(df1["score1"], df1["bin_score"]))
print("Precision: %.3f" % precision_score(df1["score1"], df1["bin_score"]))
print("Recall: %.3f" % recall_score(df1["score1"], df1["bin_score"]))
print("ROCAUC: %.3f" % roc_auc_score(df1["score1"], df1["score"]))
print("\n")

print("mist results:")
print("Accuracy: %.3f" % accuracy_score(df_mist["score1"], df_mist["bin_score"]))
print("Precision: %.3f" % precision_score(df_mist["score1"], df_mist["bin_score"]))
print("Recall: %.3f" % recall_score(df_mist["score1"], df_mist["bin_score"]))
print("ROCAUC: %.3f" % roc_auc_score(df_mist["score1"], df_mist["score"]))

print("\n")
print("table2 results:")
print("Accuracy: %.3f" % accuracy_score(df2["score2"], df2["bin_score"]))
print("Precision: %.3f" % precision_score(df2["score2"], df2["bin_score"]))
print("Recall: %.3f" % recall_score(df2["score2"], df2["bin_score"]))
print("ROCAUC: %.3f" % roc_auc_score(df2["score2"], df2["score"]))


print("\n")
print("score mist&saint results:")
print("Accuracy: %.3f" % accuracy_score(df4["score4"], df4["bin_score"]))
print("Precision: %.3f" % precision_score(df4["score4"], df4["bin_score"]))
print("Recall: %.3f" % recall_score(df4["score4"], df4["bin_score"]))
print("ROCAUC: %.3f" % roc_auc_score(df4["score4"], df4["score"]))


print("\n")
print("score mist&table2 results:")
print("Accuracy: %.3f" % accuracy_score(df5["score5"], df5["bin_score"]))
print("Precision: %.3f" % precision_score(df5["score5"], df5["bin_score"]))
print("Recall: %.3f" % recall_score(df5["score5"], df5["bin_score"]))
try:
    print("ROCAUC: %.3f" % roc_auc_score(df5["score5"], df5["score"]))
except:
    pass

print("\n")
print("score saint&table2 results:")
print("Accuracy: %.3f" % accuracy_score(df3["score3"], df3["bin_score"]))
print("Precision: %.3f" % precision_score(df3["score3"], df3["bin_score"]))
print("Recall: %.3f" % recall_score(df3["score3"], df3["bin_score"]))
print("ROCAUC: %.3f" % roc_auc_score(df3["score3"], df3["score"]))

print("length of mist&table2 dataset:", len(df5))
print("length of mist&saint dataset:", len(df4))
print("length of saint&table2 dataset:", len(df3))
print("length of mist dataset:", len(df_mist))
print("length of table2 dataset:", len(df2))
print("length of saint dataset:", len(df1))

"""
print("Accuracy:")
print("Score1: %.3f" % accuracy_score(df["score1"], df["bin_score"]))
print("Score2: %.3f" % accuracy_score(df["score2"], df["bin_score"]))
print("Score3: %.3f" % accuracy_score(df["score3"], df["bin_score"]))

print("Precision:")
print("Score1: %.3f" % precision_score(df["score1"], df["bin_score"]))
print("Score2: %.3f" % precision_score(df["score2"], df["bin_score"]))
print("Score3: %.3f" % precision_score(df["score3"], df["bin_score"]))

print("Recall:")
print("Score1: %.3f" % recall_score(df["score1"], df["bin_score"]))
print("Score2: %.3f" % recall_score(df["score2"], df["bin_score"]))
print("Score3: %.3f" % recall_score(df["score3"], df["bin_score"]))


from sklearn.metrics import plot_roc_curve
import matplotlib.pyplot as plt
fpr, tpr, thres = roc_curve(df["score1"], df["score"])
plt.plot(fpr, tpr)
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.title("ROC curve WRT score1")
plt.show()

fpr, tpr, thres = roc_curve(df["score2"], df["score"])
plt.plot(fpr, tpr)
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.title("ROC curve WRT score2")
plt.show()

fpr, tpr, thres = roc_curve(df["score3"], df["score"])
plt.plot(fpr, tpr)
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.title("ROC curve WRT score3")
plt.show()

print("%.3f" % (sum(df["bin_score"]) / len(df)))
print("%.3f" % (sum(df["score1"]) / len(df)))
print("%.3f" % (sum(df["score2"]) / len(df)))
print("%.5f" % (sum(df["score3"]) / len(df)))
print(len(df))
print(sum(df["score1"]))
print(sum(df["score2"]))
print(sum(df["score3"]))
"""