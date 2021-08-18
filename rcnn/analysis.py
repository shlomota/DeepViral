import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, precision_recall_curve, roc_auc_score, roc_curve, confusion_matrix, plot_confusion_matrix


df = pd.read_csv("results.csv")
df["score1"] = (df.score1>0.5).astype(np.int)

df["score3"] = (df.score1) & (df.score2)

df["bin_score"] = (df.score>0.5).astype(np.int)


df1 = df[df["score1"]==0].sample(sum(df["score1"])).append(df[df["score1"]==1])
df2 = df[df["score2"]==0].sample(sum(df["score2"])).append(df[df["score2"]==1])

print("score1 results:")
print("Accuracy: %.3f" % accuracy_score(df1["score1"], df1["bin_score"]))
print("Precision: %.3f" % precision_score(df1["score1"], df1["bin_score"]))
print("Recall: %.3f" % recall_score(df1["score1"], df1["bin_score"]))
print("ROCAUC: %.3f" % roc_auc_score(df1["score1"], df1["score"]))


print("\n\nscore2 results:")
print("Accuracy: %.3f" % accuracy_score(df2["score2"], df2["bin_score"]))
print("Precision: %.3f" % precision_score(df2["score2"], df2["bin_score"]))
print("Recall: %.3f" % recall_score(df2["score2"], df2["bin_score"]))
print("ROCAUC: %.3f" % roc_auc_score(df2["score2"], df2["score"]))
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