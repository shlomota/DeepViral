import pandas as pd
df = pd.read_csv("train_dup.txt", sep="\t")
df = df.drop_duplicates(["Human Protein", "Virus Protein", "Species"])
df.to_csv("train.txt", sep="\t", index=False)
a = 5