import pandas as pd
df = pd.read_csv("train_dup.txt", sep="\t")
df2 = df.dropna(subset=["MIscore"])
df2 = df2.sort_values('MIscore', ascending=False).drop_duplicates(["Human Protein", "Virus Protein", "Species"]).sort_index()
# df3 = df.drop_duplicates(["Human Protein", "Virus Protein", "Species"])
df2.to_csv("train.txt", sep="\t", index=False)


