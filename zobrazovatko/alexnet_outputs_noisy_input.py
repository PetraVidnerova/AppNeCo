import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import sys

ALTERNATIVE = sys.argv[1]
# ALTERNATIVE="pretrained"

NOISE_LEVEL = sys.argv[2]
INDEX = sys.argv[3]

DIR = "outputs_for_noisy_input"
df = pd.read_parquet(f"{DIR}/alexnet_{INDEX}_{NOISE_LEVEL}_{ALTERNATIVE}.parquet")


min_val = int(df["value"].min())-1
max_val = int(df["value"].max())+1


sns.violinplot(data=df, x="output", y="value")
plt.title(f"point num. {INDEX}, noise level {NOISE_LEVEL}, {ALTERNATIVE}")
plt.savefig(f"outputs_{ALTERNATIVE}_{INDEX}_{NOISE_LEVEL}.png",
            facecolor='white', transparent=False, dpi=600)


fig, axs = plt.subplots(ncols=10, nrows=1, figsize=(100, 10))

for i in range(10):
    sns.distplot(df[df["output"] == i]["value"], ax=axs[i])

plt.savefig(f"outputs_hist_{ALTERNATIVE}_{INDEX}_{NOISE_LEVEL}.png",
            facecolor="white", transparent=False, dpi=300)
