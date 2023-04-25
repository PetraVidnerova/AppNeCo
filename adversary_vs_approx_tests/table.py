import os
import pandas as pd

files = [
    f
    for f in os.listdir(".")
    if f.startswith("mul8u_") and f.endswith(".txt")
]
print(files)

results_list = [] 
for filename in files:
    mul_name = filename[6:10]

    df = pd.read_csv(filename, header=None)
    df.columns = [mul_name]
    df[mul_name] = df[mul_name].map(lambda x: float(x[:-1]))
                                    
    results = df.describe()
    print(results)
    results_list.append(results)


df = pd.concat(results_list, axis=1).loc[["mean", "std", "min", "max"]].T
print(df)


for filename in files:
    mul_name = filename[6:10]
    
    with open(f"mul8u_{mul_name}.bin_robust_accuracy.log") as f:
        for line in f:
            if line.startswith("initial accuracy"):
                acc = float(line.split()[-1][:-1])
                print(acc)
                break
    df.loc[mul_name, "accuracy"] = acc

print(df)

df_sel = df[df["accuracy"] > 96.0]
print(df_sel)
