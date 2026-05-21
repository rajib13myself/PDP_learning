import pandas as pd

df = pd.read_csv("results.csv")

t1 = df[df["processes"] == 1]["time"].values[0]

df["speedup"] = t1 / df["time"]

df["efficiency"] = df["speedup"] / df["processes"]

df.to_csv("speedup.csv", index=False)

print(df)