import re
import numpy as np
import matplotlib.pyplot as plt

def parse_file(filename):
    P = []
    T = []

    with open(filename, 'r') as f:
        for line in f:
            if "Processes=" in line:
                p = int(re.search(r'Processes=(\d+)', line).group(1))
                t = float(re.search(r'Time=([0-9.eE+-]+)', line).group(1))
                P.append(p)
                T.append(t)

    return np.array(P), np.array(T)


P, T = parse_file("weak_results.csv")

# Sort
idx = np.argsort(P)
P = P[idx]
T = T[idx]

plt.figure(figsize=(6,5))
plt.plot(P, T, 'o-', linewidth=2)

plt.xlabel("Number of Processes")
plt.ylabel("Execution Time (s)")
plt.title("Weak Scaling Performance")

plt.grid(True)
plt.tight_layout()
plt.savefig("weak_scaling_plot.png", dpi=300)
plt.show()