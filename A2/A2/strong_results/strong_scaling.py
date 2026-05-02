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


P, T = parse_file("strong_results.csv")

# Sort (important)
idx = np.argsort(P)
P = P[idx]
T = T[idx]

# Speedup
T1 = T[0]
S = T1 / T

# Efficiency
E = S / P

# =========================
# Plot Speedup
# =========================
plt.figure(figsize=(6,5))
plt.plot(P, S, 'o-', label="Measured Speedup", linewidth=2)
plt.plot(P, P, '--', label="Ideal Speedup", linewidth=2)

plt.xlabel("Number of Processes")
plt.ylabel("Speedup")
plt.title("Strong Scaling: Speedup")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("speedup_plot.png", dpi=300)
plt.show()

# =========================
# Plot Efficiency
# =========================
plt.figure(figsize=(6,5))
plt.plot(P, E, 's-', label="Efficiency", linewidth=2)

plt.xlabel("Number of Processes")
plt.ylabel("Efficiency")
plt.title("Strong Scaling: Efficiency")
plt.ylim(0,1.1)

plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("efficiency_plot.png", dpi=300)
plt.show()