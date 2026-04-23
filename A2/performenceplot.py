import matplotlib.pyplot as plt

# -------------------------
# Helper: read times only
# -------------------------
def read_times(filename):
    times = []
    with open(filename, "r") as f:
        for line in f:
            line = line.strip()

            # skip text like "Processes:"
            if line.startswith("Processes") or line.startswith("Weak") or line.startswith("Strong"):
                continue

            try:
                times.append(float(line))
            except:
                pass

    return times


# -------------------------
# Load data
# -------------------------
strong_times = read_times("strong_results.txt")
weak_times = read_times("weak_results.txt")

# reconstruct process counts (adjust if needed)
strong_p = [1, 2, 4, 8][:len(strong_times)]
weak_p   = [1, 2, 4, 8][:len(weak_times)]

# -------------------------
# Plot
# -------------------------
plt.figure(figsize=(12, 5))

# ---- Strong scaling ----
plt.subplot(1, 2, 1)
plt.plot(strong_p, strong_times, marker="o")
plt.title("Strong Scaling")
plt.xlabel("MPI Processes")
plt.ylabel("Time (s)")
plt.grid(True)

# ---- Weak scaling ----
plt.subplot(1, 2, 2)
plt.plot(weak_p, weak_times, marker="o", color="green")
plt.title("Weak Scaling")
plt.xlabel("MPI Processes")
plt.ylabel("Time (s)")
plt.grid(True)

plt.tight_layout()
plt.show()