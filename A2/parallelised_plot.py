import numpy as np
import matplotlib.pyplot as plt
import sys

def read_values(filename):
    """
    Robust reader:
    - ignores non-numeric junk
    - works for both input and output formats
    """

    values = []

    with open(filename, "r") as f:
        for token in f.read().strip().split():
            try:
                values.append(float(token))
            except:
                pass  # ignore anything non-numeric

    return np.array(values)


# -------------------------
# Arguments
# -------------------------
if len(sys.argv) != 3:
    print("Usage: python plot.py <input_file> <output_file>")
    sys.exit(1)

input_file = sys.argv[1]
output_file = sys.argv[2]

# -------------------------
# Load data
# -------------------------
input_values = read_values(input_file)
output_values = read_values(output_file)

# -------------------------
# Safety check
# -------------------------
if len(output_values) == 0:
    print("ERROR: Output file parsed as empty. Check file format.")
    sys.exit(1)

# Trim mismatch if needed
n = min(len(input_values), len(output_values))
input_values = input_values[:n]
output_values = output_values[:n]

# -------------------------
# Error
# -------------------------
error = input_values - output_values

# -------------------------
# X-axis
# -------------------------
x = np.arange(n)

# -------------------------
# Plot
# -------------------------
plt.figure(figsize=(12, 8))
plt.suptitle("Input, Output and Error Comparison", fontsize=14)


plt.subplot(3, 1, 1)
plt.plot(x, input_values)
plt.title("Input")
plt.grid(True)

plt.subplot(3, 1, 2)
plt.plot(x, output_values)
plt.title("Output")
plt.grid(True)

plt.subplot(3, 1, 3)
plt.plot(x, error, color="red")
plt.title("Error (Input - Output)")
plt.grid(True)

plt.tight_layout()
plt.show()