import sys
import numpy as np
import matplotlib.pyplot as plt


def read_file(filename):
    with open(filename, 'r') as f:
        data = f.read().split()

    values = np.array(list(map(float, data)))

    # If first value looks like N (integer), remove it
    # Otherwise treat whole file as data
    if len(values) > 0 and values[0].is_integer() and int(values[0]) == len(values) - 1:
        N = int(values[0])
        values = values[1:]
    else:
        N = len(values)

    return N, values

def main():
    if len(sys.argv) != 4:
        print("Usage: python verify.py input.txt output.txt reference.txt")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]
    ref_file = sys.argv[3]

    # ---- Read files ----
    N_in, u_input = read_file(input_file)
    N_out, u_output = read_file(output_file)
    N_ref, u_ref = read_file(ref_file)

    # ---- Basic checks ----
    if not (N_in == N_out == N_ref):
        print("Error: File sizes do not match!")
        sys.exit(1)

    # ---- Compute error ----
    error = np.max(np.abs(u_output - u_ref))
    print(f"Max error: {error:.6e}")

    if error == 0:
        print("✔ Output matches reference exactly.")
    else:
        print("✘ Output differs from reference!")

    # ---- Plot ----
    
    x = np.arange(N_in)

    plt.figure(figsize=(10, 5))

    # Professional, high-contrast but balanced colors
    input_color = "#6b7280"     # light gray (background signal)
    output_color = "#1d4ed8"    # strong blue (MAIN result)
    ref_color = "#dc2626"       # strong red (reference)

    # Input (least important visually)
    plt.plot(
        x, u_input,
        label="Input",
        linewidth=1.8,
        linestyle="-",
        color=input_color,
        alpha=0.7
    )

    # Output (make THIS clearly visible)
    plt.plot(
        x, u_output,
        label="Output (MPI result)",
        linewidth=2.8,
        linestyle="-",
        color=output_color,
        zorder=3
    )

    # Reference (clearly visible but not overpowering output)
    plt.plot(
        x, u_ref,
        label="Reference",
        linewidth=2.5,
        linestyle="--",
        color=ref_color,
        zorder=2
    )

    plt.xlabel("Index", fontsize=12)
    plt.ylabel("Value", fontsize=12)
    plt.title("Stencil Verification: Output vs Reference Comparison", fontsize=13)

    plt.legend(frameon=True, fontsize=11, loc="best")

    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.4)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()