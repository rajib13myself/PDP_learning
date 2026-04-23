import math

N = 100
with open("input.txt", "w") as f:
    f.write(str(N) + "\n")
    for i in range(N):
        x = 2 * math.pi * i / N
        f.write(f"{math.sin(x)} ")