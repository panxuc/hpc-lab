import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

pattern = re.compile(
    r"(\w+)\s+(\d+)\s+(\d+)\s+Exec-time:\s+([\d.]+)\s*ms",
    re.IGNORECASE
)

data = []

with open("run_cuda.log", "r") as f:
    for line in f:
        match = pattern.search(line)
        if match:
            method = match.group(1).lower()
            bx = int(match.group(2))
            by = int(match.group(3))
            time_val = float(match.group(4))
            data.append([method, bx, by, time_val])

df = pd.DataFrame(data, columns=["method", "block_size_x", "block_size_y", "time"])

df_shared = df[df["method"]=="shared_memory"]
df_naive  = df[df["method"]=="naive"]

def plot(pivot, title, xlabel, ylabel, file_name, colorbar_label="time (s)"):
    pivot = pivot.sort_index().sort_index(axis=1)
    plt.figure(figsize=(16, 12))
    im = plt.imshow(pivot, aspect="auto", origin="lower", cmap='YlOrRd')
    
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            value = pivot.iloc[i, j]
            if not pd.isna(value):
                plt.text(j, i, f"{value:.2f}", ha="center", va="center", color="black", fontsize=8)
    
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.colorbar(im, label=colorbar_label)
    plt.xticks(ticks=np.arange(len(pivot.columns)), labels=pivot.columns)
    plt.yticks(ticks=np.arange(len(pivot.index)), labels=pivot.index)
    plt.tight_layout()
    plt.savefig(file_name)
    plt.close()

pivot_shared = df_shared.pivot(columns="block_size_x", index="block_size_y", values="time")
plot(pivot_shared, "shared_memory", "block_size_x", "block_size_y", "plot_shared_memory.svg", "Exec-time (ms)")

pivot_naive = df_naive.pivot(columns="block_size_x", index="block_size_y", values="time")
plot(pivot_naive, "naive", "block_size_x", "block_size_y", "plot_naive.svg", "Exec-time (ms)")

speedup = pivot_naive / pivot_shared
plot(speedup, "speedup", "block_size_x", "block_size_y", "plot_speedup.svg", "Speedup")
