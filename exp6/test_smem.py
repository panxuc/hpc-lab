import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

bitwidths = []
strides = []
bandwidths = []

with open("test_smem.log", "r") as file:
    lines = file.readlines()
    for i in range(0, len(lines), 4):
        bitwidth_line = lines[i].strip()
        stride_line = lines[i+1].strip()
        bandwidth_line = lines[i+2].strip()
        bitwidth = int(bitwidth_line.split(":")[1].strip())
        stride = int(stride_line.split(":")[1].strip())
        bandwidth = float(bandwidth_line.split(":")[1].strip())
        bitwidths.append(bitwidth)
        strides.append(stride)
        bandwidths.append(bandwidth)

grouped = defaultdict(list)
for bw, bw_stride, bw_bitwidth in zip(bandwidths, strides, bitwidths):
    grouped[bw_stride].append((bw_bitwidth, bw))

for key in grouped:
    grouped[key].sort()

all_strides = sorted(grouped.keys())
all_bitwidths = sorted(set(bitwidths))

x = np.arange(len(all_strides))
bar_width = 0.8 / len(all_bitwidths)

for i, bitwidth in enumerate(all_bitwidths):
    y = [dict(grouped[s]).get(bitwidth, 0) for s in all_strides]
    plt.bar(x + i * bar_width, y, width=bar_width, label=f"{bitwidth}-bit")

plt.xticks(x + bar_width * (len(all_bitwidths)-1) /
           2, [str(s) for s in all_strides])
plt.legend()
plt.title("shared memory")
plt.xlabel("stride")
plt.ylabel("bandwidth (GB/s)")
plt.savefig("test_smem.svg")
