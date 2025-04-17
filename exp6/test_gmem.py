import matplotlib.pyplot as plt

strides = []
bandwidths = []

with open("test_gmem.log", "r") as file:
    lines = file.readlines()
    for i in range(0, len(lines), 3):
        stride_line = lines[i].strip()
        bandwidth_line = lines[i+1].strip()
        stride = int(stride_line.split(":")[1].strip())
        bandwidth = float(bandwidth_line.split(":")[1].strip())
        strides.append(stride)
        bandwidths.append(bandwidth)

plt.figure()
plt.bar([str(s) for s in strides], bandwidths)
plt.title("global memory")
plt.xlabel("stride")
plt.ylabel("bandwidth (GB/s)")
plt.savefig("test_gmem.svg")
