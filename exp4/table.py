import re

pattern = re.compile(
    r"(\w+)\s+(\d+)\s+(\d+)\s+Exec-time:\s+([\d.]+)\s*ms",
    re.IGNORECASE
)

data = {}

with open("run_cuda.log", "r") as f:
    for line in f:
        match = pattern.search(line)
        if match:
            method = match.group(1).lower()
            bx = int(match.group(2))
            by = int(match.group(3))
            time_val = match.group(4)
            key = (bx, by)
            if key not in data:
                data[key] = {}
            data[key][method] = f"{time_val} ms"

sorted_keys = sorted(data.keys(), key=lambda k: (k[0], k[1]))

print("| block size | naive | shared_memory |")
print("| :--------: | :---: | :-----------: |")
for bx, by in sorted_keys:
    key = (bx, by)
    naive_time = data[key].get("naive", "?")
    shared_time = data[key].get("shared_memory", "?")
    print(f"| {bx}x{by} | {naive_time} | {shared_time} |")
