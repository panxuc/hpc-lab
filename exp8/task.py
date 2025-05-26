import json

id = int(input("Enter your ID: "))
a = b = id
MOD = 10**9 + 7

for _ in range(10):
    a = (a * 10007 + 7) % MOD
    b = ((b * 10007 + 9) ^ 1) % MOD

task1 = a % 8
task2 = (b + 1) % 2 + 1

with open(f"{id}.json", "w") as f:
    json.dump({
        "ID": id,
        "Task0": "MPI_Barrier",
        "Task1": task1,
        "Task2": task2
    }, f, indent=4)
