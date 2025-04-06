#!/bin/bash

# run on 1 machine * 28 process, feel free to change it!
target_program=$1
num_elements=$2
input_file=$3

if [ $num_elements -lt 14000 ]; then
    process=1
else
    for (( i=56; i>=6; i--)); do
        if [ $((num_elements % i)) -eq 0 ]; then
            process=$i
            break
        fi
    done
fi
node=1
if [ $process -gt 28 ]; then
    node=2
fi

echo "Running $num_elements data with $process process"
srun -N $node -n $process --cpu-bind=cores $*

