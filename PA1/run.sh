#!/bin/bash

# run on 1 machine * 28 process, feel free to change it!
target_program=$1
num_elements=$2
input_file=$3

if [ $num_elements -lt 20000 ]; then
    node=1
    process=1
elif [ $num_elements -lt 200000 ]; then
    node=1
    for (( i=28; i>=1; i--)); do
        if [ $((num_elements % i)) -eq 0 ]; then
            process=$i
            break
        fi
    done
elif [ $num_elements -lt 20000000 ]; then
    node=2
    for (( i=56; i>=1; i--)); do
        if [ $((num_elements % i)) -eq 0 ]; then
            process=$i
            break
        fi
    done
    if [ $((process % 2)) -ne 0 ]; then
        node=1
    fi
else
    node=2
    for (( i=18; i>=1; i--)); do
        if [ $((num_elements % i)) -eq 0 ]; then
            process=$i
            break
        fi
    done
    if [ $((process % 2)) -ne 0 ]; then
        node=1
    fi
fi

echo "Running $num_elements data with $process process"
srun -N $node -n $process --cpu-bind=cores $*
