#!/bin/bash

python_file="snn.py"
folder_input_spikes="spikes_inputs_train"
folder_output_weights="weights_train"
num_rep=10
max_iter=30
for((i=1; i<=$num_rep; i++))
do
    echo "/****************************/"
    echo "Computing rep number: ${i}"
    echo "/****************************/"
    for((j=1; j<=$max_iter; j++))
        do
        echo "Iter n. ${j}"
        python3.9 "$python_file" "$folder_input_spikes/scale${i}rep_0.5_s.pickle" "$folder_output_weights/scale${i}rep_${j}iter_0.5_s.pickle" 0.5 >> "out_${i}rep.txt"
        done
done