#!/bin/bash

# Ejecutando este script se genera y se guarada en un archivo .txt las simulaciones de los entrenamientos

python_file="snn.py" # Script que contiene la SNN
folder_input_spikes="spikes_inputs_train" #Carpeta de donde debe coger los .pickle como entrada a la red
folder_output_weights="weights_train" #Carpeta donde se guarda el ajuste de pesos finales de cada simulación

# La escala se repite de 1 a 10 veces. Cada audio aumenta en 1 la repetición de la escala que contienen
# Se van a generar 30 iteraciones, 30 entrenamiento para cada número de repeticiones de la escala.

num_rep=1
max_iter=30

# Hay que cambiar el tiempo definido del archivo de los input spikes que se desee probar
# Hay que modificar el parámetro final que indica la duración de notas del audio introducido, ya sea
# 0.5, 1.0 o 2.0

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