#!/bin/bash

# Ejecutando este script se genera y se guarada en un archivo .txt las simulaciones de las validaciones
# Para la validación no es necesario guardar los pesos, ya que se valida con los pesos congelados
# del entrenamiento seleccionado, que es archivo weigths.pickle que aparece en el proyecto
# en la carpeta principal.

python_file="snn.py" # Script que contiene la SNN
folder_input_spikes="spikes_inputs_validation" #Carpeta de donde debe coger los .pickle como entrada a la red
folder_output_weights="weights_train" #no se emplea

# Hay que cambiar el nombre del archivo de los input spikes que se desee probar
# Hay que modificar el parámetro final que indica la duración de nots del audio introducido, ya sea
# 0.5, 1.0 o 2.0 

python3.9 "$python_file" "$folder_input_spikes/chords0_0.5_s.pickle" "$folder_output_weights/scale${i}rep_${j}iter_1.0_s.pickle" 0.5 >> "out_info.txt"
