#!/bin/bash
#PARA LA VALIDACION NO ES NECESARIO GUARDAR LOS PESOS DEL ENTRENAMIENTO, SE VALIDA CON UNOS PESOS 
#CONGELADOS CONCRETOS, SE ESCOGE SOLO 1 ENTRENAMIENTO
python_file="snn.py"
folder_input_spikes="spikes_inputs_validation"
folder_output_weights="weights_train"


python3.9 "$python_file" "$folder_input_spikes/chords5_2.0_s.pickle" "$folder_output_weights/scale${i}rep_${j}iter_1.0_s.pickle" 2.0 >> "out_info.txt"
