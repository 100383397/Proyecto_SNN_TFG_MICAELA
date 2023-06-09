# Computación bio-inspiada para aplicaciones musicales 

Este repositorio contiene todo el código fuente empleado para la elaboración del Trabajo de Fin de Grado titulado "Computación bio-inspiada para aplicaciones musicales". También se incluyen todos los datos y figuras obtenidos como resultado de las distintas simulaciones planteadas.

## Requerimientos

En primer lugar, para poder clonar el repositorio adecuadamente, es necesario instalar Git Large File Storage: <https://git-lfs.github.com>.


Además se necesitará:

* Python 3.9.16
* Brian 2 (http://briansimulator.org)
* NumPy
* matplotlib
* Mingus para generar las secuencias de audio manualmente
* librosa

## Archivos

* `audios_train` y `audios_validation` contienen las secuencias de audio, en archivos `.wav`, empleadas para el entrenamiento y la validación de la red neuronal. 

* `audio_chords_input.py` y `audio_notes_input.py`  
son los ficheros que, haciendo uso de la libreria de `mingus`, generan las secuencias de audio mediante los archivos de sonido `.sf2` contenidos en la carpeta denominada `soundfonts`.

* `audio_spikes_input_melspectrogram.py` es el script empleado para procesar el audio de entrada y convertirlo a trenes de picos para introducirlo a la red. `audio_spikes_input_spectrogram.py` y `audio_spikes_input_chromaQconstant.py` son scripts adicionales de otros procesados del audio probados.

* `spikes_input_train` y `spikes_input_validation` son las carpetas que contienen los archivos `.pickle` que contienen la información de los trenes de picos de las neuronas para introducirlo como entrada a la red neuronal.

* `tools`  contiene módulos empleados para la simulación (funciones que definen los grupos neuronales, los procesos de sinapsis y el análisis de los datos de salida y su representación).

* `snn.py` script principal de la simulación que se ejecutará a travez de los ficheros `train_script.sh` y `validation_script.sh` (ver isntrucciones de ejecución de la simulación más abajo).

* `weigths_train` almacena todos los archivos `.pickle` que contienen los pesos finales de cada uno de los entrenamientos realizados. Y el archivo `weights.pickle` situado fuera de esta carpeta son los pesos congelados empleados para las simulaciones del proyecto.