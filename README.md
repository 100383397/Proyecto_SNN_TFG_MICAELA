# Computación bio-inspiada para aplicaciones musicales 

Este repositorio contiene todo el código fuente empleado para la elaboración del Trabajo de Fin de Grado titulado "Computación bio-inspiada para aplicaciones musicales". También se incluyen todos los datos y figuras obtenidos como resultado de las distintas simulaciones planteadas.

## Requerimientos

En primer lugar, para poder clonar el repositorio adecuadamente, es necesario instalar Git Large File Storage: <https://git-lfs.github.com>.


Además se necesitará:

* Python 3.9.16 (otras versiones son compatibles, se recomienda que sean a partir de Python 3)
* Brian 2 (http://briansimulator.org)
* NumPy
* matplotlib
* Mingus (librería para generar las secuencias de audio manualmente)
* librosa (librería para procesar el audio)

## Archivos

* `audios_train` y `audios_validation` contienen las secuencias de audio, en archivos `.wav`, empleadas para el entrenamiento y la validación de la red neuronal. 

* `audio_chords_input.py` y `audio_notes_input.py` son los ficheros que, haciendo uso de la libreria de `mingus`, generan las secuencias de audio mediante los archivos de sonido `.sf2` contenidos en la carpeta denominada `soundfonts`.

* `audio_spikes_input_melspectrogram.py` es el script empleado para procesar el audio de entrada y convertirlo a trenes de picos para introducirlo a la red. `audio_spikes_input_spectrogram.py` y `audio_spikes_input_chromaQconstant.py` son scripts adicionales de otros procesados del audio probados.

* `spikes_input_train` y `spikes_input_validation` son las carpetas que contienen los archivos `.pickle` que contienen la información de los trenes de picos de las neuronas de entrada para introducirlos a la red neuronal.

* `tools`  contiene módulos empleados para la simulación (funciones que definen los grupos neuronales, los procesos de sinapsis y el análisis de los datos de salida y su representación).

* `snn.py` script principal de la simulación que se ejecutará a través de los ficheros `train_script.sh` y `validation_script.sh` (ver instrucciones de ejecución de la simulación más abajo).

* `weigths_train` almacena todos los archivos `.pickle` que contienen los pesos finales de cada uno de los entrenamientos realizados. Y el archivo `weights.pickle`, situado fuera de esta carpeta, es el archivo de pesos congelados empleados para las validaciones del proyecto.

* `Results` carpeta que contiene todos los resultados obtenidos. En concreto, en la carpeta `images` se recogen todas las figuras generadas durante el procesado del audio y la generación de los trenes de picos iniciales (el espectrograma en escala de mel y su traducción a picos mediante Brian 2) y en la carpeta `figures` se recogen aquellas figuras obtenidas a la salida tras la simulación del entrenamiento y las validaciones realizadas.

* Se incluye material adicional en la carpeta de `Results`:`TRAIN_RESULTS.xlsx` y `VALIDATION_RESULTS.xlsx` recogen en tablas todos los resultados obtenidos de los 900 entrenamientos y las 180 validaciones (tasas de éxito, disparos correctos, incorrectos y totales por neurona, medias y desviaciones típicas). `proc_data_train.sh` y `proc_data_validation.sh` permiten pasar los datos de la simulación, grabados en un archivo de texto, a formato Excel. Y `summary note-neuron correspondences.pdf` es un documento guía que recoge los índices correspondientes para las notas de cada secuencia de audio empleadas en el entrenamiento y la validación.


## Ejecución de la simulación

Para traducir y generar los trenes de picos de los archivos de audio, que serán introducidos como entrada a la red neuronal, se debe ejecutar el siguiente comando:

`$ python3.9 audio_spikes_input_melspectrogram.py audios_validation/scale7oct5_0.5_s.wav`

Para ejecutar el número deseado de simulaciones del entrenamiento y grabar los resultados obtenidos en un fichero de texto, se emplea la siguiente orden:

`$ sh train_script.sh`

Para realizar las simulaciones de las validaciones y almacenar sus resultados, se ejecuta la siguiente línea de comando:

`$ sh validation_script.sh`
