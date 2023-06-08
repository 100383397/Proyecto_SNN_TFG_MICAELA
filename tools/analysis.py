
import numpy as np
import matplotlib.pyplot as plt
import brian2 as b2

# Este script recoge las funciones empleadas para mostrar los resultádos tanto gráficos como
# cuantitativos obtenidos a la salida de la simulación. Todos los resultados están guardados en las
# carpetas ""_results, images y figures.


# Funcion para analizar la respuesta de los spikes de las neuronas de salida frente a las notas de entrada

def note_data_responses(spike_i, spike_t, from_t, to_t, note_t, n_notes):
    
    note_t = note_t # Distancia entre las notas en segundos
    n_notes = n_notes # Numero de notas que contiene el audio. 
                      # Cambia para cada secuencia (ver variable en script snn.py)

    # A la hora de mostrar los resultados cada nota aparecerá con el indice de su primera ocurrencia,
    # Por ejemplo, para el FUR ELISE: el sol sera la note 0, el fa# la 1, pero la siguiente distinta es
    # en el indice 5, que sera la nota re.

    # Para cada neurona, se consideran sus instantes de tiempo relevantes cuando se recorren 
    # los tiempos de los disparos registrados y ese indice de ese instante de tiempo coincide
    # con el indice de neurona evaluada.

    for neuron_n in set(spike_i):

        important_spike_t = spike_t[spike_i == neuron_n]
        important_spike_t = [t for t in important_spike_t if t > from_t and t < to_t]
        important_spike_t = np.array(important_spike_t)

        # Se convierte el array que almacena los tiempos de cada neurona entre el tiempo que dura cada nota 
        # a solo los enteros (int). Se calcula el array de restos (modulo) de los tiempos entre el numero 
        # total de notas contenidas en el audio.

        note_fir_int = np.floor(important_spike_t / note_t).astype(int)
        note_resp = np.mod(note_fir_int, n_notes) 

        # Se calcula la nota más comun
        final_note = np.argmax(np.bincount(note_resp))
    
        # Incluyendo las notas y neuronas erróneas en el análisis, se calcula el número de disparos correctos
        # e incorrectos, se normalizan dividiéndolos entre el total de disparos y se calcula la tasa de éxito 
        # por neurona.
        
        if (len(note_resp) > 1): 
      
            n_correct_firings = sum(note_resp == final_note)
            n_incorrect_firings = sum(note_resp != final_note)
            n_firings = len(note_resp)

            success_pct = float(n_correct_firings) / len(note_resp) * 100
            #failed_pct = float(n_incorrect_firings) / len(note_resp) * 100

            print("Neuron %d likes note %d, %.1f%% success \t incorrect correct total\t %i , %i , %i  " \
                % (neuron_n, final_note, success_pct, n_incorrect_firings, n_correct_firings, n_firings))
        else:
            incorrect_firings = len(note_resp)
            print("Neuron %d likes note %d, mistake (%i spikes)" \
                % (neuron_n, final_note, incorrect_firings))

    return 

# Gráfica que muestra el ajuste de los pesos entre instantes para cada neurona (diferencias de pesos)

def w_diff_figure(connections, w_monitor, from_t=0, to_t=-1, newfig=True):
    
    if newfig:
        plt.figure()
    else:
        plt.clf()
    neurons = set(connections.j)
    num_neurons = len(neurons)

    plt.subplot(num_neurons, 1, 1)
    plt.title('Ajustes de peso para cada neurona')

    # Donde el tiempo del monitor de los pesos es mayor de 0s
    from_i = np.where(w_monitor.t >= from_t * b2.second)[0][0]
    if to_t == -1:
        to_i = -1
    else:
    # Donde el tiempo del monitor de los pesos es menor de -1s
        to_i = np.where(w_monitor.t <= to_t * b2.second)[0][-1]

    w_diffs = w_monitor.w[:, to_i] - w_monitor.w[:, from_i]
    max_w_diff = np.max(w_diffs)
    min_w_diff = np.min(w_diffs)

    # Se guardan las diferencias de  pesos por si quiere trabajar con ellos (al final no se requirieron)
    # np.savetxt('evaluation/weight_diffs.out', weight_diffs)

    for neuron_n in neurons:
        plt.subplot(num_neurons, 1, neuron_n+1)
        important_weights = connections.j == neuron_n
        diff = w_diffs[important_weights]
        plt.plot(diff, color='green')
        plt.ylim([min_w_diff, max_w_diff])
        plt.yticks([])
        plt.xticks([])
        plt.ylabel("%d" % neuron_n)

# Definicion de las gráficas para mostrar el comportamiento de distintos parámetros por si se quiere
# trabajar con ellos en un futuro: potencial de membrana, corriente, threshold...

def params_figures(monitor, state_vals, firing_neurons, title):
   
    plt.figure()
    n_fir_n = len(firing_neurons)
    min_val = float('inf')
    max_val = -float('inf')

    for i, neuron_n in enumerate(firing_neurons):
        plt.subplot(n_fir_n, 1, i+1)
        n_val = state_vals[neuron_n, :]
        neuron_val_min = np.amin(n_val)
        if neuron_val_min < min_val:
            min_val = neuron_val_min
        neuron_val_max = np.amax(n_val)
        if neuron_val_max > max_val:
            max_val = neuron_val_max
        plt.plot(monitor.t, n_val)
        plt.ylabel("N. %d" % neuron_n)

    for i, _ in enumerate(firing_neurons):
        plt.subplot(n_fir_n, 1, i+1)
        plt.ylim([min_val, max_val])
        plt.yticks([min_val, max_val])

    for i in range(len(firing_neurons)-1):
        plt.subplot(n_fir_n, 1, i+1)
        plt.xticks([])
    plt.subplot(n_fir_n, 1, 1)
    plt.title(title)
