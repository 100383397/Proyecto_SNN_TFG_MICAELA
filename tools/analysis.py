
import numpy as np
import matplotlib.pyplot as plt
import brian2 as b2

# Este script recoge las funciones empleadas para mostrar los resultádos tanto gráficos como
# cuantitativos

##########################################################################################

#Grafica que muestra el ajuste de los pesos entre instantes para cada neurona (diferencias de pesos)

def w_diff(connections, weight_monitor, from_t=0, to_t=-1, newfig=True):
    if newfig:
        plt.figure()
    else:
        plt.clf()
    neurons = set(connections.j)
    n_neurons = len(neurons)

    plt.subplot(n_neurons, 1, 1)
    plt.title('Ajustes de peso para cada neurona')

    # Donde el tiempo del monitor de los pesos es mayor de 0s
    from_i = np.where(weight_monitor.t >= from_t * b2.second)[0][0]

    if to_t == -1:
        to_i = -1
    else:
    # Donde el tiempo del monitor de los pesos es menor de -1s
        to_i = np.where(weight_monitor.t <= to_t * b2.second)[0][-1]

    weight_diffs = weight_monitor.w[:, to_i] - weight_monitor.w[:, from_i]
    max_diff = np.max(weight_diffs)
    min_diff = np.min(weight_diffs)

    #Guardo las diferencias de  pesos por si necesito trabajar con ellos
    np.savetxt('evaluation/weight_diffs.out', weight_diffs)

    for neuron_n in neurons:
        plt.subplot(n_neurons, 1, neuron_n+1)
        relevant_weights = connections.j == neuron_n
        diff = weight_diffs[relevant_weights]
        plt.plot(diff, color='green')
        plt.ylim([min_diff, max_diff])
        plt.yticks([])
        plt.xticks([])
        plt.ylabel("%d" % neuron_n)
    plt.savefig('evaluation/weights_diff.png')

########################################################################################

# Definicion de las gráficas para mostrar el comportamiento de distintos parámetros:
# potencial de membrana, corriente, threshold...

def plot_state_var(monitor, state_vals, firing_neurons, title):
    plt.figure()
    n_firing_neurons = len(firing_neurons)
    min_val = float('inf')
    max_val = -float('inf')

    for i, neuron_n in enumerate(firing_neurons):
        plt.subplot(n_firing_neurons, 1, i+1)
        neuron_val = state_vals[neuron_n, :]
        neuron_val_min = np.amin(neuron_val)
        if neuron_val_min < min_val:
            min_val = neuron_val_min
        neuron_val_max = np.amax(neuron_val)
        if neuron_val_max > max_val:
            max_val = neuron_val_max
        plt.plot(monitor.t, neuron_val)
        plt.ylabel("N. %d" % neuron_n)

    for i, _ in enumerate(firing_neurons):
        plt.subplot(n_firing_neurons, 1, i+1)
        plt.ylim([min_val, max_val])
        plt.yticks([min_val, max_val])

    for i in range(len(firing_neurons)-1):
        plt.subplot(n_firing_neurons, 1, i+1)
        plt.xticks([])
    plt.subplot(n_firing_neurons, 1, 1)
    plt.title(title)

#####################################################################################

# Funcion para analizar la respuesta de los spikes de las neuronas frente a las notas de entrada

def analyse_note_responses(spike_indices, spike_times,from_time, to_time):
    
    note_length = 0.5  #Distancia entre las notas en segundos
    n_notes = 12 #numero de notas analizadas

    # Para cada neurona, se consideran sus instantes de tiempo relevantes cuando se recorren 
    # los tiempos de los disparos registrados y ese indice de ese instante de tiempo coincide
    # con el indice de neurona evaluada ( se recorre con el for).

    # - Ej: se toma el tiempo en segundos de spike_times(spike indice 0 = neurona numero 0),
    #   y esos tiempos donde coincidan los indices serán los tiempos de disparo de la neurona 0

    # Y tambien son tiempos relevantes si estos tiempos de los indices coincidentes están entre
    #  el tiempo minimo y maximo de los disparos. (el primero que se produce y el último)

    for neuron_n in set(spike_indices):

        relevant_spike_times = spike_times[spike_indices == neuron_n]
        relevant_spike_times = [t for t in relevant_spike_times if t > from_time and t < to_time]
        relevant_spike_times = np.array(relevant_spike_times)
        
        # Convertimos el array que almacena los tiempos relevantes de disparo de cada neurona entre
        # el tiempo que dura cada nota, a solo los enteros (tipo int) de la primera repeticion de las 
        # notas, y calculamos el array de restos (modulo) del array de tiempos tipo int entre el numero 
        # total de notas distintas.

        note_firings_int = np.floor(relevant_spike_times / note_length).astype(int)

        # Con este if excluimos del análisis aquellas neuronas que no deberían dispararse y generan picos
        # muy esporádicos

        note_responses=0
        if(len(note_firings_int) > 20):
            note_responses = np.mod(note_firings_int, n_notes) 
        #print(note_bin_firings)
        #print(note_responses)

        # Cuento el numero de ocurrencias de cada resto de tiempo y el máximo va a ser la nota más comun
        # a la que esa neurona dispara en su presencia.
            final_note = np.argmax(np.bincount(note_responses))

        #De forma que el numero de disparos correctos es la suma de los restos que son iguales a la nota comun
            n_correct_firings = sum(note_responses == final_note)
        
        # Se calcula el porcentaje como el numero de disparos correctos / disparos totales * 100
            success_firing_pct = float(n_correct_firings) / len(note_responses) * 100
        
            print("Neuron %d likes note %d, %.1f%% success" \
                % (neuron_n, final_note, success_firing_pct))
       
    return 

#####################################################################################

#Funcion para mostrar los pesos de cada neurona sin hacer la diferencia 
'''
def plot_weight(connections, weight_monitor, from_t=0, to_t=-1, newfig=True):
    if newfig:
        plt.figure()
    else:
        plt.clf()
    plt.subplot(n_neurons, 1, 1)
    plt.title('Progresion de pesos para cada neurona')
    plt.figure()
    neurons = set(connections.j)
    n_neurons = len(neurons)
    for i in range(12):
        plt.subplot(12, 1, i+1)
        relevant_weights = connections['input-layer1e'].j == i
        weights = np.array(connections['input-layer1e'].w)[relevant_weights]
        plt.plot(weights, color= 'red')
        plt.ylim([0, 1])
        plt.yticks([])
        plt.xticks([])
        plt.ylabel("%d" % i)
    plt.savefig('evaluation/weights.png')'''