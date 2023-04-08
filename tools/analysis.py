import numpy as np
import matplotlib.pyplot as plt
import brian2 as b2


#Grafica que muestra el ajuste de los pesos para cada neurona (diferencias de pesos)

def plot_weight_diff(connections, weight_monitor, from_t=0, to_t=-1, newfig=True):
    if newfig:
        plt.figure()
    else:
        plt.clf()
    neurons = set(connections.j)
    n_neurons = len(neurons)

    plt.subplot(n_neurons, 1, 1)
    plt.title('Ajustes de peso para cada neurona')

    from_i = np.where(weight_monitor.t >= from_t * b2.second)[0][0]
    if to_t == -1:
        to_i = -1
    else:
        to_i = np.where(weight_monitor.t <= to_t * b2.second)[0][-1]

    weight_diffs = weight_monitor.w[:, to_i] - weight_monitor.w[:, from_i]
    max_diff = np.max(weight_diffs)
    min_diff = np.min(weight_diffs)

    #guardo las diferencias de  pesos por si necesito trabajar con ellos
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

#####################################################################################

#Funcion para mostrar los pesos de cada neurona sin hacer la diferencia 

def plot_weight(connections):

    plt.title('Progresion de pesos para cada neurona')
    plt.figure()
    for i in range(12):
        plt.subplot(12, 1, i+1)
        relevant_weights = connections['input-layer1e'].j == i
        weights = np.array(connections['input-layer1e'].w)[relevant_weights]
        plt.plot(weights, color= 'red')
        plt.ylim([0, 1])
        plt.yticks([])
        plt.xticks([])
        plt.ylabel("%d" % i)
    plt.savefig('evaluation/weights.png')

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

# Funcion para analizar la respuesta a las notas y calcular el porcentaje de acierto y fallo
# de la red

def analyse_note_responses(spike_indices, spike_times,
                           note_length, n_notes, from_time, to_time):
    
    max_time = np.amax(spike_times) * b2.second
    from_time = max_time/2
    to_time = max_time
    note_length = 1.0

    n_notes = 7
    max_spikes = 0
    for neuron_n in set(spike_indices):
        relevant_spike_times = spike_times[spike_indices == neuron_n]
        relevant_spike_times = [t for t in relevant_spike_times if t > from_time and t < to_time]
        n_spikes = len(relevant_spike_times)
        if n_spikes > max_spikes:
            max_spikes = n_spikes
    print("done!")
    favourite_notes = {}
    for neuron_n in set(spike_indices):
        relevant_spike_times = spike_times[spike_indices == neuron_n]
        relevant_spike_times = [t for t in relevant_spike_times if t > from_time and t < to_time]
        relevant_spike_times = np.array(relevant_spike_times)
        n_spikes = len(relevant_spike_times)
        if n_spikes == 0 or n_spikes < 0.2 * max_spikes:
            continue

        note_bin_firings = np.floor(relevant_spike_times / note_length).astype(int)
        note_responses = np.mod(note_bin_firings, n_notes)
        most_common_note = np.argmax(np.bincount(note_responses))
        n_correct_firings = sum(note_responses == most_common_note)
        success_firing_pct = float(n_correct_firings) / len(note_responses) * 100
        print("Neuron %d likes note %d, %.1f%% success" \
            % (neuron_n, most_common_note, success_firing_pct))
        favourite_notes[neuron_n] = most_common_note

    return favourite_notes