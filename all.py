#imports
from __future__ import print_function, division
import os.path
import pickle
import brian2 as b2
import numpy as np
import matplotlib.pyplot as plt
import modules.synapses as synapse_mod
import modules.neurons as neuron_mod

#b2.set_device('cpp_standalone')

###########################################################################
# Elproceso de integracion de las neuronas LIF es una especie de tira y afloja
# de potenciales electricos. El punto clave es que este proceso refleja la
# fuerza relativa excitatoria vs inhibitoria. Si la excitacion es mas fuerte
# que la inhibicion, el potencial electrico de la neurona aumenta tal vez 
# hasta el punto de superar el umbral y disparar un potencial de acción de 
# salida. Si la inhibición es más fuerte, entonces el potencial eléctrico de 
# la neurona disminuye, y así se aleja más de superar el umbral para disparar.
##########################################################################

#parametros usados para: neuronas, conexiones, monitores y ejecucion

neuron_params = {}

neuron_params['v_rest_e'] = -69 * b2.mV #potencial de reposo excitatoria original -65
neuron_params['v_rest_i'] = -64 * b2.mV #potencial de reposo inhibitoria original -60
neuron_params['v_reset_e'] = -69 * b2.mV #potencial de reset E original -65
neuron_params['v_reset_i'] = -49 * b2.mV #potencial de reset I original -45
neuron_params['v_thresh_e'] = -52 * b2.mV #umbral E
neuron_params['v_thresh_i'] = -40 * b2.mV #umbral I
neuron_params['refrac_e'] = 5 * b2.ms #periodo refractario E
neuron_params['refrac_i'] = 2 * b2.ms #periodo refractario I
neuron_params['tc_v_ex'] = 85 * b2.ms #original eran 100 ms
neuron_params['tc_v_in'] = 5 * b2.ms #origial era 10 ms
neuron_params['tc_ge'] = 1 * b2.ms
neuron_params['tc_gi'] = 2 * b2.ms
# reversal potentials for excitatory neurons
# excitatory reversal potential
neuron_params['e_ex_ex'] = 0 * b2.mV
# inhibitory reversal potential
neuron_params['e_in_ex'] = -100 * b2.mV
# reversal potentials for inhibitory neurons
neuron_params['e_ex_in'] = 0 * b2.mV
neuron_params['e_in_in'] = -85 * b2.mV
neuron_params['tc_theta'] = 1e6 * b2.ms
neuron_params['min_theta'] = 0 * b2.mV
neuron_params['offset'] = 20 * b2.mV
neuron_params['theta_coef'] = 0.02
neuron_params['max_theta'] = 60.0 * b2.mV

connection_params = {}

connection_params['tc_pre_ee'] = 20 * b2.ms
connection_params['tc_post_ee'] = 20 * b2.ms
connection_params['nu_ee_pre'] = 0.0001
connection_params['nu_ee_post'] = 0.02
connection_params['exp_ee_pre'] = 0.2
connection_params['exp_ee_post'] = 0.2
connection_params['wmax_ee'] = 1.0
connection_params['pre_w_decrease'] = 0.00025
connection_params['tc_ge'] = 1 * b2.ms
connection_params['tc_gi'] = 2 * b2.ms
connection_params['min_theta'] = 0 * b2.mV
connection_params['max_theta'] = 60 * b2.mV
connection_params['theta_coef'] = 0.02
connection_params['ex-in-w'] = 10.4 #10.4
connection_params['in-ex-w'] = 17.0 #17.0

run_params = {}

run_params['layer_n_neurons'] = 12 #numero de neuronas en la capa 1
run_params['input_spikes_filename'] = 'spikes_inputs/melscale_scale1_1.0_s.pickle'
run_params['no_standalone'] = True
# if args.run_time is not None:
#     run_params['run_time'] = float(args.run_time) * b2.second

monitor_params = {}

monitor_params['monitors_dt'] = 1000/60.0 * b2.ms

analysis_params = {}

analysis_params['save_figs'] = True
analysis_params['spikes_only'] = True
analysis_params['note_separation'] = 1.0 * b2.second
analysis_params['n_notes'] = 7


params = (neuron_params, connection_params, monitor_params, run_params, analysis_params)


####################################################################################

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

def plot_weight():

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

# Definimos gráficas para mostrar el comportamiento de distintos parámetros:
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

def order_spikes_by_note(spike_indices, spike_times, favourite_notes):
    # favourite_notes is a dictionary mapping neuron number to which
    # note it fires in response to
    # e.g. favourite_notes[3] == 2 => neuron 3 fires in response to note 2
    # extract spike times of the neurons which actually fire consistently
    relevant_times = [time
        for (spike_n, time) in enumerate(spike_times)
        if spike_indices[spike_n] in favourite_notes
    ]
    # extract the neuron indices corresponding to those spike times
    relevant_indices = [i for i in spike_indices if i in favourite_notes]

    # generate a list of which neuron in favourite_notes
    # each spike corresponds to, sorted by note order
    # (we need to do it like this instead of just plotting times against
    #  favourite_notes[spike_index] in case more than one neuron responds to
    #  each note)
    # first of all we need to sort the list by note order
    fav_note_neurons = np.array(favourite_notes.keys())
    fav_note_notes = np.array(favourite_notes.values())
    neurons_ordered_by_note = fav_note_neurons[np.argsort(fav_note_notes)]
    # now we figure out which index of that list each spike corresponds to
    neurons_ordered_by_note_indices = \
        [np.argwhere(neurons_ordered_by_note == i)[0][0]
         for i in relevant_indices]

    return (relevant_times, neurons_ordered_by_note_indices,
            neurons_ordered_by_note)

def ordered_spike_raster(spike_indices, spike_times, favourite_notes):
    (relevant_times, neurons_ordered_by_note_indices,
     neurons_ordered_by_note) = \
        order_spikes_by_note(spike_indices, spike_times, favourite_notes)

    plt.plot(relevant_times, neurons_ordered_by_note_indices,
             'k.', markersize=2)
    # of course, the y values will still correspond to indices of
    # neurons_ordered_by_note, whereas what we actually want to show is which
    # neuron is firing
    # so we need to map from note number to number
    n_notes = len(neurons_ordered_by_note)
    plt.yticks(
        range(n_notes),
        [str(neurons_ordered_by_note[i]) for i in range(n_notes)]
    )
    plt.ylim([-1, n_notes])
    plt.grid()
    plt.savefig('evaluation/ordered_spikes.png')
##########################################################################################

def analyse_note_responses(spike_indices, spike_times,
                           note_length, n_notes, from_time, to_time):
    """pickle_filename = run_params['input_spikes_filename']
    with open(pickle_filename, 'rb') as f:
        objects = pickle.load(f)
        (spike_times, spike_indices) = objects
    print("done!")"""
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
        n_misfirings = sum(note_responses != most_common_note)
        misfirings_pct = float(n_misfirings) / len(note_responses) * 100
        print("Neuron %d likes note %d, %.1f%% mistakes" \
            % (neuron_n, most_common_note, misfirings_pct))
        favourite_notes[neuron_n] = most_common_note

    return favourite_notes
###################################################################################

# pickle
# cpp standalone

# La red converge cuando deja de cambiar (cuando sus pesos dejan de cambiar), 
# cuando acaba diciendo siempre lo mismo (asi demostrariamos que la red ha aprendido)
# Hay cierta aleatoriedad a la hora de ejecutar la simulación pero al final deberian
# converger a algo similar

#####################################################################################

def load_input(run_params):
    
    #Cargamos los spikes para usarlos como entrada a la red neuronal 

    pickle_filename = run_params['input_spikes_filename']
    with open(pickle_filename, 'rb') as pickle_file:
        (input_spike_times, input_spike_indices) = pickle.load(pickle_file)
    input_spike_times = input_spike_times * b2.second

    spikes = {}
    spikes['indices'] = input_spike_indices
    spikes['times'] = input_spike_times

    return spikes

###############################################################################

def init_neurons(input_spikes, layer_n_neurons, neuron_params):
    
    #Inicializamos las neuronas

    neurons = {}

    n_inputs = 513 #numero neuronas igual al de entradas_audio_spikes.py

    neurons['input'] = neuron_mod.prespecified_spike_neurons(
        n_neurons=n_inputs,
        spike_indices=input_spikes['indices'],
        spike_times=input_spikes['times']
    )

    # Neuronas excitatorias de la capa 1
    neurons['layer1e'] = neuron_mod.excitatory_neurons(
        n_neurons=layer_n_neurons,
        params=neuron_params
    )

    # Neuronas inhibitorias de la capa 1
    neurons['layer1i'] = neuron_mod.inhibitory_neurons(
        n_neurons=layer_n_neurons,
        params=neuron_params
    )

    return neurons

#############################################################################

def init_connections(neurons, connection_params):
    
    #Iniciamos las conexiones sinapticas entre diferentes capas de neuronas
    
    connections = {}

    # input to layer 1 connections

    source = neurons['input'] #la capa input de spikes del audio
    target = neurons['layer1e'] #capa 1 de excitatorias
    connections['input-layer1e'] = synapse_mod.stdp_ex_synapses(
        source=source,
        target=target,
        connectivity=True, # all-to-all connectivity
        params=connection_params
    )

    #asignamos peso inicial aleatorio para la conexion entrada - excitatoria 1

    connections['input-layer1e'].w = 'rand() * 0.4'
    weights = np.array(connections['input-layer1e'].w)
 
    print(weights)
    #guardo los pesos por si necesito trabajar con ellos
    np.savetxt('evaluation/weights.out', weights) 

    with open('input-layer1e-weights.pickle', 'wb') as pickle_file:
        pickle.dump(weights, pickle_file)

    # conexion excitatory to inhibitory
    connections['layer1e-layer1i'] = synapse_mod.nonplastic_synapses(
        source=neurons['layer1e'],
        target=neurons['layer1i'],
        connectivity='i == j',
        synapse_type='excitatory'
    )
    connections['layer1e-layer1i'].w = connection_params['ex-in-w']

    # conexion inhibitory to excitatory
    connections['layer1i-layer1e'] = synapse_mod.nonplastic_synapses(
        source=neurons['layer1i'],
        target=neurons['layer1e'],
        connectivity='i != j',
        synapse_type='inhibitory'
    )
    connections['layer1i-layer1e'].w = connection_params['in-ex-w']

    return connections

##################################################################################

def init_monitors(neurons, connections, monitor_params):
    
    #Inicializamos los objetos Brian monitoreando variables de estado en la red.

    monitors = {
        'spikes': {}, #Para monitorizar y registrar los picos (spikes, con SpikeMonitor)
        'neurons': {}, #Para registrar variables de las neuronas: v, ge, theta....
        'connections': {} #Para registrar variables de las conexiones generadas: pesos, etc
    }

    for layer in ['input', 'layer1e']:
        monitors['spikes'][layer] = b2.SpikeMonitor(neurons[layer])

    if 'monitors_dt' not in monitor_params:
        timestep = None
    else:
        timestep = monitor_params['monitors_dt']

    monitors['neurons']['layer1e'] = b2.StateMonitor(
        neurons['layer1e'],
        ['v', 'ge', 'max_ge', 'theta'],
        record=range(len(neurons['layer1e'])),
        dt=timestep
    )

    conn = connections['input-layer1e']
    n_connections = len(conn.target) * len(conn.source)

    monitors['connections']['input-layer1e'] = b2.StateMonitor(
        connections['input-layer1e'],
        ['w', 'post', 'pre'],
        record=range(n_connections),
        dt=timestep
    )
    return monitors

#########################################################################

def run_simulation(run_params, neurons, connections, monitors):
    
    #Ejecutar la simulacion con los objetos creados hasta ahora

    net = b2.Network()
    for group in neurons:
        net.add(neurons[group])
    for connection in connections:
        net.add(connections[connection])
    for mon_type in monitors:
        for neuron_group in monitors[mon_type]:
            net.add(monitors[mon_type][neuron_group])

    net.run(run_params['run_time'], report='text')

    return net


#########################################################################


def analyse_results(monitors, connections, analysis_params):
    
    #Aalisis de los resultados y gráficas (plots)

    if len(monitors['spikes']['layer1e']) == 0:
        print("No spikes detected; not analysing")
        return
    
    if analysis_params['note_separation'] is not None and \
            analysis_params['n_notes'] is not None:
        end_time = max(monitors['spikes']['layer1e'].t)
        analyse_note_responses(
            spike_indices=monitors['spikes']['layer1e'].i,
            spike_times=monitors['spikes']['layer1e'].t,
            note_length=analysis_params['note_separation'],
            n_notes=analysis_params['n_notes'],
            from_time=end_time/2,
            to_time=end_time
        )

    plt.ion()

    plt.figure()

    plt.subplot(2, 1, 1)
    plt.title("Input spikes")
    plt.plot(
        monitors['spikes']['input'].t/b2.second,
        monitors['spikes']['input'].i,
        'k.',
        markersize=2
    )
    plt.ylabel("Neuron no.")
    plt.grid()

    plt.subplot(2, 1, 2)
    plt.title("Output spikes")
    plt.plot(
        monitors['spikes']['layer1e'].t/b2.second,
        monitors['spikes']['layer1e'].i,
        'k.',
        markersize=2
    )
    plt.ylim([-1, max(monitors['spikes']['layer1e'].i)+1])
    plt.grid()
    plt.ylabel("Neuron no.")

    plt.xlabel("Time (seconds)")
    plt.tight_layout()

    if analysis_params['spikes_only']:
        return

    firing_neurons = set(monitors['spikes']['layer1e'].i)

    plot_state_var(
        monitors['neurons']['layer1e'],
        monitors['neurons']['layer1e'].ge/b2.siemens,
        firing_neurons,
        'Current'
    )
    plot_state_var(
        monitors['neurons']['layer1e'],
        monitors['neurons']['layer1e'].theta/b2.mV,
        firing_neurons,
        'Threshold increase'
    )
    plot_state_var(
        monitors['neurons']['layer1e'],
        monitors['neurons']['layer1e'].v/b2.mV,
        firing_neurons,
        'Membrane potential'
    )

    plot_weight_diff(
        connections['input-layer1e'],
        monitors['connections']['input-layer1e']
    )

    #Para visualizar los pesos, para cada nuerona tomamos los pesos mas relevantes

    plot_weight()

    

#########################################################################


spike_filename = os.path.basename(run_params['input_spikes_filename'])
run_id = spike_filename.replace('.pickle', '')
#if not run_params['from_paramfile']:
#    param_mod.record_params(params, run_id)
input_spikes = load_input(run_params)
input_end_time = np.ceil(np.amax(input_spikes['times']))

if 'run_time' not in run_params:
    run_params['run_time'] = input_end_time
if not run_params['no_standalone']:
    if os.name == 'nt':
        build_dir = 'C:\\temp\\'
    else:
        build_dir = '/tmp/'
    build_dir += run_id
    b2.set_device('cpp_standalone', directory=build_dir)


print("Initialising neurons...")
neurons = init_neurons(
    input_spikes, run_params['layer_n_neurons'],
    neuron_params
)
print("done!")


print("Initialising connections...")
connections = init_connections(
    neurons,
    connection_params
)
print("done!")


print("Initialising monitors...")
monitors = init_monitors(neurons, connections, monitor_params)
print("done!")

    
print("Running simulation...")
net = run_simulation(run_params, neurons, connections, monitors)
print("done!")


def save_figures(name):
    print("Saving figures...")
    figs = plt.get_fignums()
    for fig in figs:
        plt.figure(fig)
        os.system('rm -f figures/%s_fig_%d.pdf' % (name, fig))
        plt.savefig('figures/%s_fig_%d.png' % (name, fig))
    print("done!")

analyse_results(monitors,connections,analysis_params)

if analysis_params['save_figs']:
        save_figures(run_id)


#############################################################################################33


##################################################################

"""plt.figure()
plt.plot()
plt.title("Analysis results, weights")
plt.plot(monitors['connections']['input-layer1e'].t/b2.second, monitors['connections']['input-layer1e'].w.T/ connection_params['wmax_ee']) #quedaria mejor con hist?? no lo veo claro
plt.ylabel("Weights no.")
plt.xlabel("Time (ms)")
plt.grid()
plt.savefig('evaluation/weights.png')
plt.plot()
plt.title("Analysis results, weights")
plt.hist(monitors['connections']['input-layer1e'].t/ b2.second, monitors['connections']['input-layer1e'].w.T/ connection_params['wmax_ee']) #quedaria mejor con hist?? no lo veo claro
plt.ylabel("Weights no.")
plt.xlabel("Time (ms)")
plt.grid()
plt.savefig('evaluation/histogram_weights.png')
"""



##################################################################

