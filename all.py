##################################################################
##################################################################
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
###########################################################################
# Elproceso de integracion de las neuronas LIF es una especie de tira y afloja
# de potenciales electricos. El punto clave es que este proceso refleja la
# fuerza relativa excitatoria vs inhibitoria. Si la excitacion es mas fuerte
# que la inhibicion, el potencial electrico de la neurona aumenta tal vez 
# hasta el punto de superar el umbral y disparar un potencial de acción de 
# salida. Si la inhibición es más fuerte, entonces el potencial eléctrico de 
# la neurona disminuye, y así se aleja más de superar el umbral para disparar.
##########################################################################
##########################################################################

#parametros usados para: neuronas, conexiones, monitores y ejecucion

neuron_params = {}

neuron_params['v_rest_e'] = -69 * b2.mV #potencial de reposo excitatoria original -65
neuron_params['v_rest_i'] = -60 * b2.mV #potencial de reposo inhibitoria original -60
neuron_params['v_reset_e'] = -69 * b2.mV #potencial de reset E original -65
neuron_params['v_reset_i'] = -45 * b2.mV #potencial de reset I original -45
neuron_params['v_thresh_e'] = -52 * b2.mV #umbral E
neuron_params['v_thresh_i'] = -40 * b2.mV #umbral I
neuron_params['refrac_e'] = 5 * b2.ms #periodo refractario E
neuron_params['refrac_i'] = 2 * b2.ms #periodo refractario I
neuron_params['vis'] = False
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
run_params['input_spikes_filename'] = 'spikes_inputs/two_notes_1.0_s.pickle'
run_params['no_standalone'] = True
# if args.run_time is not None:
#     run_params['run_time'] = float(args.run_time) * b2.second

monitor_params = {}

monitor_params['monitors_dt'] = 1000/60.0 * b2.ms

analysis_params = {}

analysis_params['save_figs'] = True
analysis_params['spikes_only'] = False
analysis_params['note_separation'] = 1 * b2.second
analysis_params['n_notes'] = 2

params = (neuron_params, connection_params, monitor_params, run_params, analysis_params)


####################################################################################

#Grafica que muestra el ajuste de los pesos para cada neurona (diferencias de)

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

    for neuron_n in neurons:
        plt.subplot(n_neurons, 1, neuron_n+1)
        relevant_weights = connections.j == neuron_n
        diff = weight_diffs[relevant_weights]
        plt.plot(diff, color='green')
        plt.ylim([min_diff, max_diff])
        plt.yticks([])
        plt.xticks([])
        plt.ylabel("%d" % neuron_n)
    
###############################################################################

#Definimos gráficas para mostrar el comportamiento de distintos parámetros:
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

############################################################################

#pickle
#cpp standalone

############################################################################

def load_input(run_params):
    
    #Cargamos los spikes para usarlos como entrada neuronal 

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

    # Neuronas excitatorias d ela capa 1
    neurons['layer1e'] = neuron_mod.excitatory_neurons(
        n_neurons=layer_n_neurons,
        params=neuron_params
    )

    # Neuronas inhibitorias de la capa 1
    neurons['layer1i'] = neuron_mod.inhibitory_neurons(
        n_neurons=layer_n_neurons,
        params=neuron_params
    )

    # visualisation neurons
    """if neuron_params['vis']:
        neurons['layer1vis'] = neuron_mod.visualisation_neurons(
            n_neurons=layer_n_neurons,
            params=neuron_params
        )"""

    return neurons

##########################################################################333

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

    #asignamos pesos para la capa excitatoria 1

    connections['input-layer1e'].w = 'rand() * 0.4'
    weights = np.array(connections['input-layer1e'].w)
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

    # excitatory to visualisation
    """if 'layer1vis' in neurons:
        connections['layer1e-layer1vis'] = synapse_mod.visualisation_synapses(
            source=neurons['layer1e'],
            target=neurons['layer1vis'],
            connectivity='i == j',
        )"""

    return connections

##################################################################################

def init_monitors(neurons, connections, monitor_params):
    
    #Inicializamos los objetos Brian monitoreando variables de estado en la red.

    monitors = {
        'spikes': {},
        'neurons': {},
        'connections': {}
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
        # record=True is currently broken for standalone simulations
        record=range(len(neurons['layer1e'])),
        dt=timestep
    )
    """if 'layer1vis' in neurons:
        monitors['neurons']['layer1vis'] = b2.StateMonitor(
            neurons['layer1vis'],
            ['v'],
            # record=True is currently broken for standalone simulations
            record=range(len(neurons['layer1vis'])),
            dt=b2.second/60
        )"""

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
    """
    Analyse results of simulation and plot graphs.
    """

    if len(monitors['spikes']['layer1e']) == 0:
        print("No spikes detected; not analysing")
        return

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

"""if run_params['save_results']:
        print("Saving results...")
        pickle_results(monitors, run_id)
        print("done!")"""



##################################################################
##################################################################
#primera capa ex ENTRADA

#segunda capa EX 1

#tercera capa IN 1

##################################################################
##################################################################
#conexiones

##################################################################

##################################################################
##################################################################

