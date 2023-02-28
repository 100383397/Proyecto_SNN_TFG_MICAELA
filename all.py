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

neuron_params['v_rest_e'] = -65 * b2.mV #potencial de reposo excitatoria
neuron_params['v_rest_i'] = -60 * b2.mV #potencial de reposo inhibitoria
neuron_params['v_reset_e'] = -65 * b2.mV #potencial de reset E
neuron_params['v_reset_i'] = -45 * b2.mV #potencial de reset I
neuron_params['v_thresh_e'] = -52 * b2.mV #umbral E
neuron_params['v_thresh_i'] = -40 * b2.mV #umbral I
neuron_params['refrac_e'] = 5 * b2.ms #periodo refractario E
neuron_params['refrac_i'] = 2 * b2.ms #periodo refractario I
neuron_params['vis'] = False
neuron_params['tc_v_ex'] = 100 * b2.ms
neuron_params['tc_v_in'] = 10 * b2.ms
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
connection_params['ex-in-w'] = 10.4
connection_params['in-ex-w'] = 17.0

run_params = {}

run_params['layer_n_neurons'] = 16 #numero de neuronas en la capa 1
run_params['input_spikes_filename'] = 'spikes_inputs/two_notes_1.0_s.pickle'
run_params['no_standalone'] = True
# if args.run_time is not None:
#     run_params['run_time'] = float(args.run_time) * b2.second

monitor_params = {}
monitor_params['monitors_dt'] = 1000/60.0 * b2.ms

params = (neuron_params, connection_params, monitor_params, run_params)

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
    if 'layer1vis' in neurons:
        monitors['neurons']['layer1vis'] = b2.StateMonitor(
            neurons['layer1vis'],
            ['v'],
            # record=True is currently broken for standalone simulations
            record=range(len(neurons['layer1vis'])),
            dt=b2.second/60
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
##################################################################