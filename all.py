#imports
from __future__ import print_function, division
import os.path
import pickle
import brian2 as b2
import numpy as np
import matplotlib.pyplot as plt
import tools.synapses as s_mode
import tools.neurons as n_mode
import tools.analysis as a_mode

#b2.set_device('cpp_standalone')

####################################################################################

# Primero se definen los parametros usados para: neuronas, conexiones, monitores y
# para ejecutar la simulación. Asi quedan todos concentrados para que sean más faciles
# de localizar y modificar.

neurons_vars = {}

neurons_vars['v_rest_e'] = -69 * b2.mV #potencial de reposo excitatoria original -65
neurons_vars['v_rest_i'] = -64 * b2.mV #potencial de reposo inhibitoria original -60
neurons_vars['v_reset_e'] = -69 * b2.mV #potencial de reset E original -65
neurons_vars['v_reset_i'] = -49 * b2.mV #potencial de reset I original -45
neurons_vars['v_thresh_e'] = -52 * b2.mV #umbral E
neurons_vars['v_thresh_i'] = -40 * b2.mV #umbral I
neurons_vars['refrac_e'] = 6 * b2.ms #periodo refractario E
neurons_vars['refrac_i'] = 3 * b2.ms #periodo refractario I
neurons_vars['tc_v_ex'] = 95 * b2.ms #cte tiempo potencial membrana E, original eran 100 ms
neurons_vars['tc_v_in'] = 5 * b2.ms #cte tiempo poencial membrana I, origial era 10 ms
neurons_vars['tc_ge'] = 1 * b2.ms #cte tiempo de la conductancia E
neurons_vars['tc_gi'] = 2 * b2.ms #cte tiempo de la conductancia I
neurons_vars['e_ex_ex'] = 0 * b2.mV #Potencial de inversion sinaptica excitatorio neuronas excitatorias
neurons_vars['e_in_ex'] = -100 * b2.mV#Potencial de inversion sinaptica inhibitorio neuronas excitatorias
neurons_vars['e_ex_in'] = 0 * b2.mV #Potencial de inversion sinaptica excitatoria neuronas inhibitorias
neurons_vars['e_in_in'] = -86 * b2.mV#Potencial de inversion sinaptica inhibitorio neuronas inhibitorias
neurons_vars['tc_theta'] = 1e6 * b2.ms
neurons_vars['min_theta'] = 0 * b2.mV
neurons_vars['offset'] = 20 * b2.mV
neurons_vars['theta_coef'] = 0.02
neurons_vars['max_theta'] = 60.0 * b2.mV

connect_vars = {}

connect_vars['tc_pre_ee'] = 20 * b2.ms
connect_vars['tc_post_ee'] = 20 * b2.ms
connect_vars['nu_ee_pre'] = 0.0001
connect_vars['nu_ee_post'] = 0.02
connect_vars['exp_ee_pre'] = 0.2
connect_vars['exp_ee_post'] = 0.2
connect_vars['wmax_ee'] = 1.0
connect_vars['pre_w_decrease'] = 0.00025
connect_vars['tc_ge'] = 1 * b2.ms #cte tiempo de la conductancia E
connect_vars['tc_gi'] = 2 * b2.ms #cte tiempo de la conductancia I
connect_vars['min_theta'] = 0 * b2.mV
connect_vars['max_theta'] = 60 * b2.mV
connect_vars['theta_coef'] = 0.02
connect_vars['ex-in-w'] = 10.4 #PESO
connect_vars['in-ex-w'] = 17.0 #PESO

run_vars = {}

run_vars['layer_n_neurons'] = 12 #numero de neuronas en la capa 1
run_vars['input_spikes_filename'] = 'spikes_inputs/melscale_4_notes_0.5_s.pickle'
run_vars['no_standalone'] = True

mon_vars = {}

mon_vars['monitors_dt'] = 1000/60.0 * b2.ms

analysis_vars = {}

analysis_vars['save_figs'] = True

variables = (neurons_vars, connect_vars, mon_vars, run_vars, analysis_vars)

###################################################################################

# La red converge cuando deja de cambiar (cuando sus pesos dejan de cambiar), 
# cuando acaba diciendo siempre lo mismo (asi demostrariamos que la red ha aprendido)
# Hay cierta aleatoriedad a la hora de ejecutar la simulación pero al final deberian
# converger a algo similar.

#####################################################################################

def load_audio_input(run_vars):
    
    #Cargamos los spikes para usarlos como entrada a la red neuronal 

    pickle_filename = run_vars['input_spikes_filename']
    with open(pickle_filename, 'rb') as pickle_file:
        (input_spike_times, input_spike_indices) = pickle.load(pickle_file)
    input_spike_times = input_spike_times * b2.second

    spikes = {}
    spikes['indices'] = input_spike_indices
    spikes['times'] = input_spike_times

    return spikes

###############################################################################

def initialize_neurons(input_spk, layer_n_neurons, neurons_vars):
    
    #Inicializamos las neuronas

    neurons = {}

    n_inputs = 513 #numero neuronas igual al de entradas_audio_spikes.py

    neurons['input'] = n_mode.audio_spike_neurons(
        n_neurons=n_inputs,
        spike_i=input_spk['indices'],
        spike_t=input_spk['times']
    )

    # Neuronas excitatorias de la capa 1
    neurons['layer1e'] = n_mode.neuron_group_excitatory(
        n_neurons=layer_n_neurons,
        variables=neurons_vars
    )

    # Neuronas inhibitorias de la capa 1
    neurons['layer1i'] = n_mode.neuron_group_inhibitory(
        n_neurons=layer_n_neurons,
        variables=neurons_vars
    )

    return neurons

#############################################################################

def initialize_conn(neurons, connect_vars):
    
    #Iniciamos las conexiones sinapticas entre diferentes capas de neuronas
    
    conns = {}

    # input to layer 1 connections

    source = neurons['input'] #la capa input de spikes del audio
    target = neurons['layer1e'] #capa 1 de excitatorias
    conns['input-layer1e'] = s_mode.synapses_stdpEX(
        source=source,
        target=target,
        connectivity=True, # all-to-all connectivity
        params=connect_vars
    )

    #asignamos peso inicial aleatorio para la conexion entrada - excitatoria 1
    if os.path.exists('input-layer1e-weights.pickle'):
        with open('input-layer1e-weights.pickle', 'rb') as pickle_file:
            pickle_obj = pickle.load(pickle_file)
        conns['input-layer1e'].w = pickle_obj
    else:
        conns['input-layer1e'].w = 'rand() * 0.4'
        #weights = np.array(conns['input-layer1e'].w)
        #guardo los pesos por si necesito trabajar con ellos
        #np.savetxt('evaluation/weights.out', weights) 
        #with open('input-layer1e-weights.pickle', 'wb') as pickle_file:
         #   pickle.dump(weights, pickle_file)

    # conexion excitatory to inhibitory
    conns['layer1e-layer1i'] = s_mode.synapses_non_plastic(
        source=neurons['layer1e'],
        target=neurons['layer1i'],
        connectivity='i == j',
        synapse_type='excitatory'
    )
    conns['layer1e-layer1i'].w = connect_vars['ex-in-w']

    # conexion inhibitory to excitatory
    conns['layer1i-layer1e'] = s_mode.synapses_non_plastic(
        source=neurons['layer1i'],
        target=neurons['layer1e'],
        connectivity='i != j',
        synapse_type='inhibitory'
    )
    conns['layer1i-layer1e'].w = connect_vars['in-ex-w']

    return conns

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

def results_evaluation(monitors, connections, analysis_params):
    
    #Aalisis de los resultados y gráficas (plots)

    if len(monitors['spikes']['layer1e']) == 0:
        print("No spikes detected; not analysing")
        return
    
    end_time = max(monitors['spikes']['layer1e'].t)
    print(end_time)
    start_time = min(monitors['spikes']['layer1e'].t)
    print(start_time)
    a_mode.analyse_note_responses(
        spike_indices=monitors['spikes']['layer1e'].i,
        spike_times=monitors['spikes']['layer1e'].t,
        from_time=start_time,
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

    firing_neurons = set(monitors['spikes']['layer1e'].i)

    a_mode.plot_state_var(
        monitors['neurons']['layer1e'],
        monitors['neurons']['layer1e'].ge/b2.siemens,
        firing_neurons,
        'Current'
    )
    a_mode.plot_state_var(
        monitors['neurons']['layer1e'],
        monitors['neurons']['layer1e'].theta/b2.mV,
        firing_neurons,
        'Threshold increase'
    )
    a_mode.plot_state_var(
        monitors['neurons']['layer1e'],
        monitors['neurons']['layer1e'].v/b2.mV,
        firing_neurons,
        'Membrane potential'
    )

    a_mode.w_diff(
        connections['input-layer1e'],
        monitors['connections']['input-layer1e']
    )

    #Para visualizar los pesos, para cada nuerona tomamos los pesos mas relevantes

    #a_mode.plot_weight(connections['input-layer1e'])


#########################################################################


spike_filename = os.path.basename(run_vars['input_spikes_filename'])
run_id = spike_filename.replace('.pickle', '')
#if not run_params['from_paramfile']:
#    param_mod.record_params(params, run_id)
input_spikes = load_audio_input(run_vars)
input_end_time = np.ceil(np.amax(input_spikes['times']))

if 'run_time' not in run_vars:
    run_vars['run_time'] = input_end_time
if not run_vars['no_standalone']:
    if os.name == 'nt':
        build_dir = 'C:\\temp\\'
    else:
        build_dir = '/tmp/'
    build_dir += run_id
    b2.set_device('cpp_standalone', directory=build_dir)


print("Initialising neurons...")
neurons = initialize_neurons(
    input_spikes, run_vars['layer_n_neurons'],
    neurons_vars
)
print("done!")


print("Initialising connections...")
connections = initialize_conn(
    neurons,
    connect_vars
)
print("done!")


print("Initialising monitors...")
monitors = init_monitors(neurons, connections, mon_vars)
print("done!")

    
print("Running simulation...")
net = run_simulation(run_vars, neurons, connections, monitors)
guardar = False
if(guardar== True):
    weights = np.array(connections['input-layer1e'].w)
        #guardo los pesos por si necesito trabajar con ellos
    np.savetxt('evaluation/weights.out', weights) 
    with open('input-layer1e-weights.pickle', 'wb') as pickle_file:
        pickle.dump(weights, pickle_file)
print("done!")


def save_figures(name):
    print("Saving figures...")
    figs = plt.get_fignums()
    for fig in figs:
        plt.figure(fig)
        plt.savefig('figures/%s_fig_%d.png' % (name, fig))
    print("done!")

results_evaluation(monitors,connections,analysis_vars)

if analysis_vars['save_figs']:
        save_figures(run_id)



#############################################################################################33


