#imports
from __future__ import print_function, division
import sys
import os.path
import pickle
import brian2 as b2
import numpy as np
import matplotlib.pyplot as plt
import tools.synapses as s_mode
import tools.groups_neurons as n_mode
import tools.analysis as a_mode

# Este script es el principal, y contiene la estructura de la SNN, así como su ejecución, monitorización
# y representación.

# Primero se recogen y se definen los parametros usados para: neuronas, conexiones, monitores y
# para la ejecución de la simulación.

neurons_vars = {} # Variables ecuaciones grupos neuronales

neurons_vars['resting_pot_E'] = -66 * b2.mV # Pot reposo excitatoria (E)
neurons_vars['resting_pot_I'] = -61 * b2.mV # Pot reposo inhibitoria (I)
neurons_vars['reset_pot_E'] = -65 * b2.mV # Pot reset E 
neurons_vars['reset_pot_I'] = -49 * b2.mV # Pot reset I
neurons_vars['threshold_E'] = -52 * b2.mV # umbral E
neurons_vars['threshold_I'] = -40 * b2.mV # umbral I
neurons_vars['refractorytime_E'] = 5 * b2.ms # Periodo refractario E
neurons_vars['refractorytime_I'] = 2 * b2.ms # Periodo refractario I
neurons_vars['tau_pot_E'] = 100 * b2.ms # cte t pot membrana E
neurons_vars['tau_pot_I'] = 10 * b2.ms # cte t pot membrana I
neurons_vars['tau_gE'] = 1 * b2.ms # cte t conductancia E
neurons_vars['tau_gI'] = 2 * b2.ms # cte t conductancia I
neurons_vars['inv_pot_ex_E'] = 0 * b2.mV # Pot inversion sinaptica excitatorio E
neurons_vars['inv_pot_in_E'] = -100 * b2.mV # Pot inversion sinaptica inhibitorio E
neurons_vars['inv_pot_ex_I'] = 0 * b2.mV # Pot inversion sinaptica excitatorio I
neurons_vars['inv_pot_in_I'] = -86 * b2.mV # Pot inversion sinaptica inhibitorio I
neurons_vars['tau_theta'] = 1e6 * b2.ms
neurons_vars['min_theta'] = 0 * b2.mV
neurons_vars['max_theta'] = 60.0 * b2.mV
neurons_vars['offset'] = 20 * b2.mV
neurons_vars['theta_coef'] = 0.02

connect_vars = {} # Variables de STDP

connect_vars['tau_pre'] = 20 * b2.ms # cte t presinaptina
connect_vars['tau_post'] = 20 * b2.ms # cte t postsinaptica
connect_vars['nu_pre'] = 0.0001 # tasa aprendizaje presinaptica
connect_vars['nu_post'] = 0.02 # tasa aprendizaje postsinaptica
connect_vars['wmax'] = 1.0 
connect_vars['pre_w_decrease'] = 0.00025 # disminución w presinaptico
connect_vars['min_theta'] = 0 * b2.mV
connect_vars['max_theta'] = 60 * b2.mV
connect_vars['theta_coef'] = 0.02
connect_vars['ex-in-w'] = 10.4 #PESO
connect_vars['in-ex-w'] = 17.0 #PESO

run_vars = {} #variables de ejecución

run_vars['layer_n_neurons'] = 12 # Numero de neuronas de la capa de salida
run_vars['input_spikes_filename'] = sys.argv[1] #'spikes_inputs_train/scale_0.5_s.pickle'
run_vars['output_spikes_filename'] = sys.argv[2]
run_vars['no_standalone'] = True

mon_vars = {} #variable diferencial de tiempo de monitorizado

mon_vars['mon_dt'] = 1000/60.0 * b2.ms

analysis_vars = {}

analysis_vars['save_figs'] = True
analysis_vars['note_length'] = float(sys.argv[3]) #0.5, 0.1 o 0.2

analysis_vars['n_notes'] = 5 # Numero de notas contenidas en cada audio.

# Debe modificarse para cada ejecucion en función de la secuencia de audio introducida, entre 
# parentesis se muestran a continuación el numero que se debe poner para cada audio:

# scale12 (12), scale7 (7), melody1 (6), melody2 (9), melody3 (15), melody4 (16), melody5 (19)
# Bach (64), Minuet (40), Sonatine (31), Waltz (35), Yiruma (35), chords0 (5), chords1 (4),
# chords2 (6), chords3 (9), chords4 (7), chords5 (6)

variables = (neurons_vars, connect_vars, mon_vars, run_vars, analysis_vars)


# Se cargan los spikes del audio preprocesado para emplearse como entrada a la red neuronal,
# se cargan sus índices y tiempos.

def load_audio_input(run_vars):

    filename = run_vars['input_spikes_filename']
    with open(filename, 'rb') as pickle_file:
        (in_spike_t, in_spike_i) = pickle.load(pickle_file)
    in_spike_t = in_spike_t * b2.second

    spikes = {}
    spikes['indices'] = in_spike_i
    spikes['times'] = in_spike_t

    return spikes


# Se inicializan los grupos de neuronas excitatorias de entrada, salida y las inhibidoras

def initialize_neurons(input_spk, layer_n_neurons, neurons_vars):

    neurons = {}

    n_inputs = 512 # Numero neuronas empleadas en capa de entrada del audio.

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


# Se inicializan las conexiones sinapticas entre los grupos de neuronas

def initialize_conn(neurons, connect_vars):
    
    conns = {}

    # Conexion entrada a capa 1 excitatoria de salida

    source = neurons['input'] # Capa input formada por el tren de spikes el audio de entrada
    target = neurons['layer1e'] # Capa de excitatorias
    conns['input-layer1e'] = s_mode.synapses_stdpEX(
        source=source,
        target=target,
        connectivity=True, # Conexion todas con todas
        params=connect_vars
    )

    # Se comprueba si ya existen pesos guardados, y sino,
    # asigna peso inicial aleatorio para la conexion entrada - excitatoria 1

    if os.path.exists('weights.pickle'):
        with open('weights.pickle', 'rb') as pickle_file:
            pickle_obj = pickle.load(pickle_file)
        conns['input-layer1e'].w = pickle_obj
    else:
        conns['input-layer1e'].w = 'rand() * 0.4'

    # Conexion excitatoria a inhibitoria

    conns['layer1e-layer1i'] = s_mode.synapses_non_plastic(
        source=neurons['layer1e'],
        target=neurons['layer1i'],
        connectivity='i == j', # Conexion con inhibidora de mismo indice
        synapse_type='excitatory'
    )
    conns['layer1e-layer1i'].w = connect_vars['ex-in-w']

    # Conexion inhibitoria a excitatoria

    conns['layer1i-layer1e'] = s_mode.synapses_non_plastic(
        source=neurons['layer1i'],
        target=neurons['layer1e'],
        connectivity='i != j', # Conexion a todas las excitatorias menos la de mismo indice
        synapse_type='inhibitory'
    )
    conns['layer1i-layer1e'].w = connect_vars['in-ex-w']

    return conns


# Función para monitorizar las variables de estado en la red neuronal mediante objetos de Brian 2

def variable_monitors(neurons, conns, mon_vars):

    mon = {
        'spikes': {}, # Para monitorizar y registrar los picos (spikes, con SpikeMonitor)
        'neurons': {}, # Para registrar variables de las neuronas: v, ge, theta....
        'connections': {} # Para registrar variables de las conexiones generadas: pesos
    }

    for layer in ['input', 'layer1e']:
        mon['spikes'][layer] = b2.SpikeMonitor(neurons[layer])

    if 'mon_dt' not in mon_vars:
        t_step = None
    else:
        t_step = mon_vars['mon_dt']

    mon['neurons']['layer1e'] = b2.StateMonitor(
        neurons['layer1e'],
        ['v', 'ge', 'max_ge', 'theta'],
        record=range(len(neurons['layer1e'])),
        dt=t_step
    )

    conn = conns['input-layer1e']
    n_conns = len(conn.target) * len(conn.source)

    mon['connections']['input-layer1e'] = b2.StateMonitor(
        conns['input-layer1e'],
        ['w', 'post', 'pre'],
        record=range(n_conns),
        dt=t_step
    )
    return mon


# Función para ejecutar la simulación de la red neuronal con los objetos creados hasta el momento

def run_simulation(run_vars, neurons, conns, monitors):

    net = b2.Network()
    for group in neurons:
        net.add(neurons[group])
    for connection in conns:
        net.add(conns[connection])
    for mon_type in monitors:
        for neuron_group in monitors[mon_type]:
            net.add(monitors[mon_type][neuron_group])

    net.run(run_vars['run_time'], report='text')

    return net

# Función para evaluar los parámetros, gráfica y analisis de los resultados

def data_analysis(mon, conns):

    if len(mon['spikes']['layer1e']) == 0:
        print("No se detectaron picos; no hay analisis")
        return
    
    end_time = max(mon['spikes']['layer1e'].t)
    start_time = min(mon['spikes']['layer1e'].t)

    a_mode.note_data_responses(
        spike_i=mon['spikes']['layer1e'].i,
        spike_t=mon['spikes']['layer1e'].t,
        from_t=start_time,
        to_t=end_time,
        note_t = analysis_vars['note_length'], # Distancia entre las notas en segundos
        n_notes = analysis_vars['n_notes'] # Numero de notas en el audio, declarado al principio del script
    )

    # Para calcular la media y la varianza del entrenamiento y caracterizar asi el sistema,
    # se va a calcular la media y varianza de los índices de la capa excitatoria de salida
    # La media es razonable entorno a 5.5, ya que hay 12 notas (indices de 0 a 11).
    # (Se espera que para distintas duraciones de las notas, a mayor tiempo entre notas si 
    # la dispersion es mas alta, es menos fiable, y si es mas baja lo contrario).

    mean_i = np.mean(mon['spikes']['layer1e'].i)
    var_i = np.var(mon['spikes']['layer1e'].i)
    std_i = np.std(mon['spikes']['layer1e'].i)

    print("La media calculada en función de los indices es: %f"%( mean_i))
    print("La varianza resultante es: %f" %( var_i))
    print("La desviacion tipica resultante es: %f" %( std_i))
    
    plt.ion()

    # Representación de spikes de entrada y salida, umbral, corriente, potencial de membrana y
    # ajuste de pesos para cada neurona

    plt.figure()

    plt.subplot(2, 1, 1)
    plt.title("Spikes de entrada")
    plt.plot(
        mon['spikes']['input'].t/b2.second,
        mon['spikes']['input'].i,
        'k.',
        markersize=2
    )
    plt.ylabel("Neurona nº.")
    plt.grid()

    plt.subplot(2, 1, 2)
    plt.title("Spikes de salida")
    plt.plot(
        mon['spikes']['layer1e'].t/b2.second,
        mon['spikes']['layer1e'].i,
        'k.',
        markersize=2
    )
    plt.ylim([-1, max(mon['spikes']['layer1e'].i)+1])
    plt.grid()
    plt.ylabel("Neurona nº.")
    plt.xlabel("Tiempo (s)")
    plt.tight_layout()

    fir_neurons = set(mon['spikes']['layer1e'].i)

    a_mode.params_figures(
        mon['neurons']['layer1e'],
        mon['neurons']['layer1e'].ge/b2.siemens,
        fir_neurons,
        'Corrinete'
    )
    a_mode.params_figures(
        mon['neurons']['layer1e'],
        mon['neurons']['layer1e'].theta/b2.mV,
        fir_neurons,
        'Aumento del umbral'
    )
    a_mode.params_figures(
        mon['neurons']['layer1e'],
        mon['neurons']['layer1e'].v/b2.mV,
        fir_neurons,
        'Potencial de membrana'
    )

    a_mode.w_diff_figure(
        conns['input-layer1e'],
        mon['connections']['input-layer1e']
    )


# Inicialización de la simulación y ejecución de las funciones definidas previamente

spk_filename = os.path.basename(run_vars['input_spikes_filename'])
run_id = spk_filename.replace('.pickle', '')
in_spk = load_audio_input(run_vars)
in_end_t = np.ceil(np.amax(in_spk['times']))

if 'run_time' not in run_vars:
    run_vars['run_time'] = in_end_t

# Si se quier generar el código en C++ cambiar a false la variable run_vars['no_standalone'] al principio

if not run_vars['no_standalone']:
    if os.name == 'nt':
        build_dir = 'C:\\temp\\'
    else:
        build_dir = '/tmp/'
    build_dir += run_id
    b2.set_device('cpp_standalone', directory=build_dir)

# Seguimiento de la simulación y las inicializaciones de los parámetros

print("Inicializando neuronas...")
neurons = initialize_neurons(in_spk, run_vars['layer_n_neurons'],neurons_vars)
print("¡Listo!")

print("Inicializando conexiones...")
conns = initialize_conn(neurons,connect_vars)
print("¡Listo!")

print("Inicializando monitores...")
monitors = variable_monitors(neurons, conns, mon_vars)
print("¡Listo!")
   
print("Ejecutando simulacion...")
net = run_simulation(run_vars, neurons, conns, monitors)


guardar = False # Cambiar a True si quiere sobreescribir el archivo de pesos weight.pickle 
if(guardar== True):
    weights = np.array(conns['input-layer1e'].w)
    #guardo los pesos por si necesito trabajar con ellos
    #np.savetxt('evaluation/weights.out', weights)
    filename = run_vars['output_spikes_filename']
    with open(filename, 'wb') as pickle_file:
        pickle.dump(weights, pickle_file)
print("¡Listo!")


def save_figs(name):
    print("Guardando figuras...")
    figs = plt.get_fignums()
    for fig in figs:
        plt.figure(fig)
        plt.savefig('Results/figures/%s_fig_%d.png' % (name, fig))
    print("¡Listo!")

data_analysis(monitors,conns)

if analysis_vars['save_figs']:
        save_figs(run_id)

