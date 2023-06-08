
import numpy as np
import brian2 as b2

# En este script se definen las funciones que generan los grupos de neuronas de entrada y 
# las neuronas excitatorias de salida e inhibidoras.

# Función para generar el grupo de neuronas excitatorias de entrada. Traduccion del audio de entrada a 
# un tren de picos, con unos índices y tiempos.

def audio_spike_neurons(n_neurons, spike_i, spike_t): 
    neurons = b2.SpikeGeneratorGroup(N=n_neurons, indices=spike_i, times=spike_t)
    return neurons

# Función para generar el grupo de neuronas excitatorias de salida, se recogen sus variables del modelo
# de ecuaciones empleado, el umbral adaptativo, el reset y su período refractario.

def neuron_group_excitatory(n_neurons, variables):
    
    neurons_vars = {
        'thresh_E': variables['threshold_E'],
        'reset_E': variables['reset_pot_E'],
        'resting_E': variables['resting_pot_E'],
        'tau': variables['tau_pot_E'],
        'pot_ex': variables['inv_pot_ex_E'],
        'pot_in': variables['inv_pot_in_E'],
        'tau_gE': variables['tau_gE'],
        'tau_gI': variables['tau_gI'],
        'tau_theta': variables['tau_theta'],
        'theta_coef': variables['theta_coef'],
        'max_theta': variables['max_theta'],
        'min_theta': variables['min_theta'],
        'offset': variables['offset']
    }

    reset_e = '''
    v = reset_E
    theta = theta + theta_coef * (max_theta - theta)
    '''

    thresh_e = 'v > (theta - offset + thresh_E)'

    # La corriente producida por la conductancia varía segun las siguientes 2 ecuaciones de I
    # La conductancia sigue una caida exponencial (expresiones de las derivadas)
    # Un pico presinaptico provoca un aumento instantaneo de la conductancia y la conductancia 
    # provoca una corriente que atrae el potencial de membrana hacia pot_ex

    neuron_eqs = '''
    I_synE = ge * (pot_ex - v) : amp
    I_synI = gi * (pot_in - v) : amp
    dge/dt = -ge / tau_gE    : siemens
    dgi/dt = -gi / tau_gI     : siemens
    '''

    # Se emplea inhibicion de corto alcance. Para ello hay que indicar la posicion espacial
    # de cada neurona en el modelo usando dos variables de estado: x, y

    neuron_eqs_e = neuron_eqs + '''
    dv/dt = ((resting_E - v) + (I_synE + I_synI) * 1 * ohm) / tau : volt (unless refractory)
    dtheta/dt = -theta / (tau_theta)                             : volt
    theta_mod                                                   : 1
    max_ge                                                      : siemens
    x                                                           : 1
    y                                                           : 1
    '''

    # Definición del grupo de neuronas excitatorias e inicialización

    neurons = b2.NeuronGroup(
        N=n_neurons,
        model=neuron_eqs_e, 
        threshold=thresh_e,
        refractory=variables['refractorytime_E'],
        reset=reset_e,
        namespace=neurons_vars,
        method='euler'  # sugerido automáticamente por Brian 2
    )
    neurons.v = variables['resting_pot_E']
    neurons.theta = np.ones((n_neurons)) * variables['offset']
    neurons.theta_mod = np.ones((n_neurons))

    return neurons

# Función para generar el grupo de neuronas inhibitorias, se recogen sus variables del modelo
# de ecuaciones empleado, el umbral adaptativo, el reset y su período refractario.

def neuron_group_inhibitory(n_neurons, variables):
    neurons_vars = {
        'thresh_I': variables['threshold_I'],
        'reset_I': variables['reset_pot_I'],
        'resting_I': variables['resting_pot_I'],
        'tau': variables['tau_pot_I'],
        'pot_ex': variables['inv_pot_ex_I'],
        'pot_in': variables['inv_pot_in_I'],
        'tau_gE': variables['tau_gE'],
        'tau_gI': variables['tau_gI']
    }
        
    reset_i = 'v = reset_I'
    thresh_i = 'v > thresh_I'

    # La corriente producida por la conductancia varía segun las siguientes 2 ecuaciones de I

    neuron_eqs = '''
    I_synE = ge * (pot_ex - v) : amp
    I_synI = gi * (pot_in - v) : amp
    dge/dt = -ge / tau_gE     : siemens
    dgi/dt = -gi / tau_gI     : siemens
    '''

    neuron_eqs_i = neuron_eqs + '''
    dv/dt = ((resting_I - v) + (I_synE + I_synI) * 1 * ohm) / tau : volt
    '''

    # Definición del grupo de neuronas inhibidoras e inicialización

    neurons = b2.NeuronGroup(
        N=n_neurons,
        model=neuron_eqs_i, 
        threshold=thresh_i,
        refractory=variables['refractorytime_I'], 
        reset=reset_i,
        namespace=neurons_vars,
        method='euler' # sugerido automáticamente por Brian 2 
    )
    neurons.v = variables['resting_pot_I']

    return neurons

######################################################################################

# La dinámica temporal del potencial de membrana de la neurona es la primera decision
# que se concreta. Se emplea el modelo de integracion y disparo por fuga (LIF), donde el 
# potencial de membrana se modela con una ecuación diferencial unica. 
# También es importante el efecto de una neurona sobre las que esta conectadas y la forma 
# facil de implementarlo es considerando el mecanismo biológico donde los neurotransmisores en
# el pico presinaptico cambian la conductancia de la membrana postsinaptica 
# (ecuaciones de conductancia)

######################################################################################

