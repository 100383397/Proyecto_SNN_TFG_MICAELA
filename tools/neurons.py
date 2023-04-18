
import numpy as np
import brian2 as b2

######################################################################################

# La dinámica temporal del potencial de membrana de la neurona es la primera decision
# a tomar. En este caso suponiendo que el reconocimiento de los patrones musicales va 
# a depender mas de la regla de la plasticidad que de la dinamica del potencial de membrana
# se va a emplear el modelo de integracion y disparo por fuga (LIF), donde el pot de membrana
# se va a modelar con una ec. diferencial unica. 
# Este modelo sirve para una neurona, pero nos interesa la conexion de varias neuronas, por 
# tanto es importante el efecto de una neurona sobre las que esta conectadas (cómo modelar 
# potencial postsinaptico) y la forma facil de implementar este modelo es considerando los
# mecanismos biologicos subyacentes, donde los neurotransmisores liberados en el pico 
# presinaptico cambia la conductancia de la membrana postsinaptica ( no directamente el potencial)

# Elproceso de integracion de las neuronas LIF es una especie de tira y afloja de potenciales
# electricos. El punto clave es que este proceso refleja la fuerza relativa excitatoria vs 
# inhibitoria. Si la excitacion es mas fuerte que la inhibicion, el potencial electrico de la 
# neurona aumenta tal vez  hasta el punto de superar el umbral y disparar un potencial de acción 
# de salida. Si la inhibición es más fuerte, entonces el potencial eléctrico de la neurona 
# disminuye, y así se aleja más de superar el umbral para disparar.

######################################################################################

def audio_spike_neurons(n_neurons, spike_i, spike_t):
    
    neurons = b2.SpikeGeneratorGroup(N=n_neurons, indices=spike_i, times=spike_t)
    
    return neurons


def neuron_group_excitatory(n_neurons, variables):
    neurons_vars = {
        'v_thresh_e': variables['v_thresh_e'],
        'v_reset_e': variables['v_reset_e'],
        'v_rest': variables['v_rest_e'],
        'tc_v': variables['tc_v_ex'],
        'e_ex': variables['e_ex_ex'],
        'e_in': variables['e_in_ex'],
        'tc_ge': variables['tc_ge'],
        'tc_gi': variables['tc_gi'],
        'tc_theta': variables['tc_theta'],
        'theta_coef': variables['theta_coef'],
        'max_theta': variables['max_theta'],
        'min_theta': variables['min_theta'],
        'offset': variables['offset']
    }

    reset_e = '''
    v = v_reset_e
    theta = theta + theta_coef * (max_theta - theta)
    '''

    thresh_e = 'v > (theta - offset + v_thresh_e)'

    #la corriente producida por la conductancia varía segun las siguientes 2 ecuaciones de I
    neuron_eqs = '''
    I_synE = ge * (e_ex - v) : amp
    I_synI = gi * (e_in - v) : amp
    dge/dt = -ge / tc_ge     : siemens
    dgi/dt = -gi / tc_gi     : siemens
    '''

    #Se emplea inhibicion de corto alcance. Para ello hay que indicar la posicion espacial
    # de cada neurona en el modelo usando dos variables de estado: x, y
    neuron_eqs_e = neuron_eqs + '''
    dv/dt = ((v_rest - v) + (I_synE + I_synI) * 1 * ohm) / tc_v : volt (unless refractory)
    dtheta/dt = -theta / (tc_theta)                             : volt
    theta_mod                                                   : 1
    max_ge                                                      : siemens
    x                                                           : 1
    y                                                           : 1
    '''

    neurons = b2.NeuronGroup(
        N=n_neurons,
        model=neuron_eqs_e, 
        threshold=thresh_e,
        refractory=variables['refrac_e'],
        reset=reset_e,
        namespace=neurons_vars,
        method='euler' 
    )
    neurons.v = variables['v_rest_e']
    neurons.theta = np.ones((n_neurons)) * variables['offset']
    neurons.theta_mod = np.ones((n_neurons))

    return neurons

def neuron_group_inhibitory(n_neurons, variables):
    neurons_vars = {
        'v_thresh_i': variables['v_thresh_i'],
        'v_reset_i': variables['v_reset_i'],
        'v_rest': variables['v_rest_i'],
        'tc_v': variables['tc_v_in'],
        'e_ex': variables['e_ex_in'],
        'e_in': variables['e_in_in'],
        'tc_ge': variables['tc_ge'],
        'tc_gi': variables['tc_gi']
    }
        
    reset_i = 'v = v_reset_i'
    thresh_i = 'v > v_thresh_i'

    #La corriente producida por la conductancia varia segun las expresiones de I_syn...
    #La conductancia sigue una caida exponencial (expresiones de las derivadas)
    #un pico presinaptico provoca un aumento instantaneo de la conductancia
    #la conductancia provoca  una corriente que atrae el potencial de membrana hacia e_ex
    neuron_eqs = '''
    I_synE = ge * (e_ex - v) : amp
    I_synI = gi * (e_in - v) : amp
    dge/dt = -ge / tc_ge     : siemens
    dgi/dt = -gi / tc_gi     : siemens
    '''

    neuron_eqs_i = neuron_eqs + '''
    dv/dt = ((v_rest - v) + (I_synE + I_synI) * 1 * ohm) / tc_v : volt
    '''

    neurons = b2.NeuronGroup(
        N=n_neurons,
        model=neuron_eqs_i, 
        threshold=thresh_i,
        refractory=variables['refrac_i'], 
        reset=reset_i,
        namespace=neurons_vars,
        method='euler' 
    )
    neurons.v = variables['v_rest_i']

    return neurons

