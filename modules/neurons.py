import numpy as np
import brian2 as b2

def prespecified_spike_neurons(n_neurons, spike_indices, spike_times):
    neurons = b2.SpikeGeneratorGroup(
        N=n_neurons, indices=spike_indices, times=spike_times)
    return neurons

def excitatory_neurons(n_neurons, params):
    neuron_params = {
        'v_thresh_e': params['v_thresh_e'],
        'v_reset_e': params['v_reset_e'],
        'v_rest': params['v_rest_e'],
        'tc_v': params['tc_v_ex'],
        'e_ex': params['e_ex_ex'],
        'e_in': params['e_in_ex'],
        'tc_ge': params['tc_ge'],
        'tc_gi': params['tc_gi'],
        'tc_theta': params['tc_theta'],
        'theta_coef': params['theta_coef'],
        'max_theta': params['max_theta'],
        'min_theta': params['min_theta'],
        'offset': params['offset']
    }

    reset_e = '''
    v = v_reset_e
    theta = theta + theta_coef * (max_theta - theta)
    '''

    thresh_e = 'v > (theta - offset + v_thresh_e)'

    neuron_eqs = '''
    I_synE = ge * (e_ex - v) : amp
    I_synI = gi * (e_in - v) : amp
    dge/dt = -ge / tc_ge     : siemens
    dgi/dt = -gi / tc_gi     : siemens
    '''
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
        refractory=params['refrac_e'],
        reset=reset_e,
        namespace=neuron_params,
        method='euler' # automatically suggested by Brian
    )
    neurons.v = params['v_rest_e']
    neurons.theta = \
        np.ones((n_neurons)) * params['offset']
    neurons.theta_mod = np.ones((n_neurons))

    return neurons

def inhibitory_neurons(n_neurons, params):
    neuron_params = {
        'v_thresh_i': params['v_thresh_i'],
        'v_reset_i': params['v_reset_i'],
        'v_rest': params['v_rest_i'],
        'tc_v': params['tc_v_in'],
        'e_ex': params['e_ex_in'],
        'e_in': params['e_in_in'],
        'tc_ge': params['tc_ge'],
        'tc_gi': params['tc_gi'],
        'tc_gi': params['tc_gi']
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
        refractory=params['refrac_i'], 
        reset=reset_i,
        namespace=neuron_params,
        method='euler' # automatically suggested by Brian
    )
    neurons.v = params['v_rest_i']

    return neurons

"""def visualisation_neurons(n_neurons, params):
    model = '''
    dv/dt = (v_rest - v)/tc_v : volt
    '''
    neuron_params = {
        'v_rest': params['v_rest_e'],
        'tc_v': 10 * b2.ms
    }
    neurons = b2.NeuronGroup(
        N=n_neurons,
        model=model,
        namespace=neuron_params,
        method='euler'
    )
    neurons.v = params['v_rest_e']

    return neurons"""
