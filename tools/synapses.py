
import brian2 as b2

# Se consideran los modelos de STDP dependientes del tiempo, aunque el potencial y la frecuencia de
# la membrana postsinaptica modulan el efecto de STDP.

# La implementacion de STDP se lleva a cabo siguiendo el segundo tutorial de Brian, haciendo el cálculo
# de STDP empleando el sistema de rastreo sinaptico. SU implementación en la forma mas basica implica
# una actualización de los valores de los pesos sinapticos en el momento de un pico postsinaptico empleando
# una funcion exponencial por partes. De esta forma el cambio de peso no depende del valor actual del peso
# ESto se conoce como STDP aditivo 


def synapses_stdpEX(source, target, connectivity, params):
    syn_params = {
        'tc_pre_ee': params['tc_pre_ee'],
        'tc_post_ee': params['tc_post_ee'],
        'nu_ee_pre': params['nu_ee_pre'],
        'pre_w_decrease': params['pre_w_decrease'],
        'wmax_ee': params['wmax_ee'],
        'nu_ee_post': params['nu_ee_post'],
        'theta_coef': params['theta_coef'],
        'min_theta': params['min_theta'],
        'max_theta': params['max_theta']
    }

    # Para cada sinapsis se especifican las variables de estado que representan el peso y los rastros
    # sinapticos (las derivadas del modelo de ecuaciones que viene a continuacion)

    eqs_stdp_ee = '''
    w                             : 1
    dpre/dt = -pre / tc_pre_ee    : 1 (event-driven)
    dpost/dt = -post / tc_post_ee : 1 (event-driven)
    '''

    #Definimos la accion a realizar cuando la sinapsis recibe un pico presinaptico y postsinaptico

    eqs_stdp_pre_ee = '''
    ge_post += w * siemens
    pre = 1
    w = clip(w - nu_ee_pre * post - pre_w_decrease, 0, wmax_ee)
    '''

    eqs_stdp_post_ee = '''
    post = 1
    w = clip(w + nu_ee_post * pre, 0, wmax_ee)
    '''

    synapses = b2.Synapses(
        source=source,
        target=target,
        model=eqs_stdp_ee,
        on_pre=eqs_stdp_pre_ee,
        on_post=eqs_stdp_post_ee,
        namespace=syn_params
    )

    synapses.connect(connectivity)

    return synapses


def synapses_non_plastic(source, target, connectivity, synapse_type):
    if synapse_type == 'excitatory':
        pre = 'ge_post += w * siemens'
    elif synapse_type == 'inhibitory':
        pre = 'gi_post += w * siemens'
    else:
        raise Exception("Invalid synapse type: %s" % synapse_type)

    model = 'w : 1'

    synapses = b2.Synapses(
        source=source,
        target=target,
        model=model, on_pre=pre
    )

    synapses.connect(connectivity)

    return synapses
