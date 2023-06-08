
import brian2 as b2

# En este script se definen las funciones que generan las conexiones, las sinapsis, entre losç
# grupos de nueronas. Tendremos STDP y sinapsis no plástica-

# Definición de la función para el método STDP

def synapses_stdpEX(source, target, connectivity, params):
    syn_params = {
        'tau_pre': params['tau_pre'],
        'tau_post': params['tau_post'],
        'nu_pre': params['nu_pre'],
        'nu_post': params['nu_post'],
        'wmax': params['wmax'],
        'pre_w_decrease': params['pre_w_decrease'],
        'theta_coef': params['theta_coef'],
        'min_theta': params['min_theta'],
        'max_theta': params['max_theta']
    }

    # Para cada sinapsis se especifican las variables de estado que representan el peso y los rastros
    # sinapticos (las derivadas del modelo de ecuaciones que viene a continuacion)

    eqs_stdp = '''
    w                             : 1
    dpre/dt = -pre / tau_pre      : 1 (event-driven)
    dpost/dt = -post / tau_post   : 1 (event-driven)
    '''

    #Definimos la accion a realizar cuando la sinapsis recibe un pico presinaptico y postsinaptico

    eqs_stdp_pre = '''
    ge_post += w * siemens
    pre = 1
    w = clip(w - nu_pre * post - pre_w_decrease, 0, wmax)
    '''

    eqs_stdp_post = '''
    post = 1
    w = clip(w + nu_post * pre, 0, wmax)
    '''

    # Definición de la sinapsis STDP

    synapses = b2.Synapses(
        source=source,
        target=target,
        model=eqs_stdp,
        on_pre=eqs_stdp_pre,
        on_post=eqs_stdp_post,
        namespace=syn_params
    )

    synapses.connect(connectivity)

    return synapses

# Definición de la función de conexiónes no plásticas entre inhibidoras y excitatorias

def synapses_non_plastic(source, target, connectivity, synapse_type):
    if synapse_type == 'excitatory':
        pre = 'ge_post += w * siemens'
    elif synapse_type == 'inhibitory':
        pre = 'gi_post += w * siemens'
    else:
        raise Exception("Tipo de sinapsis no válida: %s" % synapse_type)

    model = 'w : 1'

    # Definición de la sinapsis de conexiones no plásticas

    synapses = b2.Synapses(
        source=source,
        target=target,
        model=model, on_pre=pre
    )

    synapses.connect(connectivity)

    return synapses

##############################################################################################

# Se toma el modelo de STDP dependiente del tiempo, aunque el potencial y la frecuencia de
# la membrana postsinaptica tambien modulan el efecto de STDP.

# La implementacion de STDP se lleva a cabo siguiendo el segundo tutorial de Brian, empleando el 
# sistema de rastreo sinaptico. SU implementación más básica implica una actualización de los 
# valores de los pesos sinapticos en el momento de un pico postsinaptico, empleando una funcion 
# exponencial por partes. 

##############################################################################################