
import numpy as np
import brian2 as b2
import matplotlib.pyplot as plt

#En Brian 2 todos los modelos de neuronas simples estan definidos por ecuaciones diferenciales
N= 1
tau = 10* b2.ms
eqs = '''
dv/dt = (1-v)/tau : 1 (unless refractory)
'''
#El comportamiento de picos se genera añadiendo los parametros de threshold y reset
# Parametro refractory: significa que después de que la neurona dispara un pico, se 
# vuelve refractaria durante un cierto tiempo y no puede disparar otro pico hasta 
# que finaliza este período
G = b2.NeuronGroup(N, eqs, threshold='v>0.8', refractory=5*b2.ms, reset='v = 0', method='exact')
# El StateMonitor se usa para registrar los valores de una variable de neurona 
# mientras se ejecuta la simulación. Record = 0 seria que se registran los valores
# para la neurona 0
M = b2.StateMonitor(G, 'v', record=0)
#Brian registra los picos. SpikeMonitor toma el grupo cuyos picos desea 
#registrar como su argumento y almacena los tiempos de los picos en la variable
spikemon = b2.SpikeMonitor(G)
# Ejecuto la simulacion durante 100 ms
b2.run(100*b2.ms)

print('Spike times: %s' % spikemon.t[:])

plt.figure()
#plt.plot(M.t/b2.ms, M.v[0], 'C0', label='Brian')
plt.plot(M.t/b2.ms, M.v[0])
for t in spikemon.t:
    plt.axvline(t/b2.ms, ls='--', c='C1', lw=3)#dibuja linea vertical dicontinua en los picos
plt.xlabel('Time (ms)')
plt.ylabel('v')
plt.savefig('SNN_figures/one_neuron.png' )

