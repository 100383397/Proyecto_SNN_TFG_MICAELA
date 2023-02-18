import numpy as np
import brian2 as b2
import matplotlib.pyplot as plt

#Ahora trabajamos con un grupo de neuronas, ya no es una sino que es N neuronas
#Agregamos la declaración antes de la ejecución. Lo que esto hace es inicializar
#cada neurona con un valor aleatorio uniforme diferente entre 0 y 1.

N = 100
tau = 10*b2.ms
eqs = '''
dv/dt = (2-v)/tau : 1
'''

G = b2.NeuronGroup(N, eqs, threshold='v>1', reset='v=0', method='exact')
#Para trazar los datos al final usamos esta linea
G.v = 'rand()'

spikemon = b2.SpikeMonitor(G)

b2.run(50*b2.ms)

#Además de la variable spikemon.t con los tiempos de todos los picos, también 
#usamos la variable spikemon.i que da el índice de neurona correspondiente 
# para cada pico

plt.figure()
plt.plot(spikemon.t/b2.ms, spikemon.i, '.k')
plt.xlabel('Time (ms)')
plt.ylabel('Neuron index')
plt.savefig('SNN_figures/multiple_neurons.png' )