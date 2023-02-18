
import numpy as np
import brian2 as b2
import matplotlib.pyplot as plt


#Ejemplo de sinapsis mas simple en el que creamos 3 neuronas con la misma ec diferencial
#pero distintos valores de I (corriente impulsora) y tau. La segunda que tiene I = 0
#por si sola sin la sinapsis no se dispara en absoluto

eqs = '''
dv/dt = (I-v)/tau : 1
I : 1
tau : second
'''

#Para 3 neuronas conectamos la 0 a la 1 y 2

G = b2.NeuronGroup(3, eqs, threshold='v>1', reset='v = 0', method='exact')
G.I = [2, 0, 0]
G.tau = [10, 100, 100]*b2.ms

# Definimos el modelo de la sinapsis. En este caso origen y destino son el mismo grupo G
# cuando la neurona fuente dispara pico, la neurona objetivo tendra su valor 
# aumentado en 0,2. ES decir codificamos el peso para que sea el valor 0.2
#S = b2.Synapses(G, G, on_pre='v_post += 0.2')
#S.connect(i=0, j=1)

#Ahora tenemos una variable de peso sinaptico: w
S = b2.Synapses(G, G, 'w : 1', on_pre='v_post += w')
S.connect(i=0, j=[1, 2])
S.w = 'j*0.2' #pesos de 0.2 y 0.4 para neuronas 1 y 2
S.delay = 'j*2*ms' #le podemos peter un retraso

M = b2.StateMonitor(G, 'v', record=True)

b2.run(100*b2.ms)

plt.figure()
plt.plot(M.t/b2.ms, M.v[0], label='Neuron 0')
plt.plot(M.t/b2.ms, M.v[1], label='Neuron 1')
plt.plot(M.t/b2.ms, M.v[2], label='Neuron 2')
plt.xlabel('Time (ms)')
plt.ylabel('v')
plt.legend()
plt.savefig('SNN_figures/sinapsis.png' )