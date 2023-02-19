import numpy as np
import brian2 as b2
import matplotlib.pyplot as plt


N = 10
G = b2.NeuronGroup(N, 'v:1')
S = b2.Synapses(G, G)

S.connect(condition='i!=j', p=0.5)

#Otra condición de conectividad que solo conecta las neuronas vecinas seria la siguiente:
#S.connect(condition='abs(i-j)<4 and i!=j')

#S.connect(j='k for k in range(i-3, i+4) if i!=k', skip_if_invalid=True)
#n esta otra linea skip_if_invalid evita errores en los limites.

#S.connect(j='i') esta linea conecta cada neurona origen con neurona destino. conectividad 1 a 1

#La imagen generada muestra en ambas gráficas lo mismo de dstinta
#forma. La de la izquierda tiene las neuronas de origena la izq y las
#de destino a la derecha y la linea entre las neuronas que tienen
#sinapsis.
#Si cambiamos la probabilidad de una conexion se modifican las cifras
#generadas


def visualise_connectivity(S):
    Ns = len(S.source)
    Nt = len(S.target)
    plt.figure(figsize=(10, 4))
    plt.subplot(121)
    plt.plot(np.zeros(Ns), np.arange(Ns), 'ok', ms=10)
    plt.plot(np.ones(Nt), np.arange(Nt), 'ok', ms=10)
    for i, j in zip(S.i, S.j):
        plt.plot([0, 1], [i, j], '-k')
    plt.xticks([0, 1], ['Source', 'Target'])
    plt.ylabel('Neuron index')
    plt.xlim(-0.1, 1.1)
    plt.ylim(-1, max(Ns, Nt))
    plt.subplot(122)
    plt.plot(S.i, S.j, 'ok')
    plt.xlim(-1, Ns)
    plt.ylim(-1, Nt)
    plt.xlabel('Source neuron index')
    plt.ylabel('Target neuron index')
    plt.savefig('SNN_figures/sinapsis_compleja.png' )

visualise_connectivity(S)