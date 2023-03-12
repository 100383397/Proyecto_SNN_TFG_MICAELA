##!/usr/bin/env python

from __future__ import print_function, division
import brian2 as b2
import matplotlib.pyplot as plt
import numpy as np
import os.path
import pickle
import argparse
from scipy import ndimage
import librosa

#Cargo los archivos de audio que he creado y grabado con Mingus

parser = argparse.ArgumentParser()
parser.add_argument('wav_file')
parser.add_argument('--interactive', action='store_true')
args = parser.parse_args()

input_filename = args.wav_file
input_name = os.path.basename(input_filename).replace(".wav", "")

#Cargamos el audio haciendo uso de librosa y realizamos un cronograma  de Q constante

x, sr = librosa.load(input_filename)
time= librosa.get_duration(y=x, sr=sr)
                     
#fmin = librosa.midi_to_hz(36)
hop_length = 1024
C =  np.abs(librosa.cqt(x, sr=sr, n_bins=72,  hop_length=hop_length))

if args.interactive:
    plt.ion()

logC = librosa.amplitude_to_db(C, ref=np.max)
plt.figure(figsize=(15, 5))
librosa.display.specshow(logC, sr=sr, x_axis='time', y_axis='cqt_note')
plt.ylabel('Frecuencia (Hz)')
plt.xlabel('Tiempo (s)')
plt.colorbar(format='%+2.0f dB')
plt.title('Constant-Q power spectrum')
plt.tight_layout()
plt.savefig('images/%s_chroma_cq.png' % input_name)

chroma_spectral_power = logC

#Calculamos el espectrograma normalizado (entre 0 y 1.)
min_power = np.amin(chroma_spectral_power)
max_power = np.amax(chroma_spectral_power)
power_range = max_power - min_power
chroma_spectral_p_normalised = (chroma_spectral_power - min_power)/power_range

#Se enfatizan los componentes que se extienden horizontalmente en el tiempo
kernel_len = 4
kernel = np.ones((1, kernel_len))
chroma_spectral_input = ndimage.convolve(chroma_spectral_p_normalised, kernel)
chroma_spectral_input[chroma_spectral_input < 0.7*kernel_len] = 0

plt.figure()
plt.imshow(chroma_spectral_input, aspect='auto', origin='lower')
plt.savefig('images/%s_chroma_cq_spectral_input.png' % input_name)

dt = 100 * b2.ms
sound_input = b2.TimedArray(chroma_spectral_input.T, dt=dt)

eqs = '''
dv/dt = (I-v)/(10*ms) : 1
I = sound_input(t, i): 1
'''

# SE crea un grupo de neuronas (513) que comparten las mismas ecuaciones que 
# definen sus propiedades
anf = b2.NeuronGroup(N=72, model=eqs, reset='v=0', threshold='v>1')

#Registramos los picos generados 
m = b2.SpikeMonitor(anf)
print("Building and running simulation...")
b2.run(time * b2.second, report='stdout')
print("Done!")

print("Writing spike files...")
indices = np.array(m.i)
times = np.array(m.t)
pickle_file = 'spikes_inputs/' +  'chroma_' + input_name + '.pickle'
with open(pickle_file, 'wb') as f:
    pickle.dump((times, indices), f)
print("done!")

#Picos repetidos de la misma neurona generaran lineas horizontales

plt.figure()
plt.plot(m.t/b2.second, m.i, 'k.', markersize=1)
plt.ylim([0, 72])
plt.ylabel('Numero de neuronas aferentes')
plt.xlabel('Tiempo (s)')
plt.savefig('images/chroma_cq_%s_spikes.png' % input_name)


#Para poder traducir las entradas de audio a picos una opción es hacer una TF del audio
#(o mas preciso calcular la densidad espectral de potencia) y usar la potencia 
#instantanea para controlar las tasas de disparo de un numero de neuronas a la entrada
