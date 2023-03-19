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

filename = args.wav_file
name = os.path.basename(filename).replace(".wav", "")

#Cargamos el audio haciendo uso de librosa y realizamos un cronograma  de Q constante

audio, sr = librosa.load(filename, sr= 16000)
time= librosa.get_duration(y=audio, sr=sr)
bins = librosa.times_like(audio)
dt = (bins[1] - bins[0])* b2.second

print(time)
print(bins)
print(dt)  

fmin = librosa.midi_to_hz(36)
hop_length = 2048
n_bins = 72

print(hop_length)

C =  np.abs(librosa.cqt(audio, sr=sr, n_bins= n_bins, fmin = fmin,  hop_length=hop_length))
logC = librosa.amplitude_to_db(C, ref=np.max)

if args.interactive:
    plt.ion()

plt.figure(figsize=(15, 5))
librosa.display.specshow(logC, sr=sr, x_axis='time', y_axis='cqt_note')
plt.ylabel('CQT NOTE')
plt.xlabel('Tiempo (s)')
plt.colorbar(format='%+2.0f dB')
plt.title('Constant-Q power spectrum')
plt.tight_layout()
plt.savefig('images/%s_chroma_cq.png' % name)

chroma_spectral_pw = logC 

#Calculamos el espectrograma normalizado (entre 0 y 1.)
pw_min = np.amin(chroma_spectral_pw)
pw_max = np.amax(chroma_spectral_pw)
pw_range = pw_max - pw_min
chroma_spectral_p_norm = (chroma_spectral_pw - pw_min)/pw_range

#Se enfatizan los componentes que se extienden horizontalmente en el tiempo
kernel_len = 6
kernel = np.ones((1, kernel_len))
chroma_spectral_input = ndimage.convolve(chroma_spectral_p_norm, kernel)
chroma_spectral_input[chroma_spectral_input < 0.6*kernel_len] = 0

plt.figure()
plt.imshow(chroma_spectral_input, aspect='auto', origin='lower')
plt.savefig('images/%s_chroma_cq_spectral_input.png' % name)

audio_input = b2.TimedArray(chroma_spectral_input.T, dt=dt)

eqs = '''
dv/dt = (I-v)/(10*ms) : 1
I = audio_input(t, i): 1
'''

# SE crea un grupo de neuronas (513) que comparten las mismas ecuaciones que 
# definen sus propiedades
neuronG = b2.NeuronGroup(N=n_bins, model=eqs, reset='v=0', threshold='v>1', dt = dt/1000)

#Registramos los picos generados 
spikeR = b2.SpikeMonitor(neuronG)

print("Construyendo y ejecutando la simulación...")
b2.run(time * b2.second, report='stdout')
print("Listo!")

print("Escribiendo los archivos de los spikes...")
indices = np.array(spikeR.i)
times = np.array(spikeR.t)
pickle_file = 'spikes_inputs/' +  'chroma_' + name + '.pickle'
with open(pickle_file, 'wb') as f:
    pickle.dump((times, indices), f)
print("Listo!")

#Picos repetidos de la misma neurona generaran lineas horizontales

plt.figure()
plt.plot(spikeR.t/b2.second, spikeR.i, 'k.', markersize=1)
plt.ylim([0, n_bins])
plt.ylabel('Numero de neuronas aferentes')
plt.xlabel('Tiempo (s)')
plt.savefig('images/chroma_cq_%s_spikes.png' % name)


#Para poder traducir las entradas de audio a picos una opción es hacer una TF del audio
#(o mas preciso calcular la densidad espectral de potencia) y usar la potencia 
#instantanea para controlar las tasas de disparo de un numero de neuronas a la entrada
