##!/usr/bin/env python

from __future__ import print_function, division
import brian2 as b2
import brian2hears as b2h
import matplotlib.pyplot as plt
import numpy as np
import os.path
import pickle
import pylab
import argparse
from scipy import ndimage

#Cargo los archivos de audio que he creado y grabado con Mingus

parser = argparse.ArgumentParser()
parser.add_argument('wav_file')
parser.add_argument('--interactive', action='store_true')
args = parser.parse_args()

input_filename = args.wav_file
input_name = os.path.basename(input_filename).replace(".wav", "")

sound = b2h.loadsound(input_filename)

if args.interactive:
    plt.ion()

plt.figure()
(pxx, freqs, bins, im) = \
        pylab.specgram(x=sound[:, 0].flatten(), NFFT=1024, Fs=sound.samplerate)
n_freqs = len(freqs)

#Generamos espectrograma del audio de entrada (A-5 TIENE UNA FREC DE 880 Hz)
print(freqs) #las frecuencias van de 0 a 5 kHz en saltos de 10Hz aprox (aumenta si Fs sube)
print(n_freqs) #el número total de frecuencias usadas es 513
print(sound.samplerate) #Frecuencia de Muestreo a 10kHz

plt.ylabel('Frecuencia (Hz)')
plt.xlabel('Tiempo (s)')
plt.savefig('images/%s_spectrogram.png' % input_name)
spectral_power = 10 * np.log10(pxx)

#Calculamos el espectrograma normalizado (entre 0 y 1.)
min_power = np.amin(spectral_power)
max_power = np.amax(spectral_power)
power_range = max_power - min_power
spectral_power_normalised = (spectral_power - min_power)/power_range

#Se enfatizan los componentes que se extienden horizontalmente en el tiempo
kernel_len = 4
kernel = np.ones((1, kernel_len))
spectral_input = ndimage.convolve(spectral_power_normalised, kernel)
spectral_input[spectral_input < 0.7*kernel_len] = 0

plt.figure()
plt.imshow(spectral_input, aspect='auto', origin='lower')
plt.savefig('images/%s_spectral_input.png' % input_name)

#Ahora se traducen los audios de entrada en una serie de picos mediante Brian2
# y se obtiene un tren de picos de 513 neuronas de entrada que reflejan el 
#contenido espectral de cada nota

dt = (bins[1] - bins[0])
#dt = 100 * b2.ms
print(spectral_input.T)
print(bins)
sound_input = b2.TimedArray(spectral_input.T, dt=dt)

eqs = '''
dv/dt = (I-v)/(10*ms) : 1
I = sound_input(t, i): 1
'''

# SE crea un grupo de neuronas (513) que comparten las mismas ecuaciones que 
# definen sus propiedades
anf = b2.NeuronGroup(N=n_freqs, model=eqs, reset='v=0', threshold='v>1')

#Registramos los picos generados 
m = b2.SpikeMonitor(anf)

print("Building and running simulation...")
b2.run(sound.duration, report='stdout')
print("Done!")

print("Writing spike files...")
indices = np.array(m.i)
times = np.array(m.t)
pickle_file = 'spikes_inputs/' + input_name + '.pickle'
with open(pickle_file, 'wb') as f:
    pickle.dump((times, indices), f)
print("done!")

#Picos repetidos de la misma neurona generaran lineas horizontales

plt.figure()
plt.plot(m.t/b2.second, m.i, 'k.', markersize=1)
plt.ylim([0, n_freqs])
plt.ylabel('Numero de neuronas aferentes')
plt.xlabel('Tiempo (s)')
plt.savefig('images/spectrogram_%s_spikes.png' % input_name)


#Para poder traducir las entradas de audio a picos una opción es hacer una TF del audio
#(o mas preciso calcular la densidad espectral de potencia) y usar la potencia 
#instantanea para controlar las tasas de disparo de un numero de neuronas a la entrada
