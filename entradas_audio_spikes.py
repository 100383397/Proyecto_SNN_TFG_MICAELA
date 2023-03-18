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

#Se cargan los archivos de audio que se han creado y grabado con Mingus.

parser = argparse.ArgumentParser()
parser.add_argument('wav_file')
parser.add_argument('--interactive', action='store_true')
args = parser.parse_args()

filename = args.wav_file
name = os.path.basename(filename).replace(".wav", "")

audio = b2h.loadsound(filename)

if args.interactive:
    plt.ion()

plt.figure()
(pxx, freqs, bins, im) = pylab.specgram(x=audio[:, 0].flatten(), NFFT=1024, Fs=audio.samplerate, noverlap = 864)
num_freqs = len(freqs)

#Relacion entre NFFT y noverlap (-> numero de puntos de superposicion entre bloques)
#Tenemos 16000 muestras/s
#Tengo una trama de tamaño 1024 y queremos que se desplace 10 ms ((1024-x)/16000 = 10 ms) -> solapamiento (NOVERLAP) de 864

#Generamos espectrograma del audio de entrada (A-5 TIENE UNA FREC DE 880 Hz)
#las frecuencias van de 0 a 8 kHz en saltos de 10Hz aprox (aumenta si Fs sube)
#el número total de frecuencias usadas es 513, numero de neuronas que se inicializará
#Frecuencia de Muestreo a 16kHz

plt.ylabel('Frecuencia (Hz)')
plt.xlabel('Tiempo (s)')
plt.savefig('images/%s_spectrogram.png' % name)

spectro_pw = 10 * np.log10(pxx) #Pasamos a dB

#Calculamos el espectrograma normalizado (entre 0 y 1.)
pw_min = np.amin(spectro_pw)
pw_max = np.amax(spectro_pw)
pw_amplitude = pw_max - pw_min
spectral_pw_norm = (spectro_pw - pw_min)/pw_amplitude

#Se enfatizan los componentes que se extienden horizontalmente en el tiempo
kernel_len = 4
kernel = np.ones((1, kernel_len))
spectral_input = ndimage.convolve(spectral_pw_norm, kernel)
spectral_input[spectral_input < 0.8*kernel_len] = 0

plt.figure()
plt.imshow(spectral_input, aspect='auto', origin='lower')
plt.savefig('images/%s_spectral_input.png' % name)

#Ahora se traducen los audios de entrada en una serie de picos mediante Brian2
# y se obtiene un tren de picos de 513 neuronas de entrada que reflejan el 
#contenido espectral de cada nota

dt = (bins[1] - bins[0])

print(bins)
sound_input = b2.TimedArray(spectral_input.T, dt=dt)

eqs = '''
dv/dt = (I-v)/(10*ms) : 1
I = sound_input(t, i): 1
'''

# SE crea un grupo de neuronas (513) que comparten las mismas ecuaciones que 
# definen sus propiedades
#dt: paso de tiempo que se emplea para la simulación
neuronG = b2.NeuronGroup(N=num_freqs, model=eqs, reset='v=0', threshold='v>1', dt = dt/10)

#Registramos los picos generados 
m = b2.SpikeMonitor(neuronG)

print("Construyendo y ejecutando la simulación...")
b2.run(audio.duration, report='stdout')
print("Listo!")

print("Escribiendo los archivos de los spikes...")
indices = np.array(m.i)
times = np.array(m.t)
pickle_file = 'spikes_inputs/' + name + '.pickle'
with open(pickle_file, 'wb') as f:
    pickle.dump((times, indices), f)
print("Listo!")

#Picos repetidos de la misma neurona generaran lineas horizontales

plt.figure()
plt.plot(m.t/b2.second, m.i, 'k.', markersize=1)
plt.ylim([0, num_freqs])
plt.ylabel('Numero de neuronas aferentes')
plt.xlabel('Tiempo (s)')
plt.savefig('images/spectrogram_%s_spikes.png' % name)

#Para poder traducir las entradas de audio a picos una opción es hacer una TF del audio
#(o mas preciso calcular la densidad espectral de potencia) y usar la potencia 
#instantanea para controlar las tasas de disparo de un numero de neuronas a la entrada