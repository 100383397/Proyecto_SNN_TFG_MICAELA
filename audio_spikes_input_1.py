
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

# Se cargan los archivos de audio que se han creado y grabado con Mingus y se lleva a cabo el 
# preprocesado del audio.

parser = argparse.ArgumentParser()
parser.add_argument('wav_file')
parser.add_argument('--interactive', action='store_true')
args = parser.parse_args()

filename = args.wav_file
name = os.path.basename(filename).replace(".wav", "")

audio = b2h.loadsound(filename) #Cargamos el audio con Brian2

if args.interactive:
    plt.ion()

# Primer preprocesado evaluado empleando el espectrograma, con NFFT = 1024
# Relacion entre NFFT y noverlap (-> numero de puntos de superposicion entre bloques)
# Tenemos 44100 muestras/s (Fs)
# Tenemos una trama de tamaño 1024 y queremos que se desplace 10 ms 
# ((1024-x)/44100 = 10 ms) -> solapamiento (NOVERLAP) de 583

plt.figure()
(pxx, freqs, bins, im) = pylab.specgram(x=audio[:, 0].flatten(), NFFT=1024, Fs=audio.samplerate, noverlap = 583)

num_freqs = len(freqs)
dt = (bins[1] - bins[0]) #diferencial de tiempo entre los instantes 1 y 0

# Generamos espectrograma del audio de entrada 
# Las frecuencias van de 0 a 22 kHz en saltos de 42,88Hz aprox
# el número total de frecuencias usadas es 513, numero de neuronas que se inicializará
# Frecuencia de Muestreo a 44.1kHz

#print(freqs)
#print(num_freqs)
#print(bins)
#print(dt)

plt.ylabel('Frecuencia (Hz)')
plt.xlabel('Tiempo (s)')
plt.title('Espectrograma')
plt.savefig('images/%s_spectrogram.png' % name)

spectro_pw = 10 * np.log10(pxx) #Pasamos a dB

#Calculamos el espectrograma normalizado (entre 0 y 1.)
pw_min = np.amin(spectro_pw)
pw_max = np.amax(spectro_pw)
pw_range = pw_max - pw_min
spectral_pw_norm = (spectro_pw - pw_min)/pw_range

#Se enfatizan los componentes que se extienden horizontalmente en el tiempo (kernel)
len = 4
kernel = np.ones((1, len))
spectral_input = ndimage.convolve(spectral_pw_norm, kernel)
spectral_input[spectral_input < 0.77*len] = 0

plt.figure()
plt.imshow(spectral_input, aspect='auto', origin='lower')
plt.savefig('images/%s_spectral_input.png' % name)

# Ahora se traducen los audios de entrada en una serie de picos mediante Brian2
# y se obtiene un tren de picos de 513 neuronas de entrada que reflejan el 
# contenido espectral de cada nota

audio_input = b2.TimedArray(spectral_input.T, dt=dt)

eqs_model = '''
dv/dt = (I-v)/(10*ms) : 1
I = audio_input(t, i): 1
'''

# Se crea un grupo de neuronas (513) que comparten las mismas ecuaciones que definen sus propiedades
# dt: paso de tiempo que se emplea para la simulación
neuronG = b2.NeuronGroup(N=num_freqs, model=eqs_model, reset='v=0', threshold='v>1', dt = dt/10)

#Registramos los picos generados 
spikeR = b2.SpikeMonitor(neuronG)

print("Construyendo y ejecutando la simulación...")
b2.run(audio.duration, report='stdout')
print("Listo!")

print("Escribiendo los archivos de los spikes...")
inds = np.array(spikeR.i)
times = np.array(spikeR.t)
pickle_file = 'spikes_inputs/' + name + '.pickle'
with open(pickle_file, 'wb') as f:
    pickle.dump((times, inds), f)
print("Listo!")

#Picos repetidos de la misma neurona generaran lineas horizontales

plt.figure()
plt.plot(spikeR.t/b2.second, spikeR.i, 'k.', markersize=1)
plt.ylim([0, num_freqs])
plt.ylabel('Numero de neuronas aferentes')
plt.xlabel('Tiempo (s)')
plt.title('Spike Monitor de neuronas de entrada')
plt.savefig('images/spectrogram_%s_spikes.png' % name)

#Para poder traducir las entradas de audio a picos una opción es hacer una TF del audio
#(o mas preciso calcular la densidad espectral de potencia) y usar la potencia 
#instantanea para controlar las tasas de disparo de un numero de neuronas a la entrada