
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
# procesado del audio mediante el uso del espectrograma.

parser = argparse.ArgumentParser()
parser.add_argument('wav_file')
parser.add_argument('--interactive', action='store_true')
args = parser.parse_args()

filename = args.wav_file
name = os.path.basename(filename).replace(".wav", "")

audio = b2h.loadsound(filename) #Cargamos el audio con Brian2

if args.interactive:
    plt.ion()

# Primer procesado evaluado empleando el espectrograma, con NFFT = 1024
# Relacion entre NFFT y noverlap (que es el numero de puntos de superposicion entre bloques):
# Hay 44100 muestras/s (Fs), tenenemos una trama de tamaño 1024 y queremos que se desplace 10 ms.
# ((1024-x)/44100 = 10 ms) -> solapamiento (NOVERLAP) de 583

plt.figure()
(pxx, freqs, bins, im) = pylab.specgram(x=audio[:, 0].flatten(), NFFT=1024, Fs=audio.samplerate, noverlap = 583)

num_freqs = len(freqs)
dt = (bins[1] - bins[0]) #diferencial de tiempo entre los instantes 1 y 0

# Generamos espectrograma del audio de entrada 
# Las frecuencias van de 0 a 22 kHz en saltos de 42,88Hz aprox
# El número total de frecuencias usadas es 513, numero de neuronas que se inicializará
# Frecuencia de Muestreo a 44.1kHz

plt.ylabel('Frecuencia (Hz)')
plt.xlabel('Tiempo (s)')
plt.title('Espectrograma')
plt.savefig('images/%s_spectrogram.png' % name)

# Se pasa a dB y se calcula el espectrograma normalizado (entre 0 y 1)

spectro_pw = 10 * np.log10(pxx)
pw_min = np.amin(spectro_pw)
pw_max = np.amax(spectro_pw)
pw_range = pw_max - pw_min
spectral_pw_norm = (spectro_pw - pw_min)/pw_range

# Se enfatizan los componentes que se extienden horizontalmente en el tiempo (kernel)
# Se hace la convolución para eliminar los transitorios breves al comienzo.

len = 4
kernel = np.ones((1, len))
spectr_in = ndimage.convolve(spectral_pw_norm, kernel)
spectr_in[spectr_in < 0.8*len] = 0

plt.figure()
plt.imshow(spectr_in, aspect='auto', origin='lower')
plt.savefig('images/%s_spectral_input.png' % name)

# Ahora se traducen los audios de entrada en una serie de picos mediante Brian2 y se obtiene un 
# tren de picos de 513 neuronas de entrada que reflejan el contenido espectral de cada nota

audio_input = b2.TimedArray(spectr_in.T, dt=dt)

eqs_model = '''
dv/dt = (I-v)/(10*ms) : 1
I = audio_input(t, i): 1
'''

# Se crea el grupo de neuronas (513) que comparten las mismas ecuaciones que definen sus propiedades
# dt: paso de tiempo que se emplea para la simulación

neuronG = b2.NeuronGroup(N=num_freqs, model=eqs_model, reset='v=0', threshold='v>1', dt = dt/10)

#Registramos los picos generados con el SpikeMOnitor de Brian2

spikeR = b2.SpikeMonitor(neuronG)

print("Construyendo y ejecutando la simulación...")
b2.run(audio.duration, report='stdout')
print("¡Listo!")

print("Escribiendo los archivos de los spikes...")
inds = np.array(spikeR.i)
t = np.array(spikeR.t)
#poner ruta a la carpeta donde se quiera guardar
pickle_file = 'spikes_inputs_validation/' + name + '.pickle' 
with open(pickle_file, 'wb') as f:
    pickle.dump((t, inds), f)
print("¡Listo!")

plt.figure()
plt.plot(spikeR.t/b2.second, spikeR.i, 'k.', markersize=1)
plt.ylim([0, num_freqs])
plt.ylabel('Numero de neuronas aferentes')
plt.xlabel('Tiempo (s)')
plt.title('Spike Monitor de neuronas de entrada')
plt.savefig('images/spectrogram_%s_spikes.png' % name)
