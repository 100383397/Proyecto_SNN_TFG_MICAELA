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

# Se cargan los archivos de audio que se han creado y grabado con Mingus y se lleva a cabo el 
# preprocesado del audio.

parser = argparse.ArgumentParser()
parser.add_argument('wav_file')
parser.add_argument('--interactive', action='store_true')
args = parser.parse_args()

filename = args.wav_file
name = os.path.basename(filename).replace(".wav", "")

#Cargamos el audio haciendo uso de librosa y realizamos un espectrograma con la escala de Mel

audio, sr = librosa.load(filename, sr=16000)
time= librosa.get_duration(y=audio, sr=sr)
bins = librosa.times_like(audio)
dt = (bins[1] - bins[0])* b2.second

print(time)
print(bins)
print(dt)

n_mels = 512 
Nfft = 2048
hl= dt * sr/b2.second #modificando esto, el bins cambia y afecta al eje temporal de los spikes 372 (16k * 23,22ms)
hop_length = int(np.round(hl, 0))

print(hl)
print(hop_length)

S = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels= n_mels, n_fft= Nfft, hop_length=hop_length)
S_dB = librosa.power_to_db(S, ref=np.max)


if args.interactive:
    plt.ion()

plt.figure(figsize=(15, 5))

img = librosa.display.specshow(S_dB, x_axis='time', y_axis='mel', sr=sr)
plt.ylabel('Frecuencia (Hz)')
plt.xlabel('Tiempo (s)')
plt.colorbar(format='%+2.0f dB')
plt.title('Espectrograma de Mel')
plt.tight_layout()
plt.savefig('images/%s_melspectrogram.png' % name)

mel_spectral_pw = S_dB

#Calculamos el espectrograma normalizado (entre 0 y 1.)
pw_min = np.amin(mel_spectral_pw)
pw_max = np.amax(mel_spectral_pw)
pw_range = pw_max - pw_min
mel_spectral_pw_norm = (mel_spectral_pw - pw_min)/pw_range

#Se enfatizan los componentes que se extienden horizontalmente en el tiempo
len = 4
kernel = np.ones((1, len))
mel_spectral_input = ndimage.convolve(mel_spectral_pw_norm, kernel)
mel_spectral_input[mel_spectral_input < 0.57*len] = 0

plt.figure()
plt.imshow(mel_spectral_input, aspect='auto', origin='lower')
plt.savefig('images/%s_melspectrogram_spectral_input.png' % name)

audio_input = b2.TimedArray(mel_spectral_input.T, dt=dt)

eqs = '''
dv/dt = (I-v)/(10*ms) : 1
I = audio_input(t, i): 1
'''

# SE crea un grupo de neuronas que comparten las mismas ecuaciones que definen sus propiedades
neuronG = b2.NeuronGroup(N=n_mels, model=eqs, reset='v=0', threshold='v>1', dt = dt/100)

#Registramos los picos generados 
spikeR = b2.SpikeMonitor(neuronG)

print("Construyendo y ejecutando la simulación...")
b2.run(time * b2.second, report='stdout')
print("Listo!")

print("Escribiendo los archivos de los spikes...")
inds = np.array(spikeR.i)
times = np.array(spikeR.t)
pickle_file = 'spikes_inputs/' +  'melscale_' + name + '.pickle'
with open(pickle_file, 'wb') as f:
    pickle.dump((times, inds), f)
print("Listo!")

#Picos repetidos de la misma neurona generaran lineas horizontales

plt.figure()
plt.plot(spikeR.t/b2.second, spikeR.i, 'k.', markersize=1)
plt.ylim([0, n_mels])
plt.ylabel('Numero de neuronas aferentes')
plt.xlabel('Tiempo (s)')
plt.savefig('images/melscale_%s_spikes.png' % name)


#Para poder traducir las entradas de audio a picos una opción es hacer una TF del audio
#(o mas preciso calcular la densidad espectral de potencia) y usar la potencia 
#instantanea para controlar las tasas de disparo de un numero de neuronas a la entrada
