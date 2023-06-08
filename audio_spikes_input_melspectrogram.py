
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
# preprocesado del audio mediante el uso del espectrograma en escala de Mel.

parser = argparse.ArgumentParser()
parser.add_argument('wav_file')
parser.add_argument('--interactive', action='store_true')
args = parser.parse_args()

filename = args.wav_file
name = os.path.basename(filename).replace(".wav", "")

#Cargamos el audio haciendo uso de librosa 

audio, sr = librosa.load(filename, sr=16000)
time= librosa.get_duration(y=audio, sr=sr)
bins = librosa.times_like(audio)

dt = (bins[1] - bins[0])* b2.second #diferencial de tiempo entre los instantes 1 y 0
n_mels = 512 
Nfft = 2048
hl= dt * sr/b2.second #Modificando este parámetro, el bins cambia y afecta al eje temporal de los spikes 372 (16k * 23,22ms)
hop_length = int(np.round(hl, 0))

S = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels= n_mels, n_fft= Nfft, hop_length=hop_length)

S_dB = librosa.power_to_db(S, ref=np.max) # Se pasa a dB

if args.interactive:
    plt.ion()

plt.figure(figsize=(15, 5))

img = librosa.display.specshow(S_dB, x_axis='time', y_axis='mel', sr=sr)
plt.ylabel('Frecuencias escala Mel (Hz)')
plt.xlabel('Tiempo (s)')
plt.colorbar(format='%+2.0f dB')
plt.title('Espectrograma a escala de Mel')
plt.tight_layout()
plt.savefig('images/%s_melspectrogram.png' % name)

#Calculamos el espectrograma en escala de mel normalizado (entre 0 y 1.)

pw_min = np.amin(S_dB)
pw_max = np.amax(S_dB)
pw_range = pw_max - pw_min
mel_spectral_pw_norm = (S_dB - pw_min)/pw_range

# Se enfatizan los componentes que se extienden horizontalmente en el tiempo (kernel)
# Se hace la convolución para eliminar los transitorios breves al comienzo de cada nota.

len = 4
kernel = np.ones((1, len))
mel_spectr_in = ndimage.convolve(mel_spectral_pw_norm, kernel)
mel_spectr_in[mel_spectr_in < 0.55*len] = 0

plt.figure()
plt.imshow(mel_spectr_in, aspect='auto', origin='lower')
plt.savefig('images/%s_melspectrogram_spectral_input.png' % name)

# Ahora se traducen los audios de entrada en una serie de picos mediante Brian2 y se obtiene un 
# tren de picos de 512 neuronas de entrada que reflejan el contenido espectral de cada nota

audio_input = b2.TimedArray(mel_spectr_in.T, dt=dt)

eqs_model = '''
dv/dt = (I-v)/(10*ms) : 1
I = audio_input(t, i): 1
'''

# Se crea el grupo de neuronas (512) que comparten las mismas ecuaciones que definen sus propiedades
# dt: paso de tiempo que se emplea para la simulación

neuronG = b2.NeuronGroup(N=n_mels, model=eqs_model, reset='v=0', threshold='v>1', dt = dt/1000)

#Registramos los picos generados con el SpikeMonitor de Brian2

spikeR = b2.SpikeMonitor(neuronG)

print("Construyendo y ejecutando la simulación...")
b2.run(time * b2.second, report='stdout')
print("¡Listo!")

print("Escribiendo los archivos de los spikes...")
inds = np.array(spikeR.i)
times = np.array(spikeR.t)
#poner ruta a la carpeta donde se quiera guardar
pickle_file = 'spikes_inputs_validation/' + name + '.pickle'
with open(pickle_file, 'wb') as f:
    pickle.dump((times, inds), f)
print("¡Listo!")

plt.figure()
plt.plot(spikeR.t/b2.second, spikeR.i, 'k.', markersize=1)
plt.ylim([0, n_mels])
plt.ylabel('Numero de neuronas aferentes')
plt.xlabel('Tiempo (s)')
plt.title('Spike Monitor de neuronas de entrada')
plt.savefig('images/melscale_%s_spikes.png' % name)
