
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
# preprocesado del audio mediante el uso del cromagrama de Q constante.

parser = argparse.ArgumentParser()
parser.add_argument('wav_file')
parser.add_argument('--interactive', action='store_true')
args = parser.parse_args()

filename = args.wav_file
name = os.path.basename(filename).replace(".wav", "")

#Cargamos el audio haciendo uso de librosa.

audio, sr = librosa.load(filename, sr= 16000)
time= librosa.get_duration(y=audio, sr=sr)
bins = librosa.times_like(audio)

dt = (bins[1] - bins[0])* b2.second
fmin = librosa.midi_to_hz(36)
hop_length = 2048
n_bins = 72

C =  np.abs(librosa.cqt(audio, sr=sr, n_bins= n_bins, fmin = fmin,  hop_length=hop_length))
logC = librosa.amplitude_to_db(C, ref=np.max) # Se pasa a dB

if args.interactive:
    plt.ion()

plt.figure(figsize=(15, 5))
librosa.display.specshow(logC, sr=sr, x_axis='time', y_axis='cqt_hz')
plt.ylabel('Frecuencia (Hz)')
plt.xlabel('Tiempo (s)')
plt.colorbar(format='%+2.0f dB')
plt.title('Cromagrama de Q constante')
plt.tight_layout()
plt.savefig('images/%s_chroma_cq.png' % name)

#Calculamos el cromagrama normalizado (entre 0 y 1)

pw_min = np.amin(logC)
pw_max = np.amax(logC)
pw_range = pw_max - pw_min
chroma_spectral_p_norm = (logC - pw_min)/pw_range

# Se enfatizan los componentes que se extienden horizontalmente en el tiempo (kernel)
# Se hace la convolución para eliminar los transitorios breves al comienzo de cada nota.

kernel_len = 5
kernel = np.ones((1, kernel_len))
chroma_spectr_in = ndimage.convolve(chroma_spectral_p_norm, kernel)
chroma_spectr_in[chroma_spectr_in < 0.65*kernel_len] = 0

plt.figure()
plt.imshow(chroma_spectr_in, aspect='auto', origin='lower')
plt.savefig('images/%s_chroma_cq_spectral_input.png' % name)

# Ahora se traducen los audios de entrada en una serie de picos mediante Brian2

audio_input = b2.TimedArray(chroma_spectr_in.T, dt=dt)

eqs_model = '''
dv/dt = (I-v)/(10*ms) : 1
I = audio_input(t, i): 1
'''

# Se crea un grupo de neuronas que comparten las mismas ecuaciones que definen sus propiedades
# dt: paso de tiempo que se emplea para la simulación

neuronG = b2.NeuronGroup(N=n_bins, model=eqs_model, reset='v=0', threshold='v>1', dt = dt/1000)

#Registramos los picos generados con el SpikeMonitor de Brian2

spikeR = b2.SpikeMonitor(neuronG)

print("Construyendo y ejecutando la simulación...")
b2.run(time * b2.second, report='stdout')
print("¡Listo!")

print("Escribiendo los archivos de los spikes...")
inds = np.array(spikeR.i)
times = np.array(spikeR.t)
#poner ruta a la carpeta donde se quiera guardar
pickle_file = 'spikes_inputs_train/' +  'chroma_' + name + '.pickle'
with open(pickle_file, 'wb') as f:
    pickle.dump((times, inds), f)
print("¡Listo!")

plt.figure()
plt.plot(spikeR.t/b2.second, spikeR.i, 'k.', markersize=1)
plt.ylim([0, n_bins])
plt.ylabel('Numero de neuronas aferentes')
plt.xlabel('Tiempo (s)')
plt.title('Spike Monitor de neuronas de entrada')
plt.savefig('images/chroma_cq_%s_spikes.png' % name)
