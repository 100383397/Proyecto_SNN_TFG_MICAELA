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

#Cargamos el audio haciendo uso de librosa y realizamos un cronograma con la escala de Mel

x, sr = librosa.load(input_filename)
time= librosa.get_duration(y=x, sr=sr)
                     
S = librosa.feature.melspectrogram(y=x, sr=sr, n_mels=256,fmax=5000)

if args.interactive:
    plt.ion()

S_dB = librosa.power_to_db(S, ref=np.max)
plt.figure(figsize=(15, 5))

img = librosa.display.specshow(S_dB, x_axis='time', y_axis='mel', sr=sr, fmax=5000)
plt.ylabel('Frecuencia (Hz)')
plt.xlabel('Tiempo (s)')
plt.colorbar(format='%+2.0f dB')
plt.title('Constant-Q power spectrum')
plt.tight_layout()
plt.savefig('images/%s_melspectrogram.png' % input_name)

mel_spectral_power = S_dB

#Calculamos el espectrograma normalizado (entre 0 y 1.)
min_power = np.amin(mel_spectral_power)
max_power = np.amax(mel_spectral_power)
power_range = max_power - min_power
mel_spectral_p_normalised = (mel_spectral_power - min_power)/power_range

#Se enfatizan los componentes que se extienden horizontalmente en el tiempo
kernel_len = 4
kernel = np.ones((1, kernel_len))
mel_spectral_input = ndimage.convolve(mel_spectral_p_normalised, kernel)
mel_spectral_input[mel_spectral_input < 0.6*kernel_len] = 0

plt.figure()
plt.imshow(mel_spectral_input, aspect='auto', origin='lower')
plt.savefig('images/%s_melspectrogram_spectral_input.png' % input_name)

#dt = 30* b2.ms Este es el dt usado para two_notes_1.0_s
dt = 40 * b2.ms #Este es el dt usado para scale1_1.0_s a  enor dt aparecen mas secuencias de escalas
sound_input = b2.TimedArray(mel_spectral_input.T, dt=dt)

eqs = '''
dv/dt = (I-v)/(10*ms) : 1
I = sound_input(t, i): 1
'''

# SE crea un grupo de neuronas (513) que comparten las mismas ecuaciones que 
# definen sus propiedades
anf = b2.NeuronGroup(N=256, model=eqs, reset='v=0', threshold='v>1')

#Registramos los picos generados 
m = b2.SpikeMonitor(anf)
print("Building and running simulation...")
b2.run(time * b2.second, report='stdout')
print("Done!")

print("Writing spike files...")
indices = np.array(m.i)
times = np.array(m.t)
pickle_file = 'spikes_inputs/' +  'melscale_' + input_name + '.pickle'
with open(pickle_file, 'wb') as f:
    pickle.dump((times, indices), f)
print("done!")

#Picos repetidos de la misma neurona generaran lineas horizontales

plt.figure()
plt.plot(m.t/b2.second, m.i, 'k.', markersize=1)
plt.ylim([0, 256])
plt.ylabel('Numero de neuronas aferentes')
plt.xlabel('Tiempo (s)')
plt.savefig('images/melscale_%s_spikes.png' % input_name)


#Para poder traducir las entradas de audio a picos una opción es hacer una TF del audio
#(o mas preciso calcular la densidad espectral de potencia) y usar la potencia 
#instantanea para controlar las tasas de disparo de un numero de neuronas a la entrada
