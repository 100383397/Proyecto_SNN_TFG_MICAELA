
import os
from mingus.midi import fluidsynth
from mingus.containers import NoteContainer 

# Este script se emplea para crear los archivos de audio con los que se va a trabajar y se va
# a entrenar y a validar la SNN. Con este script y el archivo que genera las melodías (llamado 
# audio_notes_input) se crearán una serie de archivos que constituirán una base de datos única 
# y propia para realizar los experimentos a los que va destinado este proyecto de fin de grado. 
# Los arhivos de audio generados, de tipo .wav, se almacenarán en las carpetas denominadas
# audios_train y audios_validation.

# A continuación se presentan las secuencias de acordes generadas para este proyecto

# Secuencia de acordes 0
chord = {}
chord['chord1'] = NoteContainer(['C-4','E-4'])
chord['chord2'] = NoteContainer(['D-4','F-4'])
chord['chord3'] = NoteContainer(['E-4','B-4'])
chord['chord4'] = NoteContainer(['F-4','A-4'])
chord['chord5'] = NoteContainer(['C-4','E-4'])
# Secuencia de acordes 1
chord = {}
chord['chord1'] = NoteContainer(['C-4','E-4'])
chord['chord2'] = NoteContainer(['D-4','F-4'])
chord['chord3'] = NoteContainer(['F-4','A-4'])
chord['chord4'] = NoteContainer(['E-4','G-4'])
# Secuencia de acordes 2
chord = {}
chord['chord1'] = NoteContainer(['F-4','G#-4'])
chord['chord2'] = NoteContainer(['G-4','A#-4'])
chord['chord3'] = NoteContainer(['E-4','G-4'])
chord['chord4'] = NoteContainer(['C-4','G#-4'])
chord['chord5'] = NoteContainer(['C-4','G-4'])
chord['chord6'] = NoteContainer(['C-4','F-4'])
# Secuencia de acordes 3
chord = {}
chord['chord1'] = NoteContainer(['G-4','D-5'])
chord['chord2'] = NoteContainer(['G-4','B-4'])
chord['chord3'] = NoteContainer(['E-4','C-5'])
chord['chord4'] = NoteContainer(['F-4','A-4'])
chord['chord5'] = NoteContainer(['F-4','G-4'])
chord['chord6'] = NoteContainer(['D-4','F-4'])
chord['chord7'] = NoteContainer(['C-4','E-4'])
chord['chord8'] = NoteContainer(['E-4','G-4'])
chord['chord9'] = NoteContainer(['C-4','C-5'])
# Secuencia de acordes 4
chord = {}
chord['chord1'] = NoteContainer(['D-4','F#-4','A-4'])
chord['chord2'] = NoteContainer(['D-4','G-4','B-4'])
chord['chord3'] = NoteContainer(['E-4','G-4','B-4'])
chord['chord4'] = NoteContainer(['E-4','G-4','A-4','C#-5'])
chord['chord5'] = NoteContainer(['D#-4','F#-4','B-4'])
chord['chord6'] = NoteContainer(['B-3','D#-4','F#-4'])
chord['chord7'] = NoteContainer(['B-3','E-4','G-4'])
# Secuencia de acordes 5
chord = {}
chord['chord1'] = NoteContainer(['A-3','C-4','E-4'])
chord['chord2'] = NoteContainer(['C-4','E-4','G-4'])
chord['chord3'] = NoteContainer(['B-3','D-4','F-4'])
chord['chord4'] = NoteContainer(['C-4','F-4','A-4'])
chord['chord6'] = NoteContainer(['B-3','D-4','G#-4'])
chord['chord7'] = NoteContainer(['C-4','E-4','A-4'])

# Separación entre los acordes para la realización de este experimento con distintos tiempos. 
# Estos tiempos son, también, el tiempo que se mantienes las notas del acorde sostenidas.

s_separation = [0.5, 1, 2]

fluidsynth.init(sf2='soundfonts/FluidR3 GM2-2.SF2') # Archivo .sf2 para grabar acordes a piano

def play_chord_seq(chord, time_seconds):
    for chord_i in chord:
        fluidsynth.play_NoteContainer(chord[chord_i])
        fluidsynth.midi.sleep(seconds=time_seconds)
        fluidsynth.stop_NoteContainer(chord[chord_i])

# Se generan los audios y se guardan. El número contenido en el "for" final es el número de veces que se quiere
# repetir la secuencia de acordes en el mismo audio.

temp_n = 1
for time_seconds in s_separation:
    aux_name_file = '/tmp/temp%d.wav' % temp_n
    fluidsynth.midi.start_recording(aux_name_file)
    for _ in range(1):
        play_chord_seq(chord, time_seconds)

    final_filename = 'audios_validation/chords_%.1f_s.wav' % time_seconds
    
    # colocacion en el canal derecho y mezcla a 44100
    os.system("sox %s %s remix 1 rate 44100" % (aux_name_file, final_filename))
    temp_n += 1
