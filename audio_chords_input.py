
import os
from mingus.midi import fluidsynth
from mingus.containers import NoteContainer 

# Este script se emplea para crear los archivos de audio con los que se va a trabajar, y se va
# a entrenar y a probar la SNN. Con este script y el archivo que genera los acordes (llamado 
# audio_chords_input)se crearán una serie de archivos que constituirán una base de datos única 
# y propia para realizar los experimentos a los que va destinado este proyecto de fin de grado. 
# Los arhivos  generados de tipo .wav de secuencias de acordes se almacenarán en la carpeta denominada
# audios

chord = {}
chord['chord1'] = NoteContainer(['C-4', 'E-4'])
chord['chord2'] = NoteContainer(['F-4', 'Ab-4'])
chord['chord3'] = NoteContainer(['D-4', 'F#-4'])
chord['chord4'] = NoteContainer(['G-4', 'Bb-4'])

s_separation = [0.5, 1, 2]

fluidsynth.init(sf2='FluidR3 GM2-2.SF2')

def play_chord_seq(chord, time_seconds):
    for chord_i in chord:
        fluidsynth.play_NoteContainer(chord[chord_i])
        fluidsynth.midi.sleep(seconds=time_seconds)
        fluidsynth.stop_NoteContainer(chord[chord_i])

# Generamos los audios y los guardamos
temp_n = 1
for time_seconds in s_separation:
    aux_name_file = '/tmp/temp%d.wav' % temp_n
    fluidsynth.midi.start_recording(aux_name_file)
    for _ in range(3):
        play_chord_seq(chord, time_seconds)

    final_filename = 'audios/chords_%.1f_s.wav' % time_seconds
    
    # colocacion en el canal derecho y mezcla a 44100
    os.system("sox %s %s remix 1 rate 44100" % (aux_name_file, final_filename))
    temp_n += 1
