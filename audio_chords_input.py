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
chord['chord1'] = NoteContainer(['C-3', 'E-3'])
chord['chord2'] = NoteContainer(['D-3', 'F-3'])
chord['chord3'] = NoteContainer(['D-4', 'F-4'])

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
    temp_filename = '/tmp/temp%d.wav' % temp_n
    fluidsynth.midi.start_recording(temp_filename)
    for _ in range(3):
        play_chord_seq(chord, time_seconds)

    final_filename = 'audios/chords_%.1f_s.wav' % time_seconds
    
    # colocacion en el canal derecho y mezcla a 16 kHz 
    os.system("sox %s %s remix 1 rate 16000" % (temp_filename, final_filename))
    temp_n += 1
