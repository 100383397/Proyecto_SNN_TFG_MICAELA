import os
from mingus.midi import fluidsynth
from mingus.containers import Note

# Este script se emplea para crear los archivos de audio con los que se va a trabajar, y se va
# a entrenar y a probar la SNN. Con este script y el archivo que genera los acordes (llamado 
# audio_chords_input)se crearán una serie de archivos que constituirán una base de datos única 
# y propia para realizar los experimentos a los que va destinado este proyecto de fin de grado. 
# Los arhivos  generados de tipo .wav de secuencias de notas se almacenarán en la carpeta denominada
# audios

music_seq = {}
music_seq['1_note'] = ['A-4']
music_seq['2_notes'] = ['C-5','G-5']
music_seq['scale1'] = [ 'C-4','D-4','E-4','F-4','G-4','A-4','B-4']
music_seq['scale2'] = [ 'C#-3','D#-3','F#-3','G#-3','A#-3',]
music_seq['scale3'] = [ 'C-5','C#-5','D-5','D#-5','E-5','F-5','F#-5','G-5','G#-5','A-5','A#-5','B-5']

#separación entre las notas de cada secuancia para la realización de experimentos con distintas separaciones
s_separation = [0.5, 1, 2] 

#Inicializamos el instrumento mediante archivo .sf2 con el que se va a trabajar, en este caso es "Piano"
fluidsynth.init(sf2='FluidR3 GM2-2.SF2')

def play_notes_seq(seq, time_seconds):
    for note_i in seq:
        note = Note(note_i)
        fluidsynth.play_Note(note)
        fluidsynth.midi.sleep(seconds=time_seconds)
        fluidsynth.stop_Note(note)

# Generamos los audios y los guardamos
n = 1
for seq in music_seq:
    for time_seconds in s_separation:
        temp_filename = '/tmp/temp%d.wav' % n
        fluidsynth.midi.start_recording(temp_filename)
        for _ in range(3):
            play_notes_seq(music_seq[seq], time_seconds)

        final_filename = 'audios/%s_%.1f_s.wav' % (seq, time_seconds)
       
        # colocacion en el canal derecho y mezcla a 44100 
        os.system("sox %s %s remix 1 rate 44100" % (temp_filename, final_filename))
        n += 1


