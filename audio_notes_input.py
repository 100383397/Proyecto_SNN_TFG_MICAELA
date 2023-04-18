
import os
from mingus.midi import fluidsynth
from mingus.containers import Note

# Este script se emplea para crear los archivos de audio con los que se va a trabajar, y se va
# a entrenar y a validar la SNN. Con este script y el archivo que genera los acordes (llamado 
# audio_chords_input)se crearán una serie de archivos que constituirán una base de datos única 
# y propia para realizar los experimentos a los que va destinado este proyecto de fin de grado. 
# Los arhivos  generados de tipo .wav de secuencias de notas se almacenarán en la carpeta denominada
# audios

#Para entrenar la red emplearemos una octava completa, con los 12 sonidos que la componen (12 semitonos)
#Y para realizar la validación presentaremos diferentes audios exponiendo distintos casos.
music_seq = {}
music_seq['scale'] = [ 'C-4','C#-4','D-4','D#-4','E-4','F-4','F#-4','G-4','G#-4','A-4','A#-4','B-4']
#music_seq['3_notes'] = [ 'C-4','E-4','G-4']
#music_seq['4_notes'] = ['C-4','G-4', 'D-4', 'F#-4']
#music_seq['melody'] = ['Ab-5','G-5', 'F-5', 'G-5', 'Eb-5', 'F-5', 'C-5', 'Bb-4', 'C-5', 'C-6']

#Separación entre las notas para la realización de experimentos con distintas separaciones
s_separation = [0.5, 1, 2] 

#Inicializamos el instrumento mediante archivo .sf2 con el que se va a trabajar, en este caso es "Piano"
fluidsynth.init(sf2='FluidR3 GM2-2.SF2')
#fluidsynth.init(sf2='Guitar Acoustic.sf2')

#Definimos la función para iniciar la consecución de notas definidas previamente

def play_notes_seq(seq, time_seconds):
    for note_i in seq:
        note = Note(note_i)
        fluidsynth.play_Note(note)
        fluidsynth.midi.sleep(seconds=time_seconds)
        fluidsynth.stop_Note(note)

# Generamos los audios y los guardamos, el número contenido en el for final es el número de repeticiones
# de la secuencia de notas que hemos definido previamente
n = 1
for seq in music_seq:
    for time_seconds in s_separation:
        aux_name_file = '/tmp/temp%d.wav' % n
        fluidsynth.midi.start_recording(aux_name_file)
        for _ in range(5):
            play_notes_seq(music_seq[seq], time_seconds)

        final_filename = 'audios/%s_%.1f_s.wav' % (seq, time_seconds)
       
        # colocacion en el canal derecho y mezcla a 44100 
        os.system("sox %s %s remix 1 rate 44100" % (aux_name_file, final_filename))
        n += 1


