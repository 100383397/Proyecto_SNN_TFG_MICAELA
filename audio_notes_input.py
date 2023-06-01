
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
#ESCALA 12 TONOS 
#music_seq['scale12'] = [ 'C-4','C#-4','D-4','D#-4','E-4','F-4','F#-4','G-4','G#-4','A-4','A#-4','B-4']
#ESCALA 7 NOTAS 
#music_seq['scale7oct5'] = [ 'C-5','D-5','E-5','F-5','G-5','A-5','B-5']
#MELODIA BASICA 1: ARPEGIOS
#music_seq['melody1oct2'] = ['G-2','E-2', 'C-2', 'A-2', 'F-2', 'D-2'] 
#MELODIA BASICA 2: PRINCIPIO VERSIONADO FUR ELISE
#music_seq['melody2oct6'] = ['G-6','F#-6','G-6', 'F#-6', 'G-6', 'D-6', 'F-6', 'D#-6', 'C-6']
#MELODIA 3: INVENCION
#music_seq['melody3Violin'] = [ 'C-4','A#-4','G#-4','E-4','F-4','G#-4','G-4','D-4','D#-4','G-4','F-4','C-4','D-4','B-4','B-4']
#MELODIA 4: PRINCIPIO VERSIONADO SWAN LAKE
#music_seq['melody4oct3'] = [ 'B-3','B-3','E-3','F#-3','G-3', 'A-3', 'B-3', 'G-3', 'B-3','E-3','G-3','E-3','C-3','G-3','E-3','E-3']
#MELODIA 5: PRIMEROS COMPASES INTRODUCCION TITANIC
#music_seq['melody5Violin'] = ['C-4','D-4','D-4','E-4','E-4','D-4','C-4','D-4','G-4','G-4','E-4','G-4','A-4','A-4','A-4','G-4','D-4','D-4','D-4']
#MELODIA 6 MINUET G MAJOR BACH
#music_seq['Minuet'] = ['D-5','D-5','G-4','A-4','B-4','C-5','D-5','D-5','G-4','G-4','E-5','E-5','C-5','D-5','E-5','F#-5','G-5','G-5',
#                      'G-4','G-4','C-5','C-5','D-5','C-5','B-4','A-4','B-4','B-4','C-5','B-4','A-4','G-4','A-4','A-4','B-4','A-4','G-4','F#-4','G-4','G-4']
#MELODIA 7 G MINOR BACH
#music_seq['BachGminor'] = ['A#-4','D-4','C-4','D-4','A#-3','D-4','C-4','D-4','A#-4','D-4','C-4','D-4','A#-3','D-4','C-4','D-4', 'A#-4','D#-4','D-4','D#-4','A#-3','D#-4','D-4','D#-4',
#                          'A#-4','D#-4','D-4','D#-4','A#-3','D#-4','D-4','D#-4','A-4','C-4','A#-3','C-4','A-3','C-4','A#-3','C-4','A-4','C-4','A#-3','C-4','A-3','C-4','A#-3','C-4',
#                           'A-4','D-4','C-4','D-4','A#-3','D-4','C-4','D-4','A-4','D-4','C-4','D-4','A#-3','D-4','C-4','D-4',]
#MELODIA 8 WALTZ CHOPIN
#music_seq['Waltz1'] = ['E-4','A-4','B-4','C-5','C-5','D-5','E-5','F-5','F-5','B-4','C-5','D-5','A-5','G-5','F-5','E-5', 'D#-5','E-5','E-5','C-5','D-5','E-5','E-5','F-5','G-5',
#                       'A-5','A-5','G-5','G-5','F#-5','G-5', 'D-6', 'F-5', 'E-5','E-5']
#MELODIA 9 SONATINA CLEMENTI
#music_seq['Sonatine'] = ['C-4','C-4','E-4','C-4','G-3','G-3','G-3','G-3','C-4','C-4','E-4','C-4','G-3','G-3','G-3','G-4','F-4','E-4','D-4','C-4','B-3','C-4','B-3','C-4','D-4','C-4','B-3',
#                         'A-3','G-3','G-3', 'G-3']
#MELODIA 10 RIVER FLOWS IN YOU
music_seq['Yiruma'] = ['A-5','B-5','A-5','G#-5','A-5','A-4','E-5','A-4','A-5','B-5','A-5','G#-5','A-5','A-4','E-5','A-4', 'A-5','B-5','A-5','G#-5','A-5','B-5','C#-6','D-6', 'E-6', 'C#-6','B-5',
                       'A-5','G#-5', 'E-5','E-5','B-4','B-4','G#-4','G#-4']


#Separación entre las notas para la realización de experimentos con distintas separaciones y tambien
#es el tiempo que se mantiene la nota sostenida (ver función play_notes_seq)
s_separation = [0.5, 1, 2] 

#Inicializamos el instrumento mediante archivo .sf2 con el que se va a trabajar, en este caso es "Piano"
fluidsynth.init(sf2='soundfonts/FluidR3 GM2-2.SF2')
#fluidsynth.init(sf2='soundfonts/Guitar Acoustic.sf2')
#fluidsynth.init(sf2='soundfonts/Clarinete.sf2')
#fluidsynth.init(sf2='soundfonts/trumpet_collection.sf2')
#fluidsynth.init(sf2='soundfonts/Violin.SF2')
#Definimos la función para iniciar la consecución de notas definidas previamente

def play_notes_seq(seq, time_seconds):
    for note_i in seq:
        note = Note(note_i)
        fluidsynth.play_Note(note)
        fluidsynth.midi.sleep(seconds=time_seconds) #tiempo de la nota sostenida
        fluidsynth.stop_Note(note) #detiene la reproducción de la nota para que no se solape con la siguiente

# Generamos los audios y los guardamos, el número contenido en el for final es el número de repeticiones
# de la secuencia de notas que hemos definido previamente
n = 1
for seq in music_seq:
    for time_seconds in s_separation:
        aux_name_file = '/tmp/temp%d.wav' % n
        fluidsynth.midi.start_recording(aux_name_file)
        for _ in range(1):
            play_notes_seq(music_seq[seq], time_seconds)

        final_filename = 'audios_validation/%s_%.1f_s.wav' % (seq, time_seconds)
       
        # colocacion en el canal derecho y mezcla a 44100 
        os.system("sox %s %s remix 1 rate 44100" % (aux_name_file, final_filename))
        n += 1


