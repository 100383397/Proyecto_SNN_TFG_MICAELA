import os
from mingus.midi import fluidsynth
from mingus.containers import Note

#Este script se emplea para crear los archivos de audio con los que se va a trabajar
#con la SNN. Se generan arhivos de tipo .wav de secuencias de notas que se almacenan
#en la carpeta audios

music_seq = {}
music_seq['1_note'] = ['A-5']
music_seq['2_notes'] = ['E-3','G-3']
music_seq['scale1'] = [ 'C-4','D-4','E-4','F-4','G-4','A-4','B-4']
music_seq['scale2'] = [ 'C#-4','D#-4','F#-4','G#-4','A#-4',]
music_seq['scale3'] = [ 'C-5','C#-5','D-5','D#-5','E-5','F-5','F#-5','G-5','G#-5','A-5','A#-5','B-5']

s_separation = [0.5, 1, 2]

fluidsynth.init(sf2='FluidR3 GM2-2.SF2')

def play_notes_seq(seq, time_seconds):
    for note_i in seq:
        note = Note(note_i)
        fluidsynth.play_Note(note)
        fluidsynth.midi.sleep(seconds=time_seconds)
        fluidsynth.stop_Note(note)

# Genero audios y los guardo metiendo como entrada las notas con distintas duraciones

n = 1
for seq in music_seq:
    for time_seconds in s_separation:
        temp_filename = '/tmp/temp%d.wav' % n
        fluidsynth.midi.start_recording(temp_filename)
        for _ in range(5):
            play_notes_seq(music_seq[seq], time_seconds)

        final_filename = 'audios/%s_%.1f_s.wav' % (seq, time_seconds)
       
        # colocacion en el canal derecho y mezcla a 16 kHz 
        os.system("sox %s %s remix 1 rate 16000" % (temp_filename, final_filename))
        n += 1


